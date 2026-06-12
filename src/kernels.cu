#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────
//  Embedding forward: out[b,d] = embed[ids[b], d]
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void embedding_fwd(
    const float* __restrict__ embed,
    const int*   __restrict__ ids,
    float*                    out,
    int E)
{
    int b = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (d >= E) return;
    out[b * E + d] = embed[ids[b] * E + d];
}

// ─────────────────────────────────────────────────────────────
//  Embedding backward: scatter-add (atomic)
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void embedding_bwd(
    const float* __restrict__ d_out,
    const int*   __restrict__ ids,
    float*                    d_embed,
    int E)
{
    int b = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (d >= E) return;
    atomicAdd(&d_embed[ids[b] * E + d], d_out[b * E + d]);
}

// ─────────────────────────────────────────────────────────────
//  Add bias in-place:  out[b,i] += bias[i]
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void add_bias(float* out, const float* bias, int batch, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / N;
    int i = idx % N;
    if (b >= batch || i >= N) return;
    out[b * N + i] += bias[i];
}

// ─────────────────────────────────────────────────────────────
//  FUSED LSTM forward — one warp per (batch, hidden) element
//
//  Input:  gates[b, 4*H]  (layout: i|f|o|g, each H wide)
//          c_prev[b, H]
//  Output: h_out[b, H],  c_out[b, H]
//
//  One thread = one (b,j) pair.
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void fused_lstm_fwd(
    const float* __restrict__ gates,   // [batch, 4H]
    const float* __restrict__ c_prev,  // [batch, H]
    float*                    h_out,   // [batch, H]
    float*                    c_out,   // [batch, H]
    int H)
{
    int b = blockIdx.x;
    int j = threadIdx.x + blockIdx.y * blockDim.x;
    if (j >= H) return;

    int base = b * 4 * H;
    float gi = gates[base +         j];
    float gf = gates[base +     H + j];
    float go = gates[base + 2 * H + j];
    float gg = gates[base + 3 * H + j];

    float i_g = __frcp_rn(1.0f + __expf(-gi));
    float f_g = __frcp_rn(1.0f + __expf(-gf));
    float o_g = __frcp_rn(1.0f + __expf(-go));
    float g_g = tanhf(gg);

    float c = f_g * c_prev[b * H + j] + i_g * g_g;
    float h = o_g * tanhf(c);

    c_out[b * H + j] = c;
    h_out[b * H + j] = h;
}

// ─────────────────────────────────────────────────────────────
//  FUSED LSTM backward
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void fused_lstm_bwd(
    const float* __restrict__ gates_raw,
    const float* __restrict__ c_prev,
    const float* __restrict__ c_cur,
    const float* __restrict__ d_h,
    const float* __restrict__ d_c_next,
    float*                    d_gates,
    float*                    d_c_prev,
    int H)
{
    int b = blockIdx.x;
    int j = threadIdx.x + blockIdx.y * blockDim.x;
    if (j >= H) return;

    int base = b * 4 * H;
    float gi = gates_raw[base +         j];
    float gf = gates_raw[base +     H + j];
    float go = gates_raw[base + 2 * H + j];
    float gg = gates_raw[base + 3 * H + j];

    float i_g = __frcp_rn(1.0f + __expf(-gi));
    float f_g = __frcp_rn(1.0f + __expf(-gf));
    float o_g = __frcp_rn(1.0f + __expf(-go));
    float g_g = tanhf(gg);

    float c  = c_cur[b * H + j];
    float tc = tanhf(c);

    float dc = d_h[b * H + j] * o_g * (1.0f - tc * tc) + d_c_next[b * H + j];

    float di = dc * g_g;
    float df = dc * c_prev[b * H + j];
    float do_ = d_h[b * H + j] * tc;
    float dg  = dc * i_g;

    d_gates[base +         j] = di  * i_g * (1.0f - i_g);
    d_gates[base +     H + j] = df  * f_g * (1.0f - f_g);
    d_gates[base + 2 * H + j] = do_ * o_g * (1.0f - o_g);
    d_gates[base + 3 * H + j] = dg  * (1.0f - g_g * g_g);

    d_c_prev[b * H + j] = dc * f_g;
}

// ─────────────────────────────────────────────────────────────
//  AdaptiveSoftmax: linear  C[b,o] = A[b,k]*W^T[o,k] + bias[o]
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void asm_linear(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ bias,
    float*                    C,
    int in_dim, int out_dim, int batch)
{
    int b = blockIdx.x;
    int o = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch || o >= out_dim) return;
    float s = bias[o];
    for (int k = 0; k < in_dim; k++) s += A[b * in_dim + k] * W[o * in_dim + k];
    C[b * out_dim + o] = s;
}

// ─────────────────────────────────────────────────────────────
//  Row-wise softmax in-place  x[batch, n]
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void asm_softmax(float* x, int n, int batch)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    float* row = x + b * n;
    float mx = row[0];
    for (int i = 1; i < n; i++) if (row[i] > mx) mx = row[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { row[i] = __expf(row[i] - mx); s += row[i]; }
    float inv = __frcp_rn(s);
    for (int i = 0; i < n; i++) row[i] *= inv;
}

// CE grad + accumulate loss
extern "C" __global__ void asm_ce_grad(
    float*       probs,        // in-place → becomes d_probs
    const int*   targets,      // -1 = masked
    float*       loss_out,     // accumulate (atomic add)
    int n, int offset, int batch)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    int t = targets[b];
    if (t < 0) return;
    int ti = t - offset;
    float p = probs[b * n + ti];
    atomicAdd(loss_out, -(p > 1e-30f ? __logf(p) : -30.0f));
    probs[b * n + ti] -= 1.0f;
}

// Weight grad: G_W[o,k] += sum_b d_out[b,o] * A[b,k]
extern "C" __global__ void asm_wgrad(
    const float* __restrict__ d_out,
    const float* __restrict__ A,
    float*                    G_W,
    int batch, int in_dim, int out_dim)
{
    int o = blockIdx.x;
    int k = threadIdx.x + blockIdx.y * blockDim.x;
    if (o >= out_dim || k >= in_dim) return;
    float s = 0.0f;
    for (int b = 0; b < batch; b++) s += d_out[b * out_dim + o] * A[b * in_dim + k];
    G_W[o * in_dim + k] += s;
}

// Bias grad: G_b[o] += sum_b d_out[b,o]
extern "C" __global__ void asm_bgrad(
    const float* __restrict__ d_out,
    float*                    G_b,
    int batch, int out_dim)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;
    float s = 0.0f;
    for (int b = 0; b < batch; b++) s += d_out[b * out_dim + o];
    G_b[o] += s;
}

// Input grad: d_A[b,k] += sum_o d_out[b,o] * W[o,k]
extern "C" __global__ void asm_igrad(
    const float* __restrict__ d_out,
    const float* __restrict__ W,
    float*                    d_A,
    int batch, int in_dim, int out_dim)
{
    int b = blockIdx.x;
    int k = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch || k >= in_dim) return;
    float s = 0.0f;
    for (int o = 0; o < out_dim; o++) s += d_out[b * out_dim + o] * W[o * in_dim + k];
    d_A[b * in_dim + k] += s;
}

// Reduce sum over batch dim: out[i] = sum_b x[b*N+i]
extern "C" __global__ void reduce_sum_batch(
    const float* __restrict__ x,
    float*                    out,
    int batch, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.0f;
    for (int b = 0; b < batch; b++) s += x[b * N + i];
    out[i] += s;
}

// ─────────────────────────────────────────────────────────────
//  Adam update
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void adam_update(
    float*       param,
    float*       m,
    float*       v,
    const float* grad,
    float lr, float b1, float b2, float eps, float bc1, float bc2,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = grad[i];
    float m_ = b1 * m[i] + (1.0f - b1) * g;
    float v_ = b2 * v[i] + (1.0f - b2) * g * g;
    m[i] = m_;
    v[i] = v_;
    param[i] -= lr * (m_ / bc1) / (__fsqrt_rn(v_ / bc2) + eps);
}

// ─────────────────────────────────────────────────────────────
//  GPU gradient clipping (two-pass, no CPU readback)
//  Pass 1: block-level reduction of ||grad||^2
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void norm_reduce(
    const float* __restrict__ grad,
    float*                    partial,
    int n)
{
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    shmem[tid] = (gid < n) ? grad[gid] * grad[gid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = shmem[0];
}

// Pass 2: reduce partial → scale if needed
extern "C" __global__ void clip_if_needed(
    const float* __restrict__ partial,
    float*                    grad,
    int partial_n, int grad_n, float clip_val)
{
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    shmem[tid] = (tid < partial_n) ? partial[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid != 0) return;
    float norm = __fsqrt_rn(shmem[0]);
    if (norm > clip_val) {
        float scale = clip_val / norm;
        for (int i = 0; i < grad_n; i++) grad[i] *= scale;
    }
}

// ─────────────────────────────────────────────────────────────
//  Accumulate mean loss on GPU (atomic, no CPU stall)
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void zero_float(float* x) { if (threadIdx.x == 0 && blockIdx.x == 0) *x = 0.0f; }
