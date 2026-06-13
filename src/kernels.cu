#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────
//  Embedding forward: out[b,d] = embed[ids[b], d]
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void embedding_fwd(
    const __half* __restrict__ embed,
    const int*   __restrict__ ids,
    __half*                   out,
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
    const __half* __restrict__ d_out,
    const int*    __restrict__ ids,
    __half*                    d_embed,
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
extern "C" __global__ void add_bias(__half* out, const __half* bias, int batch, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / N;
    int i = idx % N;
    if (b >= batch || i >= N) return;
    out[b * N + i] = __hadd(out[b * N + i], bias[i]);
}

// ─────────────────────────────────────────────────────────────
//  FUSED LSTM forward
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void fused_lstm_fwd(
    const __half* __restrict__ gates,
    const __half* __restrict__ c_prev,
    __half*                    h_out,
    __half*                    c_out,
    int H)
{
    int b = blockIdx.x;
    int j = threadIdx.x + blockIdx.y * blockDim.x;
    if (j >= H) return;

    int base = b * 4 * H;
    float gi = __half2float(gates[base +         j]);
    float gf = __half2float(gates[base +     H + j]);
    float go = __half2float(gates[base + 2 * H + j]);
    float gg = __half2float(gates[base + 3 * H + j]);

    float i_g = __frcp_rn(1.0f + __expf(-gi));
    float f_g = __frcp_rn(1.0f + __expf(-gf));
    float o_g = __frcp_rn(1.0f + __expf(-go));
    float g_g = tanhf(gg);

    float c = f_g * __half2float(c_prev[b * H + j]) + i_g * g_g;
    float h = o_g * tanhf(c);

    c_out[b * H + j] = __float2half(c);
    h_out[b * H + j] = __float2half(h);
}

// ─────────────────────────────────────────────────────────────
//  FUSED LSTM backward
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void fused_lstm_bwd(
    const __half* __restrict__ gates_raw,
    const __half* __restrict__ c_prev,
    const __half* __restrict__ c_cur,
    const __half* __restrict__ d_h,
    const __half* __restrict__ d_c_next,
    __half*                    d_gates,
    __half*                    d_c_prev,
    int H)
{
    int b = blockIdx.x;
    int j = threadIdx.x + blockIdx.y * blockDim.x;
    if (j >= H) return;

    int base = b * 4 * H;
    float gi = __half2float(gates_raw[base +         j]);
    float gf = __half2float(gates_raw[base +     H + j]);
    float go = __half2float(gates_raw[base + 2 * H + j]);
    float gg = __half2float(gates_raw[base + 3 * H + j]);

    float i_g = __frcp_rn(1.0f + __expf(-gi));
    float f_g = __frcp_rn(1.0f + __expf(-gf));
    float o_g = __frcp_rn(1.0f + __expf(-go));
    float g_g = tanhf(gg);

    float c  = __half2float(c_cur[b * H + j]);
    float tc = tanhf(c);

    float dc = __half2float(d_h[b * H + j]) * o_g * (1.0f - tc * tc) + __half2float(d_c_next[b * H + j]);

    float di = dc * g_g;
    float df = dc * __half2float(c_prev[b * H + j]);
    float do_ = __half2float(d_h[b * H + j]) * tc;
    float dg  = dc * i_g;

    d_gates[base +         j] = __float2half(di  * i_g * (1.0f - i_g));
    d_gates[base +     H + j] = __float2half(df  * f_g * (1.0f - f_g));
    d_gates[base + 2 * H + j] = __float2half(do_ * o_g * (1.0f - o_g));
    d_gates[base + 3 * H + j] = __float2half(dg  * (1.0f - g_g * g_g));

    d_c_prev[b * H + j] = __float2half(dc * f_g);
}

// ─────────────────────────────────────────────────────────────
//  AdaptiveSoftmax: linear kept in float internally for accuracy
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void asm_linear(
    const __half* __restrict__ A,
    const __half* __restrict__ W,
    const __half* __restrict__ bias,
    __half*                    C,
    int in_dim, int out_dim, int batch)
{
    int b = blockIdx.x;
    int o = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch || o >= out_dim) return;
    float s = __half2float(bias[o]);
    for (int k = 0; k < in_dim; k++)
        s += __half2float(A[b * in_dim + k]) * __half2float(W[o * in_dim + k]);
    C[b * out_dim + o] = __float2half(s);
}

// ─────────────────────────────────────────────────────────────
//  Row-wise softmax in-place  x[batch, n]
// ─────────────────────────────────────_linear───────────
extern "C" __global__ void asm_softmax(__half* x, int n, int batch)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    if (row >= batch) return;
    __half* r = x + row * n;

    float mx = -1e30f;
    for (int i = tid; i < n; i += 32) mx = fmaxf(mx, __half2float(r[i]));
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));

    float s = 0.0f;
    for (int i = tid; i < n; i += 32) {
        float e = __expf(__half2float(r[i]) - mx);
        r[i] = __float2half(e);
        s += e;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    float inv = __frcp_rn(s);
    for (int i = tid; i < n; i += 32) r[i] = __float2half(__half2float(r[i]) * inv);
}

// ─────────────────────────────────────────────────────────────
//  CE grad + per-block loss partial
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void asm_ce_grad(
    __half*       probs,
    const int*    targets,
    float*        loss_partial,
    int n, int offset, int batch, int loss_offset)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    extern __shared__ float shmem[];
    if (tid == 0 && threadIdx.y == 0) shmem[0] = 0.0f;
    __syncthreads();

    float local_loss = 0.0f;
    if (row < batch && tid == 0) {
        int t = targets[row];
        if (t >= 0) {
            int ti = t - offset;
            float p = __half2float(probs[row * n + ti]);
            local_loss = -(p > 1e-30f ? __logf(p) : -30.0f);
            probs[row * n + ti] = __float2half(__half2float(probs[row * n + ti]) - 1.0f);
        }
    }
    if (tid == 0) atomicAdd(&shmem[0], local_loss);
    __syncthreads();
    if (tid == 0 && threadIdx.y == 0) loss_partial[loss_offset + blockIdx.y] = shmem[0];
}

// Weight grad: G_W[o,k] += sum_b d_out[b,o] * A[b,k]
extern "C" __global__ void asm_wgrad(
    const __half* __restrict__ d_out,
    const __half* __restrict__ A,
    __half*                    G_W,
    int batch, int in_dim, int out_dim)
{
    int o = blockIdx.x;
    int k = threadIdx.x + blockIdx.y * blockDim.x;
    if (o >= out_dim || k >= in_dim) return;
    float s = 0.0f;
    for (int b = 0; b < batch; b++)
        s += __half2float(d_out[b * out_dim + o]) * __half2float(A[b * in_dim + k]);
    G_W[o * in_dim + k] = __float2half(__half2float(G_W[o * in_dim + k]) + s);
}

// Bias grad: G_b[o] += sum_b d_out[b,o]
extern "C" __global__ void asm_bgrad(
    const __half* __restrict__ d_out,
    __half*                    G_b,
    int batch, int out_dim)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= out_dim) return;
    float s = 0.0f;
    for (int b = 0; b < batch; b++) s += __half2float(d_out[b * out_dim + o]);
    G_b[o] = __float2half(__half2float(G_b[o]) + s);
}

// Input grad: d_A[b,k] += sum_o d_out[b,o] * W[o,k]
extern "C" __global__ void asm_igrad(
    const __half* __restrict__ d_out,
    const __half* __restrict__ W,
    __half*                    d_A,
    int batch, int in_dim, int out_dim)
{
    int b = blockIdx.x;
    int k = threadIdx.x + blockIdx.y * blockDim.x;
    if (b >= batch || k >= in_dim) return;
    float s = 0.0f;
    for (int o = 0; o < out_dim; o++) s += __half2float(d_out[b * out_dim + o]) * __half2float(W[o * in_dim + k]);
    d_A[b * in_dim + k] = __float2half(__half2float(d_A[b * in_dim + k]) + s);
}

// Reduce sum over batch dim: out[i] = sum_b x[b*N+i]
extern "C" __global__ void reduce_sum_batch(
    const __half* __restrict__ x,
    __half*                    out,
    int batch, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = 0.0f;
    for (int b = 0; b < batch; b++) s += __half2float(x[b * N + i]);
    out[i] = __float2half(__half2float(out[i]) + s);
}

// ─────────────────────────────────────────────────────────────
//  Adam update (f32 parameters, f32 grads)
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
//  GPU gradient clipping (f32 grads)
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

extern "C" __global__ void clip_if_needed(
    const float* __restrict__ partial,
    float*                    grad,
    int partial_n, int grad_n, float clip_val)
{
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    float s = 0.0f;
    for (int i = tid; i < partial_n; i += blockDim.x) s += partial[i];
    shmem[tid] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) shmem[tid] += shmem[tid + off];
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
//  Adam update (FP16 parameters, FP16 grads, FP32 moments)
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void adam_update_f16(
    __half*       param,
    float*        m,
    float*        v,
    const __half* grad,
    float lr, float b1, float b2, float eps, float bc1, float bc2,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = __half2float(grad[i]);
    float m_ = b1 * m[i] + (1.0f - b1) * g;
    float v_ = b2 * v[i] + (1.0f - b2) * g * g;
    m[i] = m_;
    v[i] = v_;
    float p = __half2float(param[i]) - lr * (m_ / bc1) / (__fsqrt_rn(v_ / bc2) + eps);
    param[i] = __float2half(p);
}

extern "C" __global__ void sgd_update_f16(
    __half*       param,
    const __half* grad,
    float lr, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float p = __half2float(param[i]) - lr * __half2float(grad[i]);
    param[i] = __float2half(p);
}

extern "C" __global__ void scale_f16(__half* x, float scale, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2half(__half2float(x[i]) * scale);
}

// ─────────────────────────────────────────────────────────────
//  GPU gradient clipping for FP16 grads (moments stay f32)
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void norm_reduce_f16(
    const __half* __restrict__ grad,
    float*                     partial,
    int n)
{
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float g = (gid < n) ? __half2float(grad[gid]) : 0.0f;
    shmem[tid] = g * g;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = shmem[0];
}

extern "C" __global__ void clip_if_needed_f16(
    const float* __restrict__ partial,
    __half*                   grad,
    int partial_n, int grad_n, float clip_val)
{
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    float s = 0.0f;
    for (int i = tid; i < partial_n; i += blockDim.x) s += partial[i];
    shmem[tid] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) shmem[tid] += shmem[tid + off];
        __syncthreads();
    }
    if (tid != 0) return;
    float norm = __fsqrt_rn(shmem[0]);
    if (norm > clip_val) {
        float scale = clip_val / norm;
        for (int i = 0; i < grad_n; i++) grad[i] = __float2half(__half2float(grad[i]) * scale);
    }
}

// ─────────────────────────────────────────────────────────────
//  Reduce n floats into a single output (atomic add)
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void reduce_sum(const float* in, float* out, int n)
{
    extern __shared__ float shmem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    shmem[tid] = (gid < n) ? in[gid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, shmem[0]);
}

extern "C" __global__ void zero_float(float* x) { if (threadIdx.x == 0 && blockIdx.x == 0) *x = 0.0f; }

// ─────────────────────────────────────────────────────────────
//  FP16 → FP32 copy helpers (for grad/Adam which stay f32)
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void f16_to_f32(const __half* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __half2float(in[i]);
}

extern "C" __global__ void f32_to_f16(const float* in, __half* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(in[i]);
}
