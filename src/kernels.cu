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
        float v = __half2float(r[i]) - mx;
        if (v < -30.0f) v = -30.0f;  // prevent underflow to zero
        float e = __expf(v);
        r[i] = __float2half(e);
        s += e;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    float inv = s > 0.0f ? __frcp_rn(s) : 0.0f;
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
    if (row < batch) {
        int t = targets[row];
        if (t >= 0 && tid == 0) {
            int ti = t - offset;
            float p = __half2float(probs[row * n + ti]);
            local_loss = -(p > 1e-30f ? __logf(p) : -30.0f);
            probs[row * n + ti] = __float2half(__half2float(probs[row * n + ti]) - 1.0f);
        } else if (t < 0) {
            for (int i = tid; i < n; i += 32) probs[row * n + i] = __float2half(0.0f);
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

// ─────────────────────────────────────────────────────────────
//  LayerNorm forward: out = (x - mean) / sqrt(var + eps) * gamma + beta
//  x, out: [N, D]   mean, rstd: [N]   gamma, beta: [D]
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void layer_norm_fwd(
    __half*       out,
    float*        mean,
    float*        rstd,
    const __half* x,
    const __half* gamma,
    const __half* beta,
    int N, int D, float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return;

    const __half* xr = x + row * D;
    __half*       or_ = out + row * D;

    // Compute mean via warp reduction
    float s = 0.0f;
    for (int i = tid; i < D; i += 32) s += __half2float(xr[i]);
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    float mu = s / (float)D;

    // Compute variance
    float sv = 0.0f;
    for (int i = tid; i < D; i += 32) {
        float d = __half2float(xr[i]) - mu;
        sv += d * d;
    }
    for (int off = 16; off > 0; off >>= 1) sv += __shfl_xor_sync(0xffffffff, sv, off);
    float rs = __frsqrt_rn(sv / (float)D + eps);

    if (tid == 0) { mean[row] = mu; rstd[row] = rs; }

    for (int i = tid; i < D; i += 32) {
        float n = (__half2float(xr[i]) - mu) * rs;
        or_[i] = __float2half(n * __half2float(gamma[i]) + __half2float(beta[i]));
    }
}

// ─────────────────────────────────────────────────────────────
//  LayerNorm backward
//  dy, x, out: [N, D]   gamma: [D]   mean, rstd: [N]
//  dx += ...,  dgamma += ...,  dbeta += ...
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void layer_norm_bwd(
    __half*       dx,
    __half*       dgamma,
    __half*       dbeta,
    const __half* dy,
    const __half* x,
    const float*  mean,
    const float*  rstd,
    const __half* gamma,
    int N, int D)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return;

    const __half* dyr = dy    + row * D;
    const __half* xr  = x    + row * D;
    __half*       dxr = dx   + row * D;
    float mu = mean[row];
    float rs = rstd[row];

    // sum(dy * gamma) and sum(dy * gamma * xhat)
    float s1 = 0.0f, s2 = 0.0f;
    for (int i = tid; i < D; i += 32) {
        float dyi = __half2float(dyr[i]);
        float gi  = __half2float(gamma[i]);
        float xh  = (__half2float(xr[i]) - mu) * rs;
        s1 += dyi * gi;
        s2 += dyi * gi * xh;
    }
    for (int off = 16; off > 0; off >>= 1) {
        s1 += __shfl_xor_sync(0xffffffff, s1, off);
        s2 += __shfl_xor_sync(0xffffffff, s2, off);
    }

    float inv_D = 1.0f / (float)D;
    for (int i = tid; i < D; i += 32) {
        float dyi = __half2float(dyr[i]);
        float gi  = __half2float(gamma[i]);
        float xh  = (__half2float(xr[i]) - mu) * rs;
        float dx_ = rs * (dyi * gi - inv_D * s1 - inv_D * xh * s2);
        dxr[i] = __float2half(__half2float(dxr[i]) + dx_);
        // dgamma and dbeta accumulate across all rows — use atomicAdd
        atomicAdd((float*)dgamma + i, dyi * xh);  // store in float trick
        atomicAdd((float*)dbeta  + i, dyi);
    }
}

// ─────────────────────────────────────────────────────────────
//  GELU forward: out[i] = x[i] * 0.5 * (1 + tanh(c*(x + 0.044715*x^3)))
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void gelu_fwd(__half* out, const __half* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = __half2float(x[i]);
    float c = 0.7978845608f; // sqrt(2/pi)
    float inner = c * (xi + 0.044715f * xi * xi * xi);
    float t = tanhf(inner);
    out[i] = __float2half(0.5f * xi * (1.0f + t));
}

// ─────────────────────────────────────────────────────────────
//  GELU backward: dx += dy * (0.5*(1+tanh) + 0.5*x*sech^2*c*(1+3*0.044715*x^2))
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void gelu_bwd(__half* dx, const __half* dy, const __half* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi  = __half2float(x[i]);
    float dyi = __half2float(dy[i]);
    float c     = 0.7978845608f;
    float inner = c * (xi + 0.044715f * xi * xi * xi);
    float t   = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float grad = 0.5f * (1.0f + t) + 0.5f * xi * sech2 * c * (1.0f + 3.0f * 0.044715f * xi * xi);
    dx[i] = __float2half(__half2float(dx[i]) + dyi * grad);
}

// ─────────────────────────────────────────────────────────────
//  Causal softmax in-place: attn[BH, T, T]
//  For each row i: zero out j > i (future), then softmax
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void causal_softmax_fwd(__half* attn, int BH, int T)
{
    int bh  = blockIdx.x;
    int row = blockIdx.y;   // query position (0..T)
    int tid = threadIdx.x;
    if (bh >= BH || row >= T) return;

    __half* r = attn + bh * T * T + row * T;

    // Find max over valid positions [0..row]
    float mx = -1e30f;
    for (int j = tid; j <= row; j += 32) mx = fmaxf(mx, __half2float(r[j]));
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));

    // Exp and sum
    float s = 0.0f;
    for (int j = tid; j < T; j += 32) {
        float v;
        if (j <= row) {
            v = __expf(__half2float(r[j]) - mx);
        } else {
            v = 0.0f;
        }
        r[j] = __float2half(v);
        if (j <= row) s += v;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffff, s, off);
    float inv = s > 0.0f ? __frcp_rn(s) : 0.0f;
    for (int j = tid; j <= row; j += 32) r[j] = __float2half(__half2float(r[j]) * inv);
}

// ─────────────────────────────────────────────────────────────
//  Attention softmax backward: ds = p*(dy - sum_j(p_j * dy_j))
//  p, dy, ds: [BH, T, T]
// ─────────────────────────────────────────────────────────────
extern "C" __global__ void attn_softmax_bwd(__half* ds, const __half* p, const __half* dy, int BH, int T)
{
    int bh  = blockIdx.x;
    int row = blockIdx.y;
    int tid = threadIdx.x;
    if (bh >= BH || row >= T) return;

    const __half* pr  = p  + bh * T * T + row * T;
    const __half* dyr = dy + bh * T * T + row * T;
    __half*       dsr = ds + bh * T * T + row * T;

    // dot = sum_j p[j] * dy[j]  over valid [0..row]
    float dot = 0.0f;
    for (int j = tid; j <= row; j += 32) dot += __half2float(pr[j]) * __half2float(dyr[j]);
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);

    for (int j = tid; j <= row; j += 32) {
        float pj = __half2float(pr[j]);
        float d  = pj * (__half2float(dyr[j]) - dot);
        dsr[j] = __float2half(__half2float(dsr[j]) + d);
    }
}

// ─────────────────────────────────────────────────────────────
//  GPU TRAINING KERNELS (v2 — fully on-GPU forward/backward)
// ─────────────────────────────────────────────────────────────

// Embedding + positional add: x[t,d] = embed[ids[t],d] + pos[t,d]
extern "C" __global__ void embedding_pos_fwd(
    __half* out, const __half* embed, const __half* pos,
    const int* ids, int T, int D)
{
    int t = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (t >= T || d >= D) return;
    out[t*D+d] = __hadd(embed[(long)ids[t]*D+d], pos[t*D+d]);
}

// Split qkv[T, 3D] → q[H,T,dh], k[H,T,dh], v[H,T,dh]
// Grid: (T, H, 1)  Block: (dh, 1, 1)
extern "C" __global__ void qkv_split_heads(
    const __half* qkv, __half* q, __half* k, __half* v,
    int T, int H, int dh)
{
    int t  = blockIdx.x;
    int hd = blockIdx.y;
    int d  = threadIdx.x;
    int D  = H * dh;
    if (t >= T || hd >= H || d >= dh) return;
    q[hd*T*dh + t*dh + d] = qkv[t*3*D + hd*dh + d];
    k[hd*T*dh + t*dh + d] = qkv[t*3*D + D + hd*dh + d];
    v[hd*T*dh + t*dh + d] = qkv[t*3*D + 2*D + hd*dh + d];
}

// Merge [H,T,dh] → out[T,D]
// Grid: (T, H, 1)  Block: (dh, 1, 1)
extern "C" __global__ void heads_merge(
    const __half* ctx, __half* out,
    int T, int H, int dh)
{
    int t  = blockIdx.x;
    int hd = blockIdx.y;
    int d  = threadIdx.x;
    if (t >= T || hd >= H || d >= dh) return;
    out[t*H*dh + hd*dh + d] = ctx[hd*T*dh + t*dh + d];
}

// heads_split: same as heads_merge but direction reversed [T,D] → [H,T,dh]
// Grid: (T, H, 1)  Block: (dh, 1, 1)
extern "C" __global__ void heads_split(
    const __half* inp, __half* out,
    int T, int H, int dh)
{
    int t  = blockIdx.x;
    int hd = blockIdx.y;
    int d  = threadIdx.x;
    if (t >= T || hd >= H || d >= dh) return;
    out[hd*T*dh + t*dh + d] = inp[t*H*dh + hd*dh + d];
}

// Merge d_q,d_k,d_v [H,T,dh] → d_qkv[T,3D]
// Grid: (T, H, 1)  Block: (dh, 1, 1)
extern "C" __global__ void qkv_grad_merge(
    const __half* dq, const __half* dk, const __half* dv,
    __half* d_qkv, int T, int H, int dh)
{
    int t  = blockIdx.x;
    int hd = blockIdx.y;
    int d  = threadIdx.x;
    int D  = H * dh;
    if (t >= T || hd >= H || d >= dh) return;
    d_qkv[t*3*D + hd*dh + d]   = dq[hd*T*dh + t*dh + d];
    d_qkv[t*3*D + D + hd*dh + d]   = dk[hd*T*dh + t*dh + d];
    d_qkv[t*3*D + 2*D + hd*dh + d] = dv[hd*T*dh + t*dh + d];
}

// In-place add: a[i] += b[i]  (f16)
extern "C" __global__ void add_inplace(__half* a, const __half* b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __hadd(a[i], b[i]);
}

// Copy f16 buffer
extern "C" __global__ void copy_f16(__half* dst, const __half* src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}

// Zero f16 buffer
extern "C" __global__ void zero_f16(__half* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2half(0.0f);
}

// Zero f32 buffer
extern "C" __global__ void zero_f32(float* x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = 0.0f;
}

// Multi-head attention scores: scores[h,qi,ki] = scale*dot(Q[h,qi],K[h,ki])
// Grid: (H, T, 1)  Block: (T, 1, 1)  — T <= 256
extern "C" __global__ void mha_scores(
    const __half* q, const __half* k, __half* scores,
    int H, int T, int dh, float scale)
{
    int h  = blockIdx.x;
    int qi = blockIdx.y;
    int ki = threadIdx.x;
    if (h >= H || qi >= T || ki >= T) return;
    const __half* qv = q + h*T*dh + qi*dh;
    const __half* kv = k + h*T*dh + ki*dh;
    float dot = 0.0f;
    for (int d = 0; d < dh; d++) dot += __half2float(qv[d]) * __half2float(kv[d]);
    scores[h*T*T + qi*T + ki] = __float2half(dot * scale);
}

// Multi-head attention context: ctx[h,qi,d] = sum_{ki<=qi} attn[h,qi,ki]*V[h,ki,d]
// Grid: (H, T, 1)  Block: (dh, 1, 1)
extern "C" __global__ void mha_context(
    const __half* attn, const __half* v, __half* ctx,
    int H, int T, int dh)
{
    int h  = blockIdx.x;
    int qi = blockIdx.y;
    int d  = threadIdx.x;
    if (h >= H || qi >= T || d >= dh) return;
    float s = 0.0f;
    const __half* av = attn + h*T*T + qi*T;
    for (int ki = 0; ki <= qi; ki++)
        s += __half2float(av[ki]) * __half2float(v[h*T*dh + ki*dh + d]);
    ctx[h*T*dh + qi*dh + d] = __float2half(s);
}

// Backward d_v: dv[h,ki,d] += sum_{qi>=ki} attn[h,qi,ki]*d_ctx[h,qi,d]
// Grid: (H, T, 1)  Block: (dh, 1, 1)
extern "C" __global__ void mha_dv(
    const __half* attn, const __half* d_ctx, __half* dv,
    int H, int T, int dh)
{
    int h  = blockIdx.x;
    int ki = blockIdx.y;
    int d  = threadIdx.x;
    if (h >= H || ki >= T || d >= dh) return;
    float s = 0.0f;
    for (int qi = ki; qi < T; qi++)
        s += __half2float(attn[h*T*T + qi*T + ki]) * __half2float(d_ctx[h*T*dh + qi*dh + d]);
    dv[h*T*dh + ki*dh + d] = __float2half(__half2float(dv[h*T*dh + ki*dh + d]) + s);
}

// Backward d_attn: d_attn[h,qi,ki] = dot(d_ctx[h,qi], V[h,ki])  (ki<=qi)
// Grid: (H, T, 1)  Block: (T, 1, 1)
extern "C" __global__ void mha_dattn(
    const __half* d_ctx, const __half* v, __half* d_attn,
    int H, int T, int dh)
{
    int h  = blockIdx.x;
    int qi = blockIdx.y;
    int ki = threadIdx.x;
    if (h >= H || qi >= T || ki >= T) return;
    if (ki > qi) { d_attn[h*T*T + qi*T + ki] = __float2half(0.0f); return; }
    float s = 0.0f;
    const __half* dcv = d_ctx + h*T*dh + qi*dh;
    const __half* vv  = v + h*T*dh + ki*dh;
    for (int d = 0; d < dh; d++) s += __half2float(dcv[d]) * __half2float(vv[d]);
    d_attn[h*T*T + qi*T + ki] = __float2half(s);
}

// Backward d_q: dq[h,qi,d] += sum_{ki<=qi} d_attn_pre[h,qi,ki]*K[h,ki,d]
// Grid: (H, T, 1)  Block: (dh, 1, 1)
extern "C" __global__ void mha_dq(
    const __half* d_attn_pre, const __half* k, __half* dq,
    int H, int T, int dh)
{
    int h  = blockIdx.x;
    int qi = blockIdx.y;
    int d  = threadIdx.x;
    if (h >= H || qi >= T || d >= dh) return;
    float s = 0.0f;
    for (int ki = 0; ki <= qi; ki++)
        s += __half2float(d_attn_pre[h*T*T + qi*T + ki]) * __half2float(k[h*T*dh + ki*dh + d]);
    dq[h*T*dh + qi*dh + d] = __float2half(__half2float(dq[h*T*dh + qi*dh + d]) + s);
}

// Backward d_k: dk[h,ki,d] += sum_{qi>=ki} d_attn_pre[h,qi,ki]*Q[h,qi,d]
// Grid: (H, T, 1)  Block: (dh, 1, 1)
extern "C" __global__ void mha_dk(
    const __half* d_attn_pre, const __half* q, __half* dk,
    int H, int T, int dh)
{
    int h  = blockIdx.x;
    int ki = blockIdx.y;
    int d  = threadIdx.x;
    if (h >= H || ki >= T || d >= dh) return;
    float s = 0.0f;
    for (int qi = ki; qi < T; qi++)
        s += __half2float(d_attn_pre[h*T*T + qi*T + ki]) * __half2float(q[h*T*dh + qi*dh + d]);
    dk[h*T*dh + ki*dh + d] = __float2half(__half2float(dk[h*T*dh + ki*dh + d]) + s);
}

// CE loss + d_logits for masked positions
// Grid: (T, 1, 1)  Block: (256, 1, 1)  shared: 256*4 bytes
extern "C" __global__ void softmax_ce_masked(
    const __half* logits,
    float*        d_logits,
    float*        loss_acc,
    const int*    targets,
    const float*  mask,
    int T, int V, float scale)
{
    extern __shared__ float shmem[];
    int t   = blockIdx.x;
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    if (t >= T) return;
    if (fabsf(mask[t]) < 0.5f) { return; }

    const __half* row  = logits   + (long)t * V;
    float*        drow = d_logits + (long)t * V;
    int tgt = targets[t];
    if (tgt < 0 || tgt >= V) return;

    // max
    float mx = -1e30f;
    for (int v = tid; v < V; v += bsz) mx = fmaxf(mx, __half2float(row[v]));
    shmem[tid] = mx;
    __syncthreads();
    for (int s = bsz/2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] = fmaxf(shmem[tid], shmem[tid+s]);
        __syncthreads();
    }
    mx = shmem[0];
    __syncthreads();

    // sum exp
    float sm = 0.0f;
    for (int v = tid; v < V; v += bsz) sm += __expf(__half2float(row[v]) - mx);
    shmem[tid] = sm;
    __syncthreads();
    for (int s = bsz/2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid+s];
        __syncthreads();
    }
    float inv = __frcp_rn(shmem[0]);
    __syncthreads();

    // d_logits = (prob - onehot) * scale
    for (int v = tid; v < V; v += bsz) {
        float prob = __expf(__half2float(row[v]) - mx) * inv;
        drow[v] = (prob - (v == tgt ? 1.0f : 0.0f)) * scale;
    }
    if (tid == 0) {
        float tp = __expf(__half2float(row[tgt]) - mx) * inv;
        atomicAdd(loss_acc, -__logf(fmaxf(tp, 1e-9f)));
    }
}

// Bias grad: g_bias[j] += sum_t dx[t,j]  (atomicAdd into f32)
// Grid: ceil(N/256)  Block: 256
extern "C" __global__ void bias_grad_f16_to_f32(
    const __half* dx, float* g_bias, int T, int N)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float s = 0.0f;
    for (int t = 0; t < T; t++) s += __half2float(dx[t*N+j]);
    atomicAdd(&g_bias[j], s);
}

// LayerNorm backward v2: dx (f16 +=), dgamma/dbeta f32 atomicAdd
// One block per row, 32 threads
extern "C" __global__ void layer_norm_bwd_v2(
    __half*       dx,
    float*        dgamma,
    float*        dbeta,
    const __half* dy,
    const __half* x,
    const float*  mean,
    const float*  rstd,
    const __half* gamma,
    int N, int D)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= N) return;

    const __half* dyr = dy  + row * D;
    const __half* xr  = x   + row * D;
    __half*       dxr = dx  + row * D;
    float mu = mean[row], rs = rstd[row];

    float s1 = 0.0f, s2 = 0.0f;
    for (int i = tid; i < D; i += 32) {
        float dyi = __half2float(dyr[i]);
        float gi  = __half2float(gamma[i]);
        float xh  = (__half2float(xr[i]) - mu) * rs;
        s1 += dyi * gi;
        s2 += dyi * gi * xh;
    }
    for (int off = 16; off > 0; off >>= 1) {
        s1 += __shfl_xor_sync(0xffffffff, s1, off);
        s2 += __shfl_xor_sync(0xffffffff, s2, off);
    }
    float inv_D = 1.0f / (float)D;
    for (int i = tid; i < D; i += 32) {
        float dyi = __half2float(dyr[i]);
        float gi  = __half2float(gamma[i]);
        float xh  = (__half2float(xr[i]) - mu) * rs;
        float dx_ = rs * (dyi*gi - inv_D*s1 - inv_D*xh*s2);
        dxr[i] = __float2half(__half2float(dxr[i]) + dx_);
        atomicAdd(&dgamma[i], dyi * xh);
        atomicAdd(&dbeta[i],  dyi);
    }
}

// Adam update: f32 grad → f16 param, f32 moments
extern "C" __global__ void adam_update_f16_from_f32(
    __half*      param,
    float*       m,
    float*       v,
    const float* grad,
    float lr, float b1, float b2, float eps, float bc1, float bc2,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g  = grad[i];
    float m_ = b1*m[i] + (1.0f-b1)*g;
    float v_ = b2*v[i] + (1.0f-b2)*g*g;
    m[i] = m_; v[i] = v_;
    float p = __half2float(param[i]) - lr*(m_/bc1)/(__fsqrt_rn(v_/bc2)+eps);
    param[i] = __float2half(p);
}

// Scatter-add embedding grad (f32): g_embed[ids[t],d] += dx[t,d]
// Grid: (T, ceil(D/256), 1)  Block: (256, 1, 1)
extern "C" __global__ void embedding_bwd_f32(
    const __half* dx, const int* ids, float* g_embed, int T, int D)
{
    int t = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (t >= T || d >= D) return;
    atomicAdd(&g_embed[(long)ids[t]*D+d], __half2float(dx[t*D+d]));
}

// Positional embedding grad: g_pos[t,d] += dx[t,d]
// Grid: (T, ceil(D/256), 1)  Block: (256, 1, 1)
extern "C" __global__ void pos_grad_add_f32(
    const __half* dx, float* g_pos, int T, int D)
{
    int t = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    if (t >= T || d >= D) return;
    atomicAdd(&g_pos[t*D+d], __half2float(dx[t*D+d]));
}

// d_logits f32 → f16 conversion for GEMM
// Grid: ceil(n/256)  Block: 256
extern "C" __global__ void f32_to_f16_2d(const float* in, __half* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(in[i]);
}

// Adam update for embed/pos: f32 grad → f16 param
// (reuses adam_update_f16_from_f32, same kernel)

// Zero a single f32 scalar
extern "C" __global__ void zero_scalar_f32(float* x)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) x[0] = 0.0f;
}
