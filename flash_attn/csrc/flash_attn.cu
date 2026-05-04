#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>

#define WARP_SIZE 32

// ---------------------------------------------------------------------------
// FlashAttention-1: Forward Kernel
//
// Each CTA processes Br rows of Q.  Each warp handles one Q row exclusively.
// All threads participate in cooperative loads to shared memory.
// Inactive warps (for partial Q tiles) skip computation but still do loads.
//
// Shared memory: s_Q[Br*d] + s_K[Bc*d] + s_V[Bc*d] + s_O[Br*d]
// ---------------------------------------------------------------------------

template <int Br, int Bc, int d>
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q, // (B, H, N, d)
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int N,
    float sm_scale
) {
    static_assert(Br <= 32, "Br must be <= 32 (one warp per row)");
    static_assert(Bc <= 32, "Bc must be <= 32");
    static_assert(d <= 128, "d must be <= 128");

    const int batch_idx  = blockIdx.z / H;
    const int head_idx   = blockIdx.z % H;
    const int q_block    = blockIdx.x;

    const int q_start = q_block * Br;
    const int valid_br = min(Br, N - q_start);
    if (valid_br <= 0) return;

    const int tid      = threadIdx.x;
    const int num_thread  = blockDim.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;
    const int qi_row   = warp_id; // 每个warp单独处理Q的1行
    const int is_active = (qi_row < valid_br);

    const int offset = batch_idx * H * N * d + head_idx * N * d;

    extern __shared__ float smem[];
    float* s_Q = smem;                            // Br * d
    float* s_K = smem + Br * d;                    // Bc * d
    float* s_V = smem + Br * d + Bc * d;           // Bc * d
    float* s_O = smem + Br * d + 2 * Bc * d;       // Br * d

    // --- Cooperative load Q into shared memory ---
    for (int i = tid; i < Br * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        int n = q_start + row;
        s_Q[i] = (n < N) ? __ldg(Q + offset + n * d + col) : 0.0f;
    }

    // --- Initialize s_O to zero ---
    for (int i = tid; i < Br * d; i += num_thread) {
        s_O[i] = 0.0f;
    }
    __syncthreads();

    // --- Per-warp running statistics (in registers) ---
    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    // --- Main loop over K/V tiles ---
    const int n_kv_tiles = (N + Bc - 1) / Bc;
    for (int kv_block = 0; kv_block < n_kv_tiles; kv_block++) {
        const int kv_start = kv_block * Bc;
        const int valid_bc = min(Bc, N - kv_start);

        // Cooperative load Kj into shared memory
        for (int i = tid; i < Bc * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            int n = kv_start + row;
            s_K[i] = (n < N) ? __ldg(K + offset + n * d + col) : 0.0f;
        }
        // Cooperative load Vj into shared memory
        for (int i = tid; i < Bc * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            int n = kv_start + row;
            s_V[i] = (n < N) ? __ldg(V + offset + n * d + col) : 0.0f;
        }
        __syncthreads();

        // --- Compute for active warps ONLY; all warps still reach __syncthreads() below ---
        if (is_active) {
            // Phase 1: compute m_tile = rowmax(S) for this row
            float m_tile = -FLT_MAX;

            for (int j = lane_id; j < valid_bc; j += WARP_SIZE) {
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    dot += s_Q[qi_row * d + k] * s_K[j * d + k];
                }
                m_tile = fmaxf(m_tile, dot * sm_scale);
            }

            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, offset));
            }

            // Phase 2: online softmax update
            float m_old = m_i;
            float m_new = fmaxf(m_old, m_tile);
            float rescale = (l_i > 0.0f) ? expf(m_old - m_new) : 0.0f;

            for (int k = lane_id; k < d; k += WARP_SIZE) {
                s_O[qi_row * d + k] *= rescale;
            }
            __syncwarp();

            float exp_sum = 0.0f;

            for (int j = lane_id; j < valid_bc; j += WARP_SIZE) {
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    dot += s_Q[qi_row * d + k] * s_K[j * d + k];
                }
                float s_val = dot * sm_scale;
                float p = expf(s_val - m_new);
                exp_sum += p;

                for (int k = 0; k < d; k++) {
                    atomicAdd(s_O + qi_row * d + k, p * s_V[j * d + k]);
                }
            }

            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                exp_sum += __shfl_xor_sync(0xffffffff, exp_sum, offset);
            }

            l_i = rescale * l_i + exp_sum;
            m_i = m_new;
        }

        __syncthreads();  // sync ALL warps before next K/V tile
    }

    // --- Finalize: O = O / l, LSE = m + log(l) ---
    if (!is_active) return;

    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    int n = q_start + qi_row;
    for (int k = lane_id; k < d; k += WARP_SIZE) {
        O[offset + n * d + k] = s_O[qi_row * d + k] * inv_l;
    }
    if (lane_id == 0) {
        LSE[batch_idx * H * N + head_idx * N + n] = m_i + __logf(l_i);
    }
}

// ---------------------------------------------------------------------------
// D = rowsum(dO * O) for all B*H rows at once
// ---------------------------------------------------------------------------
__global__ void compute_D_kernel(
    const float* __restrict__ dO,
    const float* __restrict__ O,
    float* __restrict__ D,
    int total_N, int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < total_N; i += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < d; k++) {
            sum += dO[i * d + k] * O[i * d + k];
        }
        D[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// FlashAttention-1: Backward Kernel
//
// Recomputes P = softmax(S) from Q, K and stored LSE.
// Computes dQ, dK, dV tile-by-tile with same memory pattern as forward.
// ---------------------------------------------------------------------------

template <int Br, int Bc, int d>
__global__ void flash_attn_bwd_kernel(
    const float* __restrict__ dO,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ LSE,
    const float* __restrict__ D,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    int B, int H, int N,
    float sm_scale
) {
    static_assert(Br <= 32, "Br must be <= 32");
    static_assert(Bc <= 32, "Bc must be <= 32");
    static_assert(d <= 128, "d must be <= 128");

    const int batch_idx = blockIdx.z / H; //(batch, head)
    const int head_idx  = blockIdx.z % H;
    const int q_block   = blockIdx.x;

    const int q_start = q_block * Br;
    const int valid_br = min(Br, N - q_start);
    if (valid_br <= 0) return;

    const int tid      = threadIdx.x;
    const int num_thread  = blockDim.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;
    const int qi_row   = warp_id;
    const int is_active = (qi_row < valid_br);

    const int offset = batch_idx * H * N * d + head_idx * N * d;
    const int lse_offset = batch_idx * H * N + head_idx * N;

    extern __shared__ float smem[];
    float* s_Q  = smem;                                    // Br * d
    float* s_dO = smem + Br * d;                            // Br * d
    float* s_LSE = smem + 2 * Br * d;                        // Br
    float* s_D  = smem + 2 * Br * d + Br;                    // Br
    float* s_K  = smem + 2 * Br * d + 2 * Br;                // Bc * d
    float* s_V  = smem + 2 * Br * d + 2 * Br + Bc * d;       // Bc * d
    float* s_dQ = smem + 2 * Br * d + 2 * Br + 2 * Bc * d;   // Br * d

    // --- Cooperative load Qi, dOi into shared memory ---
    for (int i = tid; i < Br * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        int n = q_start + row;
        if (n < N) {
            s_Q[i]  = __ldg(Q + offset + n * d + col);
            s_dO[i] = __ldg(dO + offset + n * d + col);
        } else {
            s_Q[i]  = 0.0f;
            s_dO[i] = 0.0f;
        }
    }

    // Load LSE and D (scalars)
    for (int i = tid; i < Br; i += num_thread) {
        int n = q_start + i;
        if (n < N) {
            s_LSE[i] = __ldg(LSE + lse_offset + n);
            s_D[i]   = __ldg(D + lse_offset + n);
        } else {
            s_LSE[i] = 0.0f;
            s_D[i]   = 0.0f;
        }
    }

    // Initialize s_dQ to zero
    for (int i = tid; i < Br * d; i += num_thread) {
        s_dQ[i] = 0.0f;
    }
    __syncthreads();

    // Per-warp values
    float lse_i = is_active ? s_LSE[qi_row] : 0.0f;
    float d_i   = is_active ? s_D[qi_row]   : 0.0f;

    // --- Main loop over K/V tiles ---
    const int n_kv_tiles = (N + Bc - 1) / Bc;
    for (int kv_block = 0; kv_block < n_kv_tiles; kv_block++) {
        const int kv_start = kv_block * Bc;
        const int valid_bc = min(Bc, N - kv_start);

        // Cooperative load Kj, Vj
        for (int i = tid; i < Bc * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            int n = kv_start + row;
            s_K[i] = (n < N) ? __ldg(K + offset + n * d + col) : 0.0f;
        }
        for (int i = tid; i < Bc * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            int n = kv_start + row;
            s_V[i] = (n < N) ? __ldg(V + offset + n * d + col) : 0.0f;
        }
        __syncthreads();

        if (is_active) {
            float dQ_acc[d];
            #pragma unroll
            for (int k = 0; k < d; k++) dQ_acc[k] = 0.0f;

            for (int j = lane_id; j < valid_bc; j += WARP_SIZE) {
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    dot += s_Q[qi_row * d + k] * s_K[j * d + k];
                }
                float s_val = dot * sm_scale;
                float p = expf(s_val - lse_i);

                float dP_val = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    dP_val += s_dO[qi_row * d + k] * s_V[j * d + k];
                }
                float dS = sm_scale * p * (dP_val - d_i);

                int n_j = kv_start + j;
                for (int k = 0; k < d; k++) {
                    dQ_acc[k] += dS * s_K[j * d + k];
                    atomicAdd(dV + offset + n_j * d + k, p * s_dO[qi_row * d + k]);
                    atomicAdd(dK + offset + n_j * d + k, dS * s_Q[qi_row * d + k]);
                }
            }

            // Warp reduce dQ_acc
            #pragma unroll
            for (int k = 0; k < d; k++) {
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    dQ_acc[k] += __shfl_xor_sync(0xffffffff, dQ_acc[k], offset);
                }
            }

            // Write dQ to shared memory
            for (int k = lane_id; k < d; k += WARP_SIZE) {
                s_dQ[qi_row * d + k] += dQ_acc[k];
            }
        }

        __syncthreads();
    }

    // --- Write dQ to global memory (active warps only) ---
    if (is_active) {
        int n = q_start + qi_row;
        for (int k = lane_id; k < d; k += WARP_SIZE) {
            dQ[offset + n * d + k] = s_dQ[qi_row * d + k];
        }
    }
}

// ---------------------------------------------------------------------------
// Host launch wrappers with template instantiation for d = 32, 64, 128
//
// Tile sizes chosen to fit shared memory <= 48KB per CTA:
//   d=32:  Br=32 Bc=32  fwd: smem=(64+64)*32*4=16KB   bwd: (3072+2048+64)*4=20KB
//   d=64:  Br=32 Bc=32  fwd: (64+64)*64*4=32KB         bwd: (6144+4096+64)*4=40KB
//   d=128: Br=16 Bc=32  fwd: (32+64)*128*4=48KB        (forward only)
//          Br=16 Bc=16  bwd: (6144+4096+32)*4=40KB     (backward)
// ---------------------------------------------------------------------------

#define DISPATCH_FWD(Br, Bc, d_val)                                   \
    case d_val: {                                                      \
        int _n_blocks = (N + Br - 1) / Br;                             \
        dim3 grid(_n_blocks, 1, B * H);                                \
        dim3 block(Br * WARP_SIZE, 1, 1);                              \
        int smem_bytes = (Br + 2 * Bc + Br) * d_val * sizeof(float);  \
        flash_attn_fwd_kernel<Br, Bc, d_val>                           \
            <<<grid, block, smem_bytes, stream>>>(                     \
                Q, K, V, O, LSE, B, H, N, sm_scale);                  \
        break;                                                         \
    }

void flash_attn_forward(
    const float* Q, const float* K, const float* V,
    float* O, float* LSE,
    int B, int H, int N, int d,
    float sm_scale, cudaStream_t stream
) {
    switch (d) {
        DISPATCH_FWD(32, 32, 32)
        DISPATCH_FWD(32, 32, 64)
        DISPATCH_FWD(16, 32, 128)
        default:
            fprintf(stderr, "flash_attn_fwd: unsupported d=%d\n", d);
            break;
    }
}
#undef DISPATCH_FWD

// ---------------------------------------------------------------------------

#define DISPATCH_BWD(Br, Bc, d_val)                                                    \
    case d_val: {                                                                       \
        int _n_blocks = (N + Br - 1) / Br;                                              \
        dim3 grid(_n_blocks, 1, B * H);                                                  \
        dim3 block(Br * WARP_SIZE, 1, 1);                                               \
        int smem_bytes = (3 * Br * d_val + 2 * Bc * d_val + 2 * Br) * sizeof(float);   \
        flash_attn_bwd_kernel<Br, Bc, d_val>                                            \
            <<<grid, block, smem_bytes, stream>>>(                                      \
                dO, Q, K, V, O, LSE, D, dQ, dK, dV, B, H, N, sm_scale);               \
        break;                                                                          \
    }

void flash_attn_backward(
    const float* dO, const float* Q, const float* K, const float* V,
    const float* O, const float* LSE,
    float* dQ, float* dK, float* dV,
    int B, int H, int N, int d,
    float sm_scale, cudaStream_t stream
) {
    int total_N = B * H * N;

    // Precompute D = rowsum(dO * O)
    float* D;
    cudaMalloc(&D, total_N * sizeof(float));
    int block_dim = 256;
    int grid_dim = (total_N + block_dim - 1) / block_dim;
    compute_D_kernel<<<grid_dim, block_dim, 0, stream>>>(dO, O, D, total_N, d);

    switch (d) {
        DISPATCH_BWD(32, 32, 32)
        DISPATCH_BWD(32, 32, 64)
        DISPATCH_BWD(16, 16, 128)
        default:
            fprintf(stderr, "flash_attn_bwd: unsupported d=%d\n", d);
            break;
    }

    cudaFree(D);
}
#undef DISPATCH_BWD
