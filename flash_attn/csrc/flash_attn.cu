#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>

#define WARP_SIZE 32

// ===========================================================================
// FlashAttention-1: Forward Kernel
//
// Each CTA processes Br rows of Q.  Each warp handles one Q row exclusively.
// All threads participate in cooperative loads to shared memory.
// Inactive warps (for partial Q tiles) skip computation but still do loads.
//
// Shared memory: s_Q[Br*d] + s_K[Bc*d] + s_V[Bc*d] + s_O[Br*d]
// ===========================================================================

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

    constexpr int D_STRIDE = d + 1;  // +1 pad avoids bank conflicts (d % 32 == 0)

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
    const int qi_row   = warp_id;
    const int is_active = (qi_row < valid_br);

    const int offset = batch_idx * H * N * d + head_idx * N * d;

    extern __shared__ float smem[];
    float* s_Q = smem;                                              // Br * D_STRIDE
    float* s_K = smem + Br * D_STRIDE;                              // Bc * D_STRIDE
    float* s_V = smem + Br * D_STRIDE + Bc * D_STRIDE;              // Bc * D_STRIDE
    float* s_O = smem + Br * D_STRIDE + 2 * Bc * D_STRIDE;          // Br * D_STRIDE

    // --- Cooperative load Q into shared memory ---
    for (int i = tid; i < Br * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        int n = q_start + row;
        s_Q[row * D_STRIDE + col] = (n < N) ? __ldg(Q + offset + n * d + col) : 0.0f;
    }

    // --- Initialize s_O to zero ---
    for (int i = tid; i < Br * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        s_O[row * D_STRIDE + col] = 0.0f;
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
            s_K[row * D_STRIDE + col] = (n < N) ? __ldg(K + offset + n * d + col) : 0.0f;
        }
        // Cooperative load Vj into shared memory
        for (int i = tid; i < Bc * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            int n = kv_start + row;
            s_V[row * D_STRIDE + col] = (n < N) ? __ldg(V + offset + n * d + col) : 0.0f;
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
                    dot += s_Q[qi_row * D_STRIDE + k] * s_K[j * D_STRIDE + k];
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
                s_O[qi_row * D_STRIDE + k] *= rescale;
            }
            __syncwarp();

            float exp_sum = 0.0f;

            for (int j = lane_id; j < valid_bc; j += WARP_SIZE) {
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    dot += s_Q[qi_row * D_STRIDE + k] * s_K[j * D_STRIDE + k];
                }
                float s_val = dot * sm_scale;
                float p = expf(s_val - m_new);
                exp_sum += p;

                for (int k = 0; k < d; k++) {
                    atomicAdd(s_O + qi_row * D_STRIDE + k, p * s_V[j * D_STRIDE + k]);
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
        O[offset + n * d + k] = s_O[qi_row * D_STRIDE + k] * inv_l;
    }
    if (lane_id == 0) {
        LSE[batch_idx * H * N + head_idx * N + n] = m_i + __logf(l_i);
    }
}

// ===========================================================================
// FlashAttention-2: Forward Kernel
//
// FA2 forward uses the same online-softmax algorithm as FA1.  The
// grid splits Q along rows (grid.x = N/Br) and each CTA processes
// Br rows of Q.  All K/V tiles are visited in an inner loop.
//
// The key FA2 forward improvement — splitting the KV loop across
// multiple CTAs — requires HBM-resident O/l/m with atomic updates
// across CTAs, which for fp32 is not beneficial at typical sequence
// lengths.  The current kernel therefore keeps the FA1 grid layout
// (one CTA per Q-tile) and shares the same compute structure.
// ===========================================================================

template <int Br, int Bc, int d>
__global__ void flash_attn_fwd_v2_kernel(
    const float* __restrict__ Q,
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

    constexpr int D_STRIDE = d + 1;

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
    const int qi_row   = warp_id;
    const int is_active = (qi_row < valid_br);

    const int offset = batch_idx * H * N * d + head_idx * N * d;

    extern __shared__ float smem[];
    float* s_Q = smem;
    float* s_K = smem + Br * D_STRIDE;
    float* s_V = smem + Br * D_STRIDE + Bc * D_STRIDE;
    float* s_O = smem + Br * D_STRIDE + 2 * Bc * D_STRIDE;

    for (int i = tid; i < Br * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        int n = q_start + row;
        s_Q[row * D_STRIDE + col] = (n < N) ? __ldg(Q + offset + n * d + col) : 0.0f;
    }

    for (int i = tid; i < Br * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        s_O[row * D_STRIDE + col] = 0.0f;
    }
    __syncthreads();

    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const int n_kv_tiles = (N + Bc - 1) / Bc;
    for (int kv_block = 0; kv_block < n_kv_tiles; kv_block++) {
        const int kv_start = kv_block * Bc;
        const int valid_bc = min(Bc, N - kv_start);

        for (int i = tid; i < Bc * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            int n = kv_start + row;
            s_K[row * D_STRIDE + col] = (n < N) ? __ldg(K + offset + n * d + col) : 0.0f;
        }
        for (int i = tid; i < Bc * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            int n = kv_start + row;
            s_V[row * D_STRIDE + col] = (n < N) ? __ldg(V + offset + n * d + col) : 0.0f;
        }
        __syncthreads();

        if (is_active) {
            float m_tile = -FLT_MAX;

            // FA2: compute S values and rowmax in one pass, cache S values in registers
            float s_vals[(Bc + WARP_SIZE - 1) / WARP_SIZE];
            int s_idx = 0;
            for (int j = lane_id; j < valid_bc; j += WARP_SIZE) {
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    dot += s_Q[qi_row * D_STRIDE + k] * s_K[j * D_STRIDE + k];
                }
                float s_val = dot * sm_scale;
                s_vals[s_idx++] = s_val;
                m_tile = fmaxf(m_tile, s_val);
            }

            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffff, m_tile, offset));
            }

            float m_old = m_i;
            float m_new = fmaxf(m_old, m_tile);
            float rescale = (l_i > 0.0f) ? expf(m_old - m_new) : 0.0f;

            for (int k = lane_id; k < d; k += WARP_SIZE) {
                s_O[qi_row * D_STRIDE + k] *= rescale;
            }
            __syncwarp();

            float exp_sum = 0.0f;

            // FA2: re-use cached S values (avoids recomputing Q@K^T)
            s_idx = 0;
            for (int j = lane_id; j < valid_bc; j += WARP_SIZE) {
                float p = expf(s_vals[s_idx++] - m_new);
                exp_sum += p;

                for (int k = 0; k < d; k++) {
                    atomicAdd(s_O + qi_row * D_STRIDE + k, p * s_V[j * D_STRIDE + k]);
                }
            }

            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                exp_sum += __shfl_xor_sync(0xffffffff, exp_sum, offset);
            }

            l_i = rescale * l_i + exp_sum;
            m_i = m_new;
        }

        __syncthreads();
    }

    if (!is_active) return;

    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    int n = q_start + qi_row;
    for (int k = lane_id; k < d; k += WARP_SIZE) {
        O[offset + n * d + k] = s_O[qi_row * D_STRIDE + k] * inv_l;
    }
    if (lane_id == 0) {
        LSE[batch_idx * H * N + head_idx * N + n] = m_i + __logf(l_i);
    }
}

// ===========================================================================
// FlashAttention-2: Backward Kernel
//
// Key difference from FA1:  the grid iterates over K/V tiles (one CTA
// per K/V tile) and Q tiles are the inner loop.  This means dK and
// dV are owned by a single CTA — no HBM atomics needed for them.
// dQ, however, receives contributions from multiple CTAs (one per KV
// tile) and therefore requires HBM atomicAdd.
// ===========================================================================

template <int Br, int Bc, int d>
__global__ void flash_attn_bwd_v2_kernel(
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

    constexpr int D_STRIDE = d + 1;

    const int batch_idx = blockIdx.z / H;
    const int head_idx  = blockIdx.z % H;
    const int kv_block  = blockIdx.x;    // KV-tile index

    const int kv_start = kv_block * Bc;
    const int valid_bc = min(Bc, N - kv_start);
    if (valid_bc <= 0) return;

    const int tid      = threadIdx.x;
    const int num_thread  = blockDim.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;

    const int offset = batch_idx * H * N * d + head_idx * N * d;
    const int lse_offset = batch_idx * H * N + head_idx * N;

    extern __shared__ float smem[];
    float* s_K   = smem;                                                       // Bc * D_STRIDE
    float* s_V   = smem + Bc * D_STRIDE;                                       // Bc * D_STRIDE
    float* s_dK  = smem + 2 * Bc * D_STRIDE;                                   // Bc * D_STRIDE
    float* s_dV  = smem + 3 * Bc * D_STRIDE;                                   // Bc * D_STRIDE
    float* s_Q   = smem + 4 * Bc * D_STRIDE;                                   // Br * D_STRIDE
    float* s_dO  = smem + 4 * Bc * D_STRIDE + Br * D_STRIDE;                   // Br * D_STRIDE
    float* s_LSE = smem + 4 * Bc * D_STRIDE + 2 * Br * D_STRIDE;               // Br
    float* s_D   = smem + 4 * Bc * D_STRIDE + 2 * Br * D_STRIDE + Br;          // Br
    float* s_dQ  = smem + 4 * Bc * D_STRIDE + 2 * Br * D_STRIDE + 2 * Br;     // Br * D_STRIDE

    // Load K_j, V_j into SMEM (loaded once for this CTA)
    for (int i = tid; i < Bc * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        int n = kv_start + row;
        s_K[row * D_STRIDE + col] = (n < N) ? __ldg(K + offset + n * d + col) : 0.0f;
    }
    for (int i = tid; i < Bc * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        int n = kv_start + row;
        s_V[row * D_STRIDE + col] = (n < N) ? __ldg(V + offset + n * d + col) : 0.0f;
    }

    // Initialize dK_j, dV_j to zero in SMEM
    for (int i = tid; i < Bc * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        s_dK[row * D_STRIDE + col] = 0.0f;
        s_dV[row * D_STRIDE + col] = 0.0f;
    }
    __syncthreads();

    // Main loop over Q tiles
    const int n_q_tiles = (N + Br - 1) / Br;
    for (int q_block = 0; q_block < n_q_tiles; q_block++) {
        const int q_start = q_block * Br;
        const int valid_br = min(Br, N - q_start);
        const int qi_row = warp_id;
        const int is_active = (qi_row < valid_br);

        // Load Q_i, dO_i
        for (int i = tid; i < Br * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            int n = q_start + row;
            if (n < N) {
                s_Q[row * D_STRIDE + col]  = __ldg(Q + offset + n * d + col);
                s_dO[row * D_STRIDE + col] = __ldg(dO + offset + n * d + col);
            } else {
                s_Q[row * D_STRIDE + col]  = 0.0f;
                s_dO[row * D_STRIDE + col] = 0.0f;
            }
        }
        // Load LSE_i, D_i
        for (int i = tid; i < Br; i += num_thread) {
            int n = q_start + i;
            s_LSE[i] = (n < N) ? __ldg(LSE + lse_offset + n) : 0.0f;
            s_D[i]   = (n < N) ? __ldg(D + lse_offset + n) : 0.0f;
        }
        // Clear s_dQ for this Q tile
        for (int i = tid; i < Br * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            s_dQ[row * D_STRIDE + col] = 0.0f;
        }
        __syncthreads();

        if (is_active) {
            float lse_i = s_LSE[qi_row];
            float d_i   = s_D[qi_row];

            float dQ_acc[d];
            #pragma unroll
            for (int k = 0; k < d; k++) dQ_acc[k] = 0.0f;

            for (int j = lane_id; j < valid_bc; j += WARP_SIZE) {
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    dot += s_Q[qi_row * D_STRIDE + k] * s_K[j * D_STRIDE + k];
                }
                float s_val = dot * sm_scale;
                float p = expf(s_val - lse_i);

                float dP_val = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    dP_val += s_dO[qi_row * D_STRIDE + k] * s_V[j * D_STRIDE + k];
                }
                float dS = sm_scale * p * (dP_val - d_i);

                // dQ: accumulate to warp-local registers
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    dQ_acc[k] += dS * s_K[j * D_STRIDE + k];
                }

                // dK: atomicAdd in SMEM (multiple warps may hit same K row)
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    atomicAdd(&s_dK[j * D_STRIDE + k], dS * s_Q[qi_row * D_STRIDE + k]);
                }

                // dV: atomicAdd in SMEM (multiple warps may hit same V row)
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    atomicAdd(&s_dV[j * D_STRIDE + k], p * s_dO[qi_row * D_STRIDE + k]);
                }
            }

            // Warp-reduce dQ_acc
            #pragma unroll
            for (int k = 0; k < d; k++) {
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    dQ_acc[k] += __shfl_xor_sync(0xffffffff, dQ_acc[k], offset);
                }
            }

            // Store dQ_i contribution to SMEM
            for (int k = lane_id; k < d; k += WARP_SIZE) {
                s_dQ[qi_row * D_STRIDE + k] = dQ_acc[k];
            }
        }

        __syncthreads();

        // Write dQ_i to HBM via atomicAdd (multiple CTAs contribute to same Q row)
        for (int i = tid; i < valid_br * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            int n = q_start + row;
            atomicAdd(&dQ[offset + n * d + col], s_dQ[row * D_STRIDE + col]);
        }

        __syncthreads();
    }

    // Write dK_j, dV_j to HBM — no atomics, this CTA owns these rows
    for (int i = tid; i < valid_bc * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        int n = kv_start + row;
        dK[offset + n * d + col] = s_dK[row * D_STRIDE + col];
        dV[offset + n * d + col] = s_dV[row * D_STRIDE + col];
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

    constexpr int D_STRIDE = d + 1;  // +1 pad avoids bank conflicts

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
    float* s_Q   = smem;                                                          // Br * D_STRIDE
    float* s_dO  = smem + Br * D_STRIDE;                                          // Br * D_STRIDE
    float* s_LSE = smem + 2 * Br * D_STRIDE;                                      // Br (scalars, no padding)
    float* s_D   = smem + 2 * Br * D_STRIDE + Br;                                 // Br (scalars, no padding)
    float* s_K   = smem + 2 * Br * D_STRIDE + 2 * Br;                             // Bc * D_STRIDE
    float* s_V   = smem + 2 * Br * D_STRIDE + 2 * Br + Bc * D_STRIDE;             // Bc * D_STRIDE
    float* s_dQ  = smem + 2 * Br * D_STRIDE + 2 * Br + 2 * Bc * D_STRIDE;         // Br * D_STRIDE

    // --- Cooperative load Qi, dOi into shared memory ---
    for (int i = tid; i < Br * d; i += num_thread) {
        int row = i / d;
        int col = i % d;
        int n = q_start + row;
        if (n < N) {
            s_Q[row * D_STRIDE + col]  = __ldg(Q + offset + n * d + col);
            s_dO[row * D_STRIDE + col] = __ldg(dO + offset + n * d + col);
        } else {
            s_Q[row * D_STRIDE + col]  = 0.0f;
            s_dO[row * D_STRIDE + col] = 0.0f;
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
        int row = i / d;
        int col = i % d;
        s_dQ[row * D_STRIDE + col] = 0.0f;
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
            s_K[row * D_STRIDE + col] = (n < N) ? __ldg(K + offset + n * d + col) : 0.0f;
        }
        for (int i = tid; i < Bc * d; i += num_thread) {
            int row = i / d;
            int col = i % d;
            int n = kv_start + row;
            s_V[row * D_STRIDE + col] = (n < N) ? __ldg(V + offset + n * d + col) : 0.0f;
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
                    dot += s_Q[qi_row * D_STRIDE + k] * s_K[j * D_STRIDE + k];
                }
                float s_val = dot * sm_scale;
                float p = expf(s_val - lse_i);

                float dP_val = 0.0f;
                #pragma unroll
                for (int k = 0; k < d; k++) {
                    dP_val += s_dO[qi_row * D_STRIDE + k] * s_V[j * D_STRIDE + k];
                }
                float dS = sm_scale * p * (dP_val - d_i);

                int n_j = kv_start + j;
                for (int k = 0; k < d; k++) {
                    dQ_acc[k] += dS * s_K[j * D_STRIDE + k];
                    atomicAdd(dV + offset + n_j * d + k, p * s_dO[qi_row * D_STRIDE + k]);
                    atomicAdd(dK + offset + n_j * d + k, dS * s_Q[qi_row * D_STRIDE + k]);
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
                s_dQ[qi_row * D_STRIDE + k] += dQ_acc[k];
            }
        }

        __syncthreads();
    }

    // --- Write dQ to global memory (active warps only) ---
    if (is_active) {
        int n = q_start + qi_row;
        for (int k = lane_id; k < d; k += WARP_SIZE) {
            dQ[offset + n * d + k] = s_dQ[qi_row * D_STRIDE + k];
        }
    }
}

// ---------------------------------------------------------------------------
// Host launch wrappers with template instantiation for d = 32, 64, 128
//
// D_STRIDE = d + 1 avoids shared-memory bank conflicts (d is always a
// multiple of 32, so d%32==0 causes all threads in a warp to hit the
// same bank when accessing s_K[j*d+k] and s_V[j*d+k]).
// ---------------------------------------------------------------------------

#define DISPATCH_FWD(Br, Bc, d_val)                                         \
    case d_val: {                                                            \
        int _n_blocks = (N + Br - 1) / Br;                                   \
        dim3 grid(_n_blocks, 1, B * H);                                      \
        dim3 block(Br * WARP_SIZE, 1, 1);                                    \
        constexpr int _d_s = d_val + 1;                                      \
        int smem_bytes = (Br + 2 * Bc + Br) * _d_s * sizeof(float);         \
        flash_attn_fwd_kernel<Br, Bc, d_val>                                 \
            <<<grid, block, smem_bytes, stream>>>(                           \
                Q, K, V, O, LSE, B, H, N, sm_scale);                        \
        break;                                                               \
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

#define DISPATCH_BWD(Br, Bc, d_val)                                                            \
    case d_val: {                                                                               \
        int _n_blocks = (N + Br - 1) / Br;                                                      \
        dim3 grid(_n_blocks, 1, B * H);                                                          \
        dim3 block(Br * WARP_SIZE, 1, 1);                                                       \
        constexpr int _d_s = d_val + 1;                                                          \
        int smem_bytes = (3 * Br * _d_s + 2 * Bc * _d_s + 2 * Br) * sizeof(float);             \
        flash_attn_bwd_kernel<Br, Bc, d_val>                                                    \
            <<<grid, block, smem_bytes, stream>>>(                                              \
                dO, Q, K, V, O, LSE, D, dQ, dK, dV, B, H, N, sm_scale);                       \
        break;                                                                                  \
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

// ===========================================================================
// FlashAttention-2 Host launch wrappers
// ===========================================================================

#define DISPATCH_FWD_V2(Br, Bc, d_val)                                        \
    case d_val: {                                                              \
        int _n_blocks = (N + Br - 1) / Br;                                     \
        dim3 grid(_n_blocks, 1, B * H);                                        \
        dim3 block(Br * WARP_SIZE, 1, 1);                                      \
        constexpr int _d_s = d_val + 1;                                        \
        int smem_bytes = (Br + 2 * Bc + Br) * _d_s * sizeof(float);           \
        flash_attn_fwd_v2_kernel<Br, Bc, d_val>                                \
            <<<grid, block, smem_bytes, stream>>>(                             \
                Q, K, V, O, LSE, B, H, N, sm_scale);                          \
        break;                                                                 \
    }

void flash_attn_forward_v2(
    const float* Q, const float* K, const float* V,
    float* O, float* LSE,
    int B, int H, int N, int d,
    float sm_scale, cudaStream_t stream
) {
    switch (d) {
        DISPATCH_FWD_V2(32, 32, 32)
        DISPATCH_FWD_V2(32, 32, 64)
        DISPATCH_FWD_V2(16, 32, 128)
        default:
            fprintf(stderr, "flash_attn_fwd_v2: unsupported d=%d\n", d);
            break;
    }
}
#undef DISPATCH_FWD_V2

// ---------------------------------------------------------------------------

#define DISPATCH_BWD_V2(Br, Bc, d_val)                                                           \
    case d_val: {                                                                                 \
        int _n_blocks = (N + Bc - 1) / Bc;                                                       \
        dim3 grid(_n_blocks, 1, B * H);                                                           \
        dim3 block(Br * WARP_SIZE, 1, 1);                                                         \
        constexpr int _d_s = d_val + 1;                                                           \
        int smem_bytes = ((4 * Bc + 3 * Br) * _d_s + 2 * Br) * sizeof(float);                   \
        flash_attn_bwd_v2_kernel<Br, Bc, d_val>                                                   \
            <<<grid, block, smem_bytes, stream>>>(                                                \
                dO, Q, K, V, O, LSE, D, dQ, dK, dV, B, H, N, sm_scale);                         \
        break;                                                                                    \
    }

void flash_attn_backward_v2(
    const float* dO, const float* Q, const float* K, const float* V,
    const float* O, const float* LSE,
    float* dQ, float* dK, float* dV,
    int B, int H, int N, int d,
    float sm_scale, cudaStream_t stream
) {
    int total_N = B * H * N;

    float* D;
    cudaMalloc(&D, total_N * sizeof(float));
    int block_dim = 256;
    int grid_dim = (total_N + block_dim - 1) / block_dim;
    compute_D_kernel<<<grid_dim, block_dim, 0, stream>>>(dO, O, D, total_N, d);

    switch (d) {
        DISPATCH_BWD_V2(32, 32, 32)
        DISPATCH_BWD_V2(16, 32, 64)
        DISPATCH_BWD_V2(8, 16, 128)
        default:
            fprintf(stderr, "flash_attn_bwd_v2: unsupported d=%d\n", d);
            break;
    }

    cudaFree(D);
}
#undef DISPATCH_BWD_V2
