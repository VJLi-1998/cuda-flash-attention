# FlashAttention-1: CUDA Implementation from Scratch

Handwritten CUDA C++ implementation of **FlashAttention** (Dao et al., NeurIPS 2022),
including both forward and backward passes with exact gradient computation.

> For resume/interview: demonstrates CUDA kernel programming, GPU memory hierarchy
> optimization, online softmax, IO-aware tiling, and autograd integration with PyTorch.

## Algorithm Overview

Standard attention has O(N²) memory complexity from the N×N attention matrix.
FlashAttention tiles Q, K, V into blocks that fit in SRAM, achieving O(N) memory
while remaining mathematically exact.

### Forward Pass

1. Split Q into blocks of size `Br × d` (Q tiles)
2. For each Q tile, iterate over K/V blocks of size `Bc × d`
3. Use **online softmax** to maintain running max (`m`) and sum-exp (`l`) per row
4. Accumulate output `O` in-place within SRAM
5. Store `LSE = m + log(l)` for the backward pass

```
for i in Q_tiles:
    O_i, m_i, l_i = 0, -inf, 0
    for j in KV_tiles:
        S_ij = Q_i @ K_j^T / sqrt(d)          # [Br, Bc]
        m_ij = rowmax(S_ij)
        m_new = max(m_i, m_ij)
        P_ij = exp(S_ij - m_new)
        O_i = exp(m_i - m_new) * O_i + P_ij @ V_j
        l_i = exp(m_i - m_new) * l_i + rowsum(P_ij)
        m_i = m_new
    O_i = O_i / l_i
    LSE_i = m_i + log(l_i)
```

### Backward Pass (Recomputation)

Key insight: instead of storing the N×N attention matrix from forward,
recompute it in backward using stored LSE:

```
D = rowsum(dO * O)                    # precompute once per row

for i in Q_tiles:
    dQ_i = 0
    for j in KV_tiles:
        S_ij = Q_i @ K_j^T / sqrt(d)
        P_ij = exp(S_ij - LSE_i)      # reconstruct softmax weights
        dV_j += P_ij^T @ dO_i
        dS_ij = P_ij * (dO_i @ V_j^T - D_i) * (1/sqrt(d))
        dQ_i += dS_ij @ K_j
        dK_j += dS_ij^T @ Q_i
```

## Code Architecture

```
flash_attn/
├── csrc/
│   ├── flash_attn.h          # CUDA kernel declarations
│   ├── flash_attn.cu         # Forward + backward CUDA kernels
│   └── bindings.cpp          # PyTorch C++ extension (pybind11)
├── __init__.py               # Package exports
├── flash_attn_autograd.py    # torch.autograd.Function wrapper
├── naive_attn.py             # Reference PyTorch implementation
└── tests/
    └── test_flash_attn.py    # Correctness, gradient check, vs SDPA
```

### CUDA Kernel Design

- **Br = 32, Bc = 32** (Q/KV tile sizes, configurable via template)
- **1 warp per Q row**: each warp (32 threads) handles one row of a Q tile
- **Cooperative shared memory loads**: all threads load Q/K/V tiles
- **Warp-level reductions**: `__shfl_xor_sync` for row max and sum-exp
- **AtomicAdd** for dV/dK accumulation (correctness over raw speed)
- **Template dispatch** for `d ∈ {32, 64, 128}` head dimensions
- Shared memory: `(3*Br + 2*Bc)*d*4` bytes per CTA

### Supported Configurations

| Head dim (d) | Q tile (Br) | KV tile (Bc) | Shared mem |
|-------------|-------------|--------------|------------|
| 32          | 32          | 32           | ~20 KB     |
| 64          | 32          | 32           | ~28 KB     |
| 128         | 32          | 32           | ~44 KB     |

## Build & Run

```bash
# Install PyTorch first (CUDA 12.x):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Build the CUDA extension:
cd flash-attention
python setup.py develop

# Run tests:
python flash_attn/tests/test_flash_attn.py

# Benchmark:
python benchmark.py
```

## Test Coverage

| Test | Description |
|------|-------------|
| Forward correctness | Matches naive reference (fp32) |
| Backward correctness | dQ, dK, dV match reference gradients |
| Gradient check | `torch.autograd.gradcheck` passes |
| vs PyTorch SDPA | Matches `F.scaled_dot_product_attention` |

## Key Interview Topics

When discussing this project, be prepared to explain:

1. **IO complexity**: Why FlashAttention is O(N²d²/M) in HBM accesses vs O(N²d) for naive
2. **Online softmax**: Why `exp(m_old - m_new)` rescaling preserves numerical stability
3. **Why recompute in backward**: Trading compute for memory — O(N²) HBM writes avoided
4. **Shared memory constraints**: Tile size limits from SRAM capacity (48-100KB)
5. **Race conditions**: Why `atomicAdd` is needed for dV/dK (multiple Q tiles contribute to the same K/V row)
6. **Warp-level primitives**: `__shfl_xor_sync` for intra-warp reductions

## References

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
  [NeurIPS 2022](https://arxiv.org/abs/2205.14135)
- [Official implementation](https://github.com/Dao-AILab/flash-attention)
- [Tri Dao's CUDA MODE lecture](https://www.youtube.com/watch?v=gMOwz3n3sDc)
