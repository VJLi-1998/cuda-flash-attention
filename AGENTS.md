# AGENTS.md — FlashAttention-1 CUDA from Scratch

## Build

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
python setup.py develop          # NOT pip install -e .
```

`setup.py` bypasses CUDA version checks (`TORCH_CUDA_ARCH_LIST=""` + monkey-patched `_check_cuda_version`) and auto-detects GPU compute capability (fallback: `89`). The extension compiles as `flash_attn._C`.

## Package & entrypoints

- `flash_attn/__init__.py` exports: `FlashAttnFunction`, `flash_attention`, `naive_attention_forward`, `naive_attention_backward`, `NaiveAttention`
- `flash_attn/flash_attn_autograd.py` — `torch.autograd.Function` that calls `_C.forward` / `_C.backward`
- `flash_attn/naive_attn.py` — reference PyTorch implementation for correctness testing
- `flash_attn/csrc/flash_attn.cu` — CUDA kernels (forward + backward + `compute_D`)
- `flash_attn/csrc/bindings.cpp` — pybind11 glue (`_C` module)

## Constraints

- **d ∈ {32, 64, 128}** only. Template dispatched via `switch(d)` with different tile sizes:
  - `d=32`: Br=32 Bc=32 (fwd+bwd)
  - `d=64`: Br=32 Bc=32 (fwd+bwd)
  - `d=128`: Br=16 Bc=32 (fwd), Br=16 Bc=16 (bwd)
- All tensors must be **CUDA, contiguous, float32**. The Python wrappers ensure contiguity; no fp16/bf16 support.
- **1 warp per Q row** (32 threads). CTA size = `Br * 32`.

## Test

```bash
python flash_attn/tests/test_flash_attn.py     # direct run
# or
python -m pytest flash_attn/tests/test_flash_attn.py -v
```

Tests: forward correctness, backward correctness, `torch.autograd.gradcheck`, vs `F.scaled_dot_product_attention`. Tolerances are relaxed (`rtol=1e-2, atol=1e-3`) due to fp32 tiling accumulation differences. No CI configured.

## Benchmark & profile

```bash
python benchmark.py               # FA1 vs naive vs SDPA
python profile.py                 # forward + backward timing, ncu-friendly
ncu --kernel-name="flash_attn_fwd" --set full python profile.py
```

## Architecture notes

- Backward pass precomputes `D = rowsum(dO * O)` in a separate `compute_D_kernel` before launching the main backward kernel — this is allocated and freed per call via `cudaMalloc`/`cudaFree`.
- `atomicAdd` is used for `dV` and `dK` accumulation (correctness over raw speed); multiple Q tiles can race on the same K/V row.
- `__ldg` (read-only cache) for all HBM loads; `__shfl_xor_sync` for warp-level reductions.
- No CI, no linter, no formatter, no typechecker.
