"""
FlashAttention correctness tests and gradient checks.

Run: python -m pytest flash_attn/tests/test_flash_attn.py -v
"""

import math
import torch
import torch.nn.functional as F
from flash_attn.naive_attn import (
    naive_attention_forward,
    naive_attention_backward,
    NaiveAttention,
)
from flash_attn.flash_attn_autograd import FlashAttnFunction, flash_attention

# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
TEST_CONFIGS = [
    # (B, H, N, d)
    (1, 1, 64, 32),
    (1, 1, 64, 64),
    (1, 2, 128, 32),
    (1, 2, 128, 64),
    (2, 4, 256, 32),
    (2, 4, 256, 64),
]

RTOL = 1e-2  # relaxed due to fp32 accumulation differences in tiling
ATOL = 1e-3
GRAD_RTOL = 1e-2
GRAD_ATOL = 1e-3


def make_inputs(B, H, N, d):
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32, requires_grad=True)
    K = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32, requires_grad=True)
    V = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32, requires_grad=True)
    sm_scale = 1.0 / math.sqrt(d)
    return Q, K, V, sm_scale


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------
def test_forward_correctness():
    for B, H, N, d in TEST_CONFIGS:
        Q, K, V, sm_scale = make_inputs(B, H, N, d)

        O_ref = naive_attention_forward(Q, K, V, sm_scale)
        O_fa = flash_attention(Q.clone(), K.clone(), V.clone(), sm_scale)

        max_diff = (O_ref - O_fa).abs().max().item()
        cos_sim = F.cosine_similarity(
            O_ref.reshape(-1), O_fa.reshape(-1), dim=0
        ).item()

        assert torch.allclose(O_ref, O_fa, rtol=RTOL, atol=ATOL), (
            f"Forward mismatch: B={B}, H={H}, N={N}, d={d}, "
            f"max_diff={max_diff:.6f}, cos_sim={cos_sim:.6f}"
        )
        print(f"  [PASS] forward B={B} H={H} N={N} d={d}  max_diff={max_diff:.2e}  cos={cos_sim:.6f}")


# ---------------------------------------------------------------------------
# Backward correctness (gradients match reference)
# ---------------------------------------------------------------------------
def test_backward_correctness():
    for B, H, N, d in TEST_CONFIGS:
        torch.manual_seed(42)
        sm_scale = 1.0 / math.sqrt(d)

        Q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32, requires_grad=True)
        K = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32, requires_grad=True)
        V = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32, requires_grad=True)

        dQ_ref, dK_ref, dV_ref = naive_attention_backward(
            torch.ones_like(Q), Q, K, V, sm_scale
        )

        Q_clone = Q.clone().detach().requires_grad_(True)
        K_clone = K.clone().detach().requires_grad_(True)
        V_clone = V.clone().detach().requires_grad_(True)

        O_fa = flash_attention(Q_clone, K_clone, V_clone, sm_scale)
        O_fa.sum().backward()

        for name, g_ref, g_fa in [
            ("dQ", dQ_ref, Q_clone.grad),
            ("dK", dK_ref, K_clone.grad),
            ("dV", dV_ref, V_clone.grad),
        ]:
            assert g_fa is not None, f"{name}.grad is None"
            max_diff = (g_ref - g_fa).abs().max().item()
            assert torch.allclose(g_ref, g_fa, rtol=GRAD_RTOL, atol=GRAD_ATOL), (
                f"{name} mismatch: B={B} H={H} N={N} d={d}, max_diff={max_diff:.6f}"
            )
        print(f"  [PASS] backward B={B} H={H} N={N} d={d}")


# ---------------------------------------------------------------------------
# Gradient check (torch.autograd.gradcheck)
# ---------------------------------------------------------------------------
def test_gradient_check():
    d_head = 32
    B, H, N = 1, 2, 128
    sm_scale = 1.0 / math.sqrt(d_head)

    torch.manual_seed(123)
    Q = torch.randn(B, H, N, d_head, device="cuda", dtype=torch.float64, requires_grad=True)
    K = torch.randn(B, H, N, d_head, device="cuda", dtype=torch.float64, requires_grad=True)
    V = torch.randn(B, H, N, d_head, device="cuda", dtype=torch.float64, requires_grad=True)

    # Convert to float32 for the actual kernel (CUDA kernel is fp32)
    Q_f32 = Q.float().detach().requires_grad_()
    K_f32 = K.float().detach().requires_grad_()
    V_f32 = V.float().detach().requires_grad_()

    is_correct = torch.autograd.gradcheck(
        FlashAttnFunction.apply,
        (Q_f32, K_f32, V_f32, sm_scale),
        eps=1e-2,
        atol=GRAD_ATOL,
        rtol=GRAD_RTOL,
    )
    assert is_correct, "gradcheck failed"
    print(f"  [PASS] gradcheck B={B} H={H} N={N} d={d_head}")


# ---------------------------------------------------------------------------
# Compare with PyTorch built-in scaled_dot_product_attention
# ---------------------------------------------------------------------------
def test_vs_pytorch_sdpa():
    for B, H, N, d in TEST_CONFIGS:
        Q, K, V, sm_scale = make_inputs(B, H, N, d)

        O_ref = F.scaled_dot_product_attention(Q, K, V, scale=sm_scale)
        O_fa = flash_attention(Q.clone(), K.clone(), V.clone(), sm_scale)

        max_diff = (O_ref - O_fa).abs().max().item()
        assert torch.allclose(O_ref, O_fa, rtol=RTOL, atol=ATOL), (
            f"vs sdpa mismatch: B={B} H={H} N={N} d={d}, max_diff={max_diff:.6f}"
        )
        print(f"  [PASS] vs_sdpa B={B} H={H} N={N} d={d}  max_diff={max_diff:.2e}")


if __name__ == "__main__":
    print("=" * 60)
    print("FlashAttention-1 Correctness Tests")
    print("=" * 60)

    print("\n--- Forward Correctness ---")
    test_forward_correctness()

    print("\n--- Backward Correctness ---")
    test_backward_correctness()

    print("\n--- Gradient Check ---")
    test_gradient_check()

    print("\n--- vs PyTorch SDPA ---")
    test_vs_pytorch_sdpa()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
