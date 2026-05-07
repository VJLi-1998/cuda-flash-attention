import math
import torch
import flash_attn._C as _C


class FlashAttnFunction(torch.autograd.Function):
    """FlashAttention-1 with recomputation in backward pass."""

    @staticmethod
    def forward(ctx, Q, K, V, sm_scale=None):
        B, H, N, d = Q.shape
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(d)

        Q_ = Q.contiguous() if not Q.is_contiguous() else Q
        K_ = K.contiguous() if not K.is_contiguous() else K
        V_ = V.contiguous() if not V.is_contiguous() else V

        O, LSE = _C.forward(Q_, K_, V_, sm_scale)

        ctx.save_for_backward(Q_, K_, V_, O, LSE)
        ctx.sm_scale = sm_scale
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, LSE = ctx.saved_tensors
        dQ, dK, dV = _C.backward(
            dO.contiguous(), Q, K, V, O, LSE, ctx.sm_scale
        )
        return dQ, dK, dV, None


class FlashAttnFunctionV2(torch.autograd.Function):
    """FlashAttention-2.

    Backward pass has KV-tile outer loop, eliminating HBM atomics for
    dK and dV (they are accumulated in SMEM within each CTA).  dQ uses
    HBM atomics since multiple CTAs contribute to the same Q row.
    """

    @staticmethod
    def forward(ctx, Q, K, V, sm_scale=None):
        B, H, N, d = Q.shape
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(d)

        Q_ = Q.contiguous() if not Q.is_contiguous() else Q
        K_ = K.contiguous() if not K.is_contiguous() else K
        V_ = V.contiguous() if not V.is_contiguous() else V

        O, LSE = _C.forward_v2(Q_, K_, V_, sm_scale)

        ctx.save_for_backward(Q_, K_, V_, O, LSE)
        ctx.sm_scale = sm_scale
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, LSE = ctx.saved_tensors
        dQ, dK, dV = _C.backward_v2(
            dO.contiguous(), Q, K, V, O, LSE, ctx.sm_scale
        )
        return dQ, dK, dV, None


def flash_attention(Q, K, V, sm_scale=None):
    """Drop-in replacement for scaled dot-product attention (FA1)."""
    return FlashAttnFunction.apply(Q, K, V, sm_scale)


def flash_attention_v2(Q, K, V, sm_scale=None):
    """FlashAttention-2 drop-in replacement."""
    return FlashAttnFunctionV2.apply(Q, K, V, sm_scale)
