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


def flash_attention(Q, K, V, sm_scale=None):
    """Drop-in replacement for scaled dot-product attention."""
    return FlashAttnFunction.apply(Q, K, V, sm_scale)
