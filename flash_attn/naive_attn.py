import math
import torch
import torch.nn.functional as F


def naive_attention_forward(Q, K, V, sm_scale=None):
    """
    Standard attention with full O(N^2) memory.
    Q, K, V: (B, H, N, d)
    Returns: O (B, H, N, d)
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Q.size(-1))
    S = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    P = F.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    return O


def naive_attention_backward(dO, Q, K, V, sm_scale=None):
    """
    Reference backward pass for gradient checking.
    Returns: dQ, dK, dV
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(Q.size(-1))
    B, H, N, d = Q.shape

    S = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    P = F.softmax(S, dim=-1)
    O = torch.matmul(P, V)

    dP = torch.matmul(dO, V.transpose(-2, -1))
    D = torch.sum(dO * O, dim=-1, keepdim=True)
    dS = sm_scale * P * (dP - D)

    dQ = torch.matmul(dS, K)
    dK = torch.matmul(dS.transpose(-2, -1), Q)
    dV = torch.matmul(P.transpose(-2, -1), dO)

    return dQ, dK, dV


class NaiveAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, sm_scale):
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(Q.size(-1))
        ctx.sm_scale = sm_scale
        S = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
        P = F.softmax(S, dim=-1)
        O = torch.matmul(P, V)
        ctx.save_for_backward(Q, K, V, P)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, P = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        dP = torch.matmul(dO, V.transpose(-2, -1))
        D = torch.sum(dO * torch.matmul(P, V), dim=-1, keepdim=True)
        dS = sm_scale * P * (dP - D)
        dQ = torch.matmul(dS, K)
        dK = torch.matmul(dS.transpose(-2, -1), Q)
        dV = torch.matmul(P.transpose(-2, -1), dO)
        return dQ, dK, dV, None
