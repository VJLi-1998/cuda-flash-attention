"""
Profile helper for FlashAttention kernel.
Run: python profile.py

roofline 分析（算力/带宽利用率）：
ncu --set full --launch-skip 10 --launch-count 1 python profile.py
只看我们的 kernel（过滤掉 PyTorch 的）：
ncu --kernel-name="flash_attn_fwd" --set full python profile.py
"""

import torch
import math
from flash_attn import flash_attention

B, H, N, d = 1, 4, 512, 64
sm_scale = 1.0 / math.sqrt(d)

torch.manual_seed(42)
Q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
K = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
V = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

# Warmup
for _ in range(3):
    O = flash_attention(Q, K, V, sm_scale)

# Profile: forward
torch.cuda.cudart().cudaProfilerStart()
O = flash_attention(Q, K, V, sm_scale)
torch.cuda.cudart().cudaProfilerStop()

# Profile: forward + backward
Q_g = Q.clone().requires_grad_()
K_g = K.clone().requires_grad_()
V_g = V.clone().requires_grad_()
torch.cuda.cudart().cudaProfilerStart()
O = flash_attention(Q_g, K_g, V_g, sm_scale)
O.sum().backward()
torch.cuda.cudart().cudaProfilerStop()

print(f"Done. O shape: {O.shape}")
