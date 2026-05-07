"""
Profile helper for FlashAttention-1 and FlashAttention-2 kernels.

Run: python profile.py

roofline analysis (compute / bandwidth utilisation):
ncu --set full --launch-skip 10 --launch-count 1 python profile.py

Only profile our kernels (filter out PyTorch):
ncu --kernel-name="flash_attn_fwd" --set full python profile.py
ncu --kernel-name="flash_attn_fwd_v2" --set full python profile.py
ncu --kernel-name="flash_attn_bwd_v2" --set full python profile.py
"""

import torch
import math
from flash_attn import flash_attention, flash_attention_v2

B, H, N, d = 1, 4, 512, 64
sm_scale = 1.0 / math.sqrt(d)

torch.manual_seed(42)
Q = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
K = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)
V = torch.randn(B, H, N, d, device="cuda", dtype=torch.float32)

# Warmup
for _ in range(3):
    _ = flash_attention(Q, K, V, sm_scale)
    _ = flash_attention_v2(Q, K, V, sm_scale)

# Profile FA1 forward
print("Profiling FA1 forward...")
torch.cuda.cudart().cudaProfilerStart()
O1 = flash_attention(Q, K, V, sm_scale)
torch.cuda.cudart().cudaProfilerStop()
print(f"  FA1 O shape: {O1.shape}")

# Profile FA2 forward
print("Profiling FA2 forward...")
torch.cuda.cudart().cudaProfilerStart()
O2 = flash_attention_v2(Q, K, V, sm_scale)
torch.cuda.cudart().cudaProfilerStop()
print(f"  FA2 O shape: {O2.shape}")

# Profile FA1 forward + backward
print("Profiling FA1 forward + backward...")
Q_g1 = Q.clone().requires_grad_()
K_g1 = K.clone().requires_grad_()
V_g1 = V.clone().requires_grad_()
torch.cuda.cudart().cudaProfilerStart()
O1 = flash_attention(Q_g1, K_g1, V_g1, sm_scale)
O1.sum().backward()
torch.cuda.cudart().cudaProfilerStop()
print(f"  FA1 fwd+bwd done")

# Profile FA2 forward + backward
print("Profiling FA2 forward + backward...")
Q_g2 = Q.clone().requires_grad_()
K_g2 = K.clone().requires_grad_()
V_g2 = V.clone().requires_grad_()
torch.cuda.cudart().cudaProfilerStart()
O2 = flash_attention_v2(Q_g2, K_g2, V_g2, sm_scale)
O2.sum().backward()
torch.cuda.cudart().cudaProfilerStop()
print(f"  FA2 fwd+bwd done")

print("Done.")
