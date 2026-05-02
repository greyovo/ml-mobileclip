import torch
import bitsandbytes as bnb

# test understanding of Linear8bitLt quantization
m = bnb.nn.Linear8bitLt(10, 10, has_fp16_weights=False)
original = torch.randn(10, 10) * 0.1
m.weight.data = original.clone()
x = torch.randn(2, 10)
out = m(x)

CB = m.state.CB
SCB = m.state.SCB

# Try different reconstruction formulas
# Formula 1: CB * SCB.unsqueeze(1) / 127
r1 = CB.float() * SCB.unsqueeze(1) / 127.0
print('Formula 1 (CB * SCB / 127) diff max:', (r1 - original).abs().max().item())

# Formula 2: CB * SCB.unsqueeze(1)  
r2 = CB.float() * SCB.unsqueeze(1)
print('Formula 2 (CB * SCB) diff max:', (r2 - original).abs().max().item())

# Formula 3: maybe SCB is already the scale factor including 1/127
# Let's check: if CB values are in [-127, 127], then max(CB) = 127
# So weight = CB * (max_weight / 127) = CB * scale
# And SCB should be max_weight / 127
print('\nSCB values:', SCB)
print('max abs original per row:', original.abs().max(dim=1)[0])
print('ratio:', original.abs().max(dim=1)[0] / SCB)
