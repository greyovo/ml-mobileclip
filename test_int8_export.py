import torch
import bitsandbytes as bnb

# test if we can extract int8 weights and create a custom module for ONNX export
class Int8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('int8_weight', torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8))
        self.register_buffer('scale', torch.ones(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None
    
    def forward(self, x):
        # dequantize and matmul
        weight_fp = self.int8_weight.float() * self.scale.unsqueeze(1)
        out = torch.matmul(x, weight_fp.t())
        if self.bias is not None:
            out = out + self.bias
        return out

m = Int8Linear(10, 10)
x = torch.randn(2, 10)
out = m(x)
print('Int8Linear forward works:', out.shape)

try:
    torch.onnx.export(m, (x,), '/tmp/test_int8_custom.onnx', dynamo=False, opset_version=14)
    print('ONNX export succeeded')
except Exception as e:
    print('ONNX export failed:', e)
