from __future__ import print_function
import torch

print(torch.__version__)

x = torch.ones(2, 2, requires_grad=True)

print(x)

y = x + 2
y.requires_grad_(True)
print(y)

z = y * y * 3
out = z.mean()
print(out)
z.requires_grad_(True)

out.backward()
print(x.grad)
print(z.grad)

