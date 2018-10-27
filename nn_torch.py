import torch

device = torch.device("cuda")

batch = 1
dim_in = 64
dim_hid = 16
dim_out = 4
learn_rate = 1e-6

x = torch.randn(batch, dim_in, device=device)
y = torch.randn(batch, dim_out, device=device)

w1 = torch.randn(dim_in, dim_hid, device=device)
w2 = torch.randn(dim_hid, dim_out, device=device)

a = torch.mm(x, w1)
b = a.clamp(min=0)
c = torch.mm(b, w2)

print(c)

for i in range(10000):
    a = torch.mm(x, w1)
    b = a.clamp(min=0)
    c = torch.mm(b, w2)

    loss = torch.sum((y-c) ** 2)

    grad2 = -2 * torch.mm(b.t(), y-c)
    rela = torch.tensor([[i if i>0 else 0 for i in row] for row in a], device=device)
    grad1 = -2 * x.t().mm((y-c).mm(w2.t()) * rela)

    w1 -= grad1 * learn_rate
    w2 -= grad2 * learn_rate

print(y)
print(c)
