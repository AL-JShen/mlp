import numpy as np

batch = 1
dim_in = 64
dim_hid = 16
dim_out = 4
learn_rate = 1e-5

x = np.random.randn(batch, dim_in)
y = np.random.randn(batch, dim_out)

w1 = np.random.randn(dim_in, dim_hid)
w2 = np.random.randn(dim_hid, dim_out)

a = np.dot(x, w1)
b = np.maximum(0, a)
c = np.dot(b, w2)

print(c)

for i in range(1000):
    a = np.dot(x, w1)
    b = np.maximum(0, a)
    c = np.dot(b, w2)

    loss = np.sum((y-c) ** 2)

    grad2 = -2 * np.dot(b.T, y-c)
    rela = np.array([[i if i>0 else 0 for i in row] for row in a])
    grad1 = -2 * x.T.dot((y-c).dot(w2.T) * rela)

    w1 -= grad1 * learn_rate
    w2 -= grad2 * learn_rate

print(y)
print(c)
