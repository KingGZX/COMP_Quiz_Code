import torch
import torch.nn as nn
import math
import numpy as np

a = torch.tensor([[2, 0], [1, 2]])
b = a.view((4, 1))
print(b)

embed = torch.tensor([[2, 3, 4, 1],
                      [0, 1, 2, 2],
                      [2, 2, 2, 3],
                      [1, 1, 1, 1]], dtype=torch.float32)

Wq = torch.tensor([[1, 1, 0, 1],
                   [0, -1, 0, 0],
                   [1, 1, -1, 0]], dtype=torch.float32, requires_grad=True)

Wk = torch.tensor([[1, 1, -1, 0],
                   [0, 1, 0, 1],
                   [0, 1, 1, 0]], dtype=torch.float32, requires_grad=True)

Wv = torch.tensor([[1, -1, 1, 1],
                   [0, 0, 0, 1],
                   [0, 1, 0, 0]], dtype=torch.float32, requires_grad=True)

Q = torch.matmul(Wq, embed)
K = torch.matmul(Wk, embed)
V = torch.matmul(Wv, embed)

print("Query:\n{}\nKey:\n{}\nValue:\n{}\n".format(Q, K, V))

sm = nn.Softmax(dim=0)
attn = sm(torch.matmul(K.T, Q) / math.sqrt(3))
print(attn)

out = torch.matmul(V, attn)
print(out)

print(torch.matmul(torch.matmul(2 * out, attn.T), embed.T))

loss = torch.sum(torch.square(out))
loss.backward()
print(Wv.grad)


