The resulting context matrix is given by:
$$
C\approx
\bpmat
1.80 & 1.00 & 1.44\\
1.26 & 1.25 & 0.26
\epmat
$$
A simple implementation would look as follows:
```python
import torch
Q = torch.tensor([[1, 2], [3, 1]]).float()
K = torch.tensor([[2, 1], [1, 1], [0, 1]]).float()
V = torch.tensor([[1, 2, -2], [1, 1, 2], [0, 1, -1]]).float()
d_k = torch.tensor(K.shape[1])
M = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(d_k)
S = torch.exp(M) / torch.sum(torch.exp(M), dim=1).view(-1,1)
torch.matmul(S, V)
```
Pytorch also provides a function for scaled dot product attention:
```python
import torch.nn.functional as F
F.scaled_dot_product_attention(Q, K, V)
```