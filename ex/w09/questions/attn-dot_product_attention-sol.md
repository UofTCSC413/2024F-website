First we compute the dot products between $\fs$ and the $\fh_i$ and apply softmax resulting in:
$$
\fa = \text{Softmax}\li(
\bpmat
1 & 1 & 0\\
2 & 2 & 1\\
3 & 1 & -1
\epmat^\top
\cdot
\bpmat 1\\ 2\\ 1\epmat
\ri) 
\approx
\bpmat 0.88\\ 0.12\\ 0.00\epmat
$$
The resulting context vector is then computed as a weighted sum of the $\fh_i$:
$$
\fc 
  = a_1 \fh_1 + a_2 \fh_2 + a_3 \fh_3 
  \approx \bpmat 1.00\\ 2.00\\ 2.76\epmat
$$

A simple implementation yielding the solution is as follows:
```python
import torch
h = torch.tensor([[1, 2, 3], [1,2,1], [0,1,-1]])
s = torch.tensor([1,2,1])
a = torch.matmul(h,s).float()
a = torch.exp(a)/torch.sum(torch.exp(a)) # Softmax
c = a[0] * h[0] + a[1] * h[1] + a[2] * h[2]
print(c)
```