You are given an input matrix $X$ (consisting of a single channel) and a kernel $K$ as follows:
$$
X = \bpmat
 1 & 0 & 2 \\
 2 & 3 & 0 \\
-1 & 0 & 3
\epmat, \quad
K = \bpmat
 1 & 0 \\
 1 & 2
\epmat
$$

(a) Compute the transposed convolution by hand assuming stride 2.

(a) Compute the transposed convolution by hand assuming stride 1.

(b) Use PyTorch to verify your answers.

(c) Implement the transposed convolution in PyTorch without using its own implementaiton. You can assume no bias term a square kernel and no separate batch dimension. I.e. your task is to implement the following function 

```python
def conv_transpose2d(inp, weight, stride=1):
    # inp - input of shape (C_in, H, W)
    # weight - kernel of shape (C_in, C_out, K, K)
    # stride - stride of the transposed convolution
    # RETURNS
    # output - output of shape (C_out, H_out, W_out)
    #
    # YOUR CODE HERE
    return output
```