(a) The resulting output tensor is of shape $6 \times 6$ and is given by:
$$
Y = \bpmat
 1 &  0 &  0 &  0 &  2 &  0\\
 1 &  2 &  0 &  0 &  2 &  4\\
 2 &  0 &  3 &  0 &  0 &  0\\
 2 &  4 &  3 &  6 &  0 &  0\\
-1 &  0 &  0 &  0 &  3 &  0\\
-1 & -2 &  0 &  0 &  3 &  6 
\epmat
$$

(b) The resulting output tensor is of shape $4 \times 4$ and is given by:
$$
Y = \bpmat
 1 &  0 &  2 &  0 \\
 3 &  5 &  2 &  4 \\
 1 &  7 &  9 &  0 \\
-1 & -2 &  3 &  6
\epmat
$$

(c) We can verify our answers with PyTorch with the following commands:
    ```python
    nn.functional.conv_transpose2d(X.unsqueeze(0), K, stride=2)

    nn.functional.conv_transpose2d(X.unsqueeze(0), K)
    ```
    The call to `.unsqueeze(0)` can be left out if `X` has a batch dimension and is already a 4D tensor.

(d) A potential implementaiton (that isn't optimized in any way) of transposed convolutions looks like this:

```python

def conv_transpose2d(input, weight, stride=1):
  # input - input of shape (C_in, H, W)
  # weight - kernel of shape (C_in, C_out, K, K)
  # stride - stride of the transposed convolutio
  # RETURNS
  # output - output of shape (C_out, H_out, W_out)
  (c_in, h_in, w_in) = X.size()
  (c2_in, c_out, k, k2) = K.size()

  assert c_in == c2_in, "Number of input channels must match"
  assert k == k2, "Kernel must be square"

  h_out = (h_in - 1) * stride + k
  w_out = (w_in - 1) * stride + k
  output = torch.zeros((c_out, h_out, w_out))

  for c_cur_in in range(c_in):
    for c_cur_out in range(c_out):
      for h in range(0, h_in):
        for w in range(0, w_in):
          output[c_cur_out, h*stride:h*stride+k, w*stride:w*stride+k] \
              += weight[c_cur_in, c_cur_out,:,:] * input[c_cur_in,h,w]

  return output
```