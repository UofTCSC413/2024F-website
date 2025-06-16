(a) Output tensor size: $2\times 4\times 4$

(b) Output tensor size: $4\times 10\times 10$

(c) The first dimension is always the number of input channels, i.e. $c_{out}$. The output for the first value of the input tensor is of size $3\times 3$. For all remaining values, the stride determines the additional values of the ouput. Stride 2 means, that we need two values for each additional value. Thus, the output height $h_{out}$ and width are $w_{out}$:
$$
\begin{aligned}
h_{out} &= 3 + (h-1)\cdot 2 = 2h+1\\
w_{out} &= 3 + (w-1)\cdot 2 = 2w+1\\
\end{aligned}
$$
Thus, the resulting size is $2\times (2h+1) \times (2w+1)$.

(d) This generalization requires a similar consideration and we can basically reuse the equations from above:
$$
\begin{aligned}
h_{out} &= h_k + (h-1)\cdot s\\
w_{out} &= w_k + (w-1)\cdot s\\
\end{aligned}
$$
