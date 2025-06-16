Consider the following $4\times 4 \times 1$ input X and a $2\times 2 \times 1$ convolutional kernel K with no bias term 

$$
X = 
\bpmat
1 & 2 & -1 & 1 \\
1 & 0 & 1 & 0 \\
0 & 1 & 0 & 2 \\
2 & 1 & 0 & -1
\epmat, \qquad
% 
K = \bpmat
1, & 0 \\
2, & 1 \\
\epmat
$$

(a) What is the output of the convolutional layer for the case of stride 1 and no padding? 

(b) What if we have stride 2 and no padding?

(c) What if we have stride 2 and zero-padding of size 1?