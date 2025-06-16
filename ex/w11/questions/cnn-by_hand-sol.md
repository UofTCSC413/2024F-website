(a) Here, we simply apply the convolutional kernel over each $2\times 2$ patch of the input. There are 9 such patches. The output $Y$ is then

$$
Y = \bpmat
1 &  3 &  -1 \\
4 &  2 &  2 \\
5 &  3 & 3
\epmat
$$

(b) Same idea except that we skip every other patch resulting in only 4 patches. The output $Y$ is then

$$
Y = \bpmat 
 1 & -1 \\
 5 & 3
\epmat
$$

(c) Now, we have added zeros on each side of the input. The resulting $6\times 6$ padded input $X_\mathrm{padded}$ and corresponding output $Y$ are

$$
X_\mathrm{padded} = 
\bpmat
0 & 0 &  0 & 0 &  0 & 0 \\
0 & 1 &  0 & 1 & -1 & 0 \\
0 & 1 &  0 & 1 &  0 & 0 \\
0 & 0 &  3 & 0 &  1 & 0 \\
0 & 1 & -1 & 0 &  1 & 0 \\
0 & 0 &  0 & 0 &  0 & 0
\epmat, 
\qquad
Y = \bpmat 
 1 & 1 & 0 \\
 2 & 2 & 0 \\
 2 & -1 & 1
\epmat
$$