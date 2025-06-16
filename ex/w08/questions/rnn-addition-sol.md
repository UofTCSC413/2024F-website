(a) Since the inputs $\bf{x}^{(t)}$ are $2 \times 1$ and the hidden units
$\bf{h}^{(t)}$ are $2 \times 1$, we should have:

    - $\bf{W}$ is $3 \times 3$
    - $\bf{U}$ is $3 \times 2$
    - $\bf{b}_h$ is $3 \times 1$
    - $\bf{v}$ is $3 \times 1$

(b) We will follow the hint and implement the addition in our RNN such that:

    1. The first of our hidden units $h_1^{(t)}$ is 1 if and only if the sum $S^{(t)} \doteq x_1^{(t)} + x_2^{(t)} + c^{(t-1)} \geq 1$, where by $c^{(t-1)}$ we denote a carry ($\bf{h_2}^{(t-1)}$ from the previous addition). Note, these $S^{(t)}$ and $c^{(t-1)}$ are not variables of the model, merely our notation to help us to work out the solution.
    2. The $h_2^{(t)}$ is 1 iff the sum $S^{(t)} \geq 2$,
    3. and $h_3^{(t)}$ is 1 iff the sum $S^{(t)}$ is 3.

    Notice that the carry $c^{(t-1)}$ is going to be 1 iff $h_2^{(t-1)}=1$ and 0 otherwise\footnote{We need to initialize $\mathbf{h}^{(0)}=\mathbf{0}$.}, i.e. when the previous addition was 2 or 3. Therefore to compute $h_i^{(t)}$ we need to first compute the sum $S^{(t)} = x_1^{(t)} + x_2^{(t)} + h_2^{(t-1)}$ and then offset it by $-i+1$ so that after applying the hard threshold function we get the desired value as specified above. This can be achieved with the following set of parameters:
    $$
    \mathbf{U}= \begin{bmatrix}
		1 & 1 \\
        1 & 1 \\
        1 & 1 \end{bmatrix},\quad
    \mathbf{W}=\begin{bmatrix}
		0 & 1 & 0 \\
        0 & 1 & 0 \\
        0 & 1 & 0\end{bmatrix},\quad
    \mathbf{b_h}= \begin{bmatrix}
		-0.5 \\
        -1.5 \\
        -2.5 \end{bmatrix}. 
    $$

(c) To compute the output $y^{(t)}$ we need to check if the $S^{(t)}$ is 1 or 3, that is, if either $h_1^{(t)} = 1$ while all other hidden units are zero or all hidden units are 1. We can accomplish this by setting: $\mathbf{v}=\begin{bmatrix} 1, -1, 1 \end{bmatrix}$ and $b_y = -0.5$.