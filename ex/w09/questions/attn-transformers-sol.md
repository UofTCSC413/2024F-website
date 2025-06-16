 (a) The softmax function is applied row-wise and the shape of the result is $n_q\times n_k$. One way to see this is by looking at the shape of the dot product $QK^\top$ which is $n_q\times n_k$. Each row represents the pre-softmax scores of all keys and a given query. Because we need to normalize our attention weights per query, the normalization happens along the rows.

 (b) The value of $d$ is $d_k$. It is needed to scale the dot product so that the gradient of the softmax function does not vanish.

 (c) To obtain the computational complexity, let's look at all the operations individually:
     * $QK^\top$ requires $n_q n_k d_k$ multiplications and $n_qn_k(d_k-1)$ additions.
     * Dividing by $\sqrt{d_k}$ needs to be carried out $n_q n_k$ times.
     * Applying the softmax function can be implemented in $n_q n_k$ divisions and $n_q(n_k-1)$ additions.
     * The final matrix multiplication requires $n_qd_vn_k$ multiplications and $n_q d_v (n_k-1)$ additions.

(d) The masking matrix is a triangular matrix with $-\infty$ on its top right half. This results in softmax weights being $0$ for all key-query combinations to which $-\infty$ is added. 