First, remember that the normalized eigenvectors of a symmetric matrix are orthogonal. Thus, we have
$$
\fe_i^\top \fe_j = \begin{cases} 1 & i=j \\ 0 & i\neq j \end{cases}.
$$

Second, for symmetric $\fA$, its spectral decomposition is given by $\fA = \fQ \fLa \fQ^\top$, where $\fQ$ is a matrix where each column is an (orthogonal) eigenvector of unit length.

(a) Here, the eigenvectors are already normalized and orthogonal, so we can simply write $\fQ = (\fe_1, \fe_2)$ and $\fLa = \diag(\lambda_1, \lambda_2)$. Then, we have
  $$
  \fA = \bpmat 2 & 0 \\
    0 & 3
    \epmat
  $$

(b) Here, we have to normalize the eigenvectors first. Each has length $\sqrt{2}$, so we have to divide each of them by $\sqrt{2}$, i.e. we set $\fte_i:=\fe_i/\sqrt{2}$. With this, we can construct an orthogonal matrix of eigenvalues as $\fQ = (\fte_1, \fte_2)$. The resulting matrix $\fA$ is
  $$
  \fA = \bpmat 
    2.5 & -0.5 \\
    -0.5 & 2.5
    \epmat
  $$