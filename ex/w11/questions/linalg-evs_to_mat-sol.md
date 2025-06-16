First, remember that the normalized eigenvectors of a symmetric matrix are orthogonal. Thus, we have
$$
\fe_i^\top \fe_j = \begin{cases} 1 & i=j \\ 0 & i\neq j \end{cases}.
$$

Second, for symmetric $\fA$, its spectral decomposition is given by $\fA = \fQ \fLa \fQ^\top$, where $\fQ$ is a matrix where each column is an (orthogonal) eigenvector of unit length.

In our case, the eigenvectors are already normalized and orthogonal, so we can simply write $\fQ = (\fe_1, \fe_2)$ and $\fLa = \diag(\lambda_1, \lambda_2)$. Then, we have
  $$
  \fA = \bpmat 1.5 & -0.5 \\
    -0.5 & 1.5
    \epmat
  $$