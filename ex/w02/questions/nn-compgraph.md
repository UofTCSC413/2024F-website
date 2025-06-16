<!-- Zhang CSC311, Handout 5 -->
(a) Compute the $\frac{\partial \mathcal{L}}{\partial w_j}$  gradient of $\mathcal{L}$ with respect to a $w_j$ in the following computation:
  \begin{align*} 
  \mathcal{L}(y, t) &= - t \log(y) - (1-t) \log(1-y) ,
    & y &= \sigma(z) ,
    &  z &= {\bf w}^\top {\bf x} .
  \end{align*}

(b) Draw the computation graph for the following neural network,
showing the relevant scalar quantities.
Assume that $\fy, \fh, \fx \in \mathbb{R}^2$
  \begin{align*}
  \mathcal{L} &= \frac{1}{2}\sum_k (y_k - t_k)^2 , 
    & y_k &= \sum_i w_{ki}^{(2)} h_i + b_k^{(2)} , 
    & h_i &= \sigma(z_i) ,
    & z_i &= \sum_j w_{ij}^{(1)} x_j + b_i^{(1)} .
  \end{align*}