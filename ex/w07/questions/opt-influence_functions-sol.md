First, let's recall the definitions of $\te$ and  $\hte(\epsilon)$:
\begin{align*}
\hat{\theta} 
 &= \text{argmin}_{\theta} \frac{1}{N} \left[ \sum_{i=1}^{N} L(x_i, y_i; \theta) \right] \\
\hte({\epsilon}) 
  &= \text{argmin}_{\theta} \frac{1}{N} \left[ \sum_{i=1}^{N} L(x_i, y_i; \theta) \right] + \epsilon L(x,y; \theta) 
\end{align*}
The first order taylor series expansion around $\epsilon = 0$ is given by
$$
\hte(0) + \epsilon \fr{d\hte(\epsilon)}{d\epsilon} {\Bigr |}_{\epsilon=0} 
$$
From the definitions above, it can directly be seen that $\hte(0) = \hte$ which completes the proof.
