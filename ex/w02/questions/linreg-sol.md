(a) We obtain the derivative with respect to $w$ directly using the chain rule resulting in
$$
\frac{\partial \mathcal{E}}{\partial w} 
  = \frac{1}{N}\sum_i x^{(i)}((w x^{(i)} + b) - t^{(i)}) 
$$
Similarly, the derivative with respect to $b$ is
$$
\frac{\partial \mathcal{E}}{\partial b} 
  = \frac{1}{N}\sum_i ((w x^{(i)} + b) - t^{(i)})
$$

(b) The derivative with respect to $w_j$ is
$$
\frac{\partial \mathcal{E}}{\partial w_j} 
  = \frac{1}{N}\sum_i x_j^{(i)}((\fw^\top \fx^{(i)}+b) - t^{(i)})
$$