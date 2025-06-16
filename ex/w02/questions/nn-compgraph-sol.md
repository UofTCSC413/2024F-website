(a) Applying the chain rule, we have

    $$ \frac{\partial \mathcal{L}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial y} \frac{\partial y}{\partial z} \frac{\partial z}{\partial w_j} $$

    Looking at each term individually yields
    $$
    \begin{aligned}
    \frac{\partial \mathcal{L}}{\partial y} 
      &= \frac{\partial}{\partial y} [-t \log(y) - (1 - t) \log(1 - y)] 
      = - \frac{t}{y} + \frac{1 - t}{1 - y}\\
    \frac{\partial y}{\partial z} 
      &= \frac{\partial \sigma(z)}{\partial z} 
      = \sigma(z) (1 - \sigma(z))
      = y (1 - y)\\
    \frac{\partial z}{\partial w_j} 
      &= \frac{\partial}{\partial w_j} (w^\top x) = x_j
    \end{aligned}
    $$

    Bringing it all together yields:
    $$
    \begin{aligned}
    \frac{\partial \mathcal{L}}{\partial w_j} 
      &= \left( - \frac{t}{y} + \frac{1 - t}{1 - y} \right) \cdot y (1 - y) \cdot x_j \\
      &= (-t + ty + 1 - t - y + ty) x_j \\
      &= (y - t) x_j
    \end{aligned}
    $$

(b) The computation graph is given in the figure below.

![Computation graph for exercise 4 (b)](nn-compgraph-drawing.eps){ width=50% }