(a) The key idea here is to compute the gradient of the objective function and solve for $\fmu$. The gradient is obtained by applying the chain rule resulting in
$$0=\nabla_\fmu \sum_i \|\fx_i - \fmu\|^2 = -2 \sum_i (\fx_i - \fmu) .$$ 
Now, we solve this for $\fmu$ to obtain
$$\fmu = \frac{1}{N} \sum_i \fx_i .$$

(b) Here, we simply need to apply some algebraic manipulations to show that the two definitions are equivalent. We start with the first definition and expand the square:
\begin{align*}
\E[(X - \E[X])^2] 
  &= \E\li[X^2 - 2X\E[X]+\E[X]^2\ri] \\
  &= \E\li[X^2\ri] - \E[2X\E[X]]+\E\li[\E[X]^2\ri]\\
  &= \E\li[X^2\ri] - 2\E[X]\E[X]+\E[X]^2\\
  &= \E\li[X^2\ri] - \E[X]^2
\end{align*}