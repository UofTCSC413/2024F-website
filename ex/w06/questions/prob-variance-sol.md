First, we use the definition of variance and rewrite the left hand side as
$$
\Var(aX+bY) = \E\li[ (aX+bY)^2 \ri] - \E[aX+bY]^2.
$$
Next, we expand the squares for each of the terms on the right hand side:
\begin{align*}
\E\li[ (aX+bY)^2 \ri] 
  &= \E\li[ a^2X^2 + 2abXY + b^2Y^2 \ri] \\
  &= a^2\E\li[ X^2 \ri] + 2ab\E[ XY ] + b^2\E\li[ Y^2 \ri] \\
  &= a^2\E\li[ X^2 \ri] + 2ab\E[ X]\,\E[Y] + b^2\E\li[ Y^2 \ri] ,\\
\E[aX+bY]^2
  &= \li(a\E[X] + b\E[Y]\ri)^2 \\
  &= a^2\E[X]^2 + 2ab\E[X]\,\E[Y] + b^2\E[Y]^2.
\end{align*}
Subtracting the two terms, we get
\begin{align*}
& \E\li[ (aX+bY)^2 \ri] - \E[aX+bY]^2 \\
& \quad = a^2\E\li[ X^2 \ri] + 2ab\E[ X]\,\E[Y] + b^2\E\li[ Y^2 \ri]
    - a^2\E[X]^2 - 2ab\E[X]\,\E[Y] - b^2\E[Y]^2 \\
& \quad = a^2\E\li[ X^2 \ri] - a^2\E[X]^2 
   + b^2\E\li[ Y^2 \ri] - b^2\E[Y]^2 \\
& \quad = a^2(\E\li[ X^2 \ri] - \E[X]^2)
   + b^2\li(\E\li[ Y^2 \ri] - \E[Y]^2\ri) \\
& \quad = a^2\cdot\Var(X) + b^2\cdot\Var(Y).
\end{align*}