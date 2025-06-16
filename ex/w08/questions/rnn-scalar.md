<!-- from Week8 Practice exercises -->
Suppose we have the following vanilla RNN network, where the inputs and hidden units are scalars.
\begin{align*}
h^{(t)} &= \tanh\li(w \cdot h^{(t-1)} + u \cdot x^{(t-1)} + b_h\ri) \\
y &= \sigma\li(v \cdot h^{(T)} + b_y\ri)
\end{align*}

(a) Show that if $|w| < 1$, and the number of time steps $T$ is large, then the gradient $\frac{\partial y}{\partial x^{(0)}}$ vanishes.

(b) Why is the result from Part (a) troubling?