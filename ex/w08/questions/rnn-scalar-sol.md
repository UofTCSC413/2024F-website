
(a) To make the sequence length $T$ explicit in the notation, we will write $y$ instead of $y_T$. Formally, what we have to show is 
    $$
    |w|<1 \implies \lim_{T\to\infty} \fr{\partial y_T}{\partial x^{(0)}} = 0 .
    $$
    For the proof, we expand the derivative of $y_T$ with respect to $x^{(0)}$ using the chain rule:
    $$
    \begin{aligned}
    \fr{\partial y_T}{\partial x^{(0)}} 
      & = \si'\li(v \cdot h^{(T)} + b_y\ri) 
        \cdot v \cdot \fr{\partial h^{(T)}}{\partial x^{(0)}} \\
      & = 
      \si'\li(v \cdot h^{(T)} + b_y\ri) 
        \cdot v 
        \cdot 
          \underbrace{
            \tanh'\li(w \cdot h^{(T-1)} + u \cdot x^{(T-1)} + b_h\ri)
          }_{A_{T-1}(x^{(0)})}
        \cdot w \cdot \fr{\partial h^{(T-1)}}{\partial x^{(0)}} \\
     & = \ldots \\
     & = \si'\li(v \cdot h^{(T)} + b_y\ri) 
        \cdot v 
        \cdot \prod_{t=2}^{T-1} A_{t}(x^{(0)})
        \cdot w^{T-1} \cdot \fr{\partial h^{(1)}}{\partial x^{(0)}} .\\
    \end{aligned}
    $$
    Using this, we can analyze the absolute value of the derivative $\partial y_T/\partial x^{(0)}$. For $\tanh$ and $\si$, the absolute value of their respective derivatives is bounded by $1$. Thus, we have
    $$
    \begin{aligned}
    \li|\fr{\partial y_T}{\partial x^{(0)}} \ri|
    & = 
    \underbrace{
      \li|\si'\li(v \cdot h^{(T)} + b_y\ri) \ri|
    }_{\leq 1}
    \cdot |v| 
    \cdot \prod_{t=2}^{T-1} 
      \underbrace{\li| A_{t}(x^{(0)}) \ri|}_{\leq 1}
    \cdot \li|w^{T-1}\ri| 
    \cdot \li|\fr{\partial h^{(1)}}{\partial x^{(0)}}\ri| \\
    & \leq |v| 
    \cdot \li|w^{T-1}\ri| 
    \cdot \li|\fr{\partial h^{(1)}}{\partial x^{(0)}}\ri| \\
    \end{aligned}
    $$
    Because $|w|<1$, this converges to $0$ as $T\to\infty$ and thus $|\partial y_T/\partial x^{(0)}|$ also converges to $0$, i.e. the gradient vanishes.

(b) It implies that in the considered setting, the input has no impact on the output.