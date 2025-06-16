There are different ways to write the second order taylor series expansion a at a point $\fa$ for multivariate functions $f$. We will use the following form

$$T(x) = f(\fa) + \nabla f(\fa)^\top (\fx-\fa) + \frac{1}{2}(\fx-\fa)^\top \fH (\fx-\fa)$$

where $\nabla f(\fa)$ is the gradient of $f$ at $\fa$ and $\fH$ is the Hessian of $f$ at $\fa$. As a reminder, the Hessian is the matrix of second order partial derivatives. So, all we need to do for all of the exercises is evaluate $f(\fa)$ and compute its gradient and Hessian.

(a) $T(x) = 5 + 15 (x-1) + 15 (x-1)^2$
(b) First, we simplify $f(x,y)=x^2 (y^3+1)$ and compute $f(3,2)=81$ and then we compute the gradient and Hessian of $f$ resulting in
 $$
 \nabla f(x,y) = \bpmat 2x (y^3-1) \\ x^2(3y^2-1) \epmat,\quad
 \fH = \bpmat 2(y^3-1) & 6 xy^2 \\ 6xy^2 & 6x^2y \epmat.
 $$

 (c) This one is similar to (b) but we have to be careful with the logarithm. First, we have $f(\fx_0)=0$ because $\log(1)=0$. Then, we compute the gradient and Hessian of $f$ resulting in
    $$
    \nabla f(\fx) = \bpmat 
        3x_1^2 x_2 \log(x_2) \\ x_1^3 (1+\log(x_2)) 
        \epmat,\quad
    \fH = \bpmat 
        6x_1 x_2 \log(x_2)   & 3x_1^2 (1+\log(x_2)) \\
        3x_1^2 (1+\log(x_2)) & x_1^3 x_2^{-1} 
    \epmat
    $$

(d) Evaluating the trigonometric functions here is simpler than it seems because they are evaluated at $\pi$ and $-\pi$. We get $f(\fx_0) = \sin(-\pi) + \cos(\pi) = 0 - 1 = -1$ and
    $$
    \nabla f(\fx) = \bpmat \cos(x_1) \\ -\sin(x_2) \epmat, \quad
    \fH = \bpmat -\sin(x_1) & 0 \\ 0 & -\cos(x_2) \epmat.
    $$
