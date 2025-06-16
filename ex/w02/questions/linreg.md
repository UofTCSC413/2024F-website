<!-- Zhang CSC311, Handout 2 -->
(a) In the linear regression model with one feature, we have the following model/hypothesis:
$$y = f(x) = w x + b$$
​with parameters, $w$ and $b$, which we wish to find by minimizing the cost:
$$\mathcal{E}(w, b) = \frac{1}{2N}\sum_i ((w x^{(i)} + b) - t^{(i)})^2$$
What are the derivatives $\frac{\partial \mathcal{E}}{\partial w}$ and $\frac{\partial \mathcal{E}}{\partial b}$?

(b) In the linear regression model with many features, we have the following model/hypothesis:
$$y = f(x) = {\bf w}^\top {\bf x}+ b$$
with parameters, ${\bf w} = [w_1, w_2, \dots w_d]^T$ and $b$, which we wish to find by minimizing the cost:
$$\mathcal{E}({\bf w}, b) = \frac{1}{2N}\sum_i ((\fw^\top \fx^{(i)}+b) - t^{(i)})^2$$
What is the derivative $\frac{\partial \mathcal{E}}{\partial w_j}$ for a weight $w_j$?