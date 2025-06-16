Transformers use a scaled dot product attention mechanism given by
$$
C
 = \text{attention}(Q, K, V) 
 = \text{softmax}\left(\fr{QK^\top}{\sqrt{d}}\right) V,
$$
where $Q\in\R^{n_q\times d_k}$, $K\in\R^{n_k\times d_k}$, $V\in\R^{n_k\times d_v}$.

(a) Is the softmax function here applied row-wise or column-wise? What is the shape of the result?

(b) What is the value of $d$? Why is it needed?

(c) What is the computational complexity of this attention mechanism? How many additions and multiplications are required? Assume the canonical matrix multiplcation and not counting $\exp(x)$ towards computational cost.

(d) In the masked variant of the module, a masking matrix is added before the softmax function is applied. What are its values and its shape? For simplicity, assume $n_q=n_k$.