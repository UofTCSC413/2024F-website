Let $D=\lbrace (x_i,y_i) | i=1 \ldots n\rbrace$ be a dataset obtained from the true underlying data distribution $P$, i.e. $D\sim P^n$. And let $h_D(\cdot)$ be a classifier trained on $D$. Show the variance bias decomposition
$$
\underbrace{\mathbb{E}_{D,x,y} \li[ (h_D(x) - y)^2 \ri]}_{\text{Expected test error}}  
  = \underbrace{\mathbb{E}_{D,x} \li[ (h_D(x) - \hat{h}(x))^2 \ri]}_{\text{Variance}} + 
  \underbrace{\mathbb{E}_{x,y} \li[ (\hat{y}(x) - y)^2 \ri]}_{\text{Noise}}  +  \underbrace{\mathbb{E}_{x} \li[ (\hat{h}(x) - \hat{y}(x))^2 \ri]}_{\text{Bias}^2} 
$$
where $\hat{h}(x) = \mathbb{E}_{D \sim P^n}[h_D(x)]$ is the expected regressor over possible training sets, given the learning algorithm $\mathcal{A}$ and $\hat{y}(x) = \mathbb{E}_{y|x}[y]$ is the expected label given $x$. As mentioned in the lecture, labels might not be deterministic given x. To carry out the proof, proceed in the following steps:

(a) Show that the following identity holds
    \begin{align}
    \E_{D,x,y}\li[\li[h_{D}(x) - y\ri]^{2}\ri] 
    = \E_{D, x}\li[(\hh_{D}(x) - \hh(x))^{2}\ri] + \E_{x, y} \li[\li(\hh(x) - y\ri)^{2}\ri].
   \end{align}
(b) Next, show
    \begin{align}
	E_{x, y} \li[ \li(\hh(x) - y \ri)^{2}\ri] 
    =E_{x, y} \li[\li(\hy(x) - y\ri)^{2}\ri] + E_{x} \li[\li(\hh(x) - \hy(x)\ri)^{2}\ri]
    \end{align}
  which completes the proof by substituting (2) into (1).