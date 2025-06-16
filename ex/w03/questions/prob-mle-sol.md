First, let's quickly remember that the maximum likelihood estimator (MLE) of a probability distribution from dataapoints $\fx_1, \ldots, \fx_N$ is given by
$$
\hte_{\mathrm{MLE}} = \argmax_{\te \in \Te} \prod_{i=1}^N f(\fx_i | \te),
$$
where $f$ is the probability density function of the considered probability distribution family, $\te$ are the parameters of the distribution, and $\Te$ is the parameter space (a set containing all possible parameters). The product on the right hand side is also known as the likelihood function. In practice, we usually work with the log-likelihood function instead. Because the logarithm is monotonously increasing, the resulting estimators are the same, i.e.
$$
\hte_{\mathrm{MLE}} 
 = \argmax_{\te \in \Te} \prod_{i=1}^N f(\fx_i | \te),
 = \argmax_{\te \in \Te} \sum_{i=1}^N \log \li( f(\fx_i | \te)\ri).
$$

Second, let's remember that the probability density function of a multivariate Gaussian distribution is given by
$$
f(\fx_i | \mu, \Si) 
 = \frac{1}{(2 \pi)^{n/2} |\Si|^{1/2}} 
   \exp \li( 
     - \frac{1}{2} 
     (\fx_i - \mu)^\top
     \Si^{-1} 
     (\fx_i - \mu) 
   \ri) 
$$
with parameters $\te=(\mu, \Si)$, where $\mu\in\R^n$ is the mean vector and $\Si\in\R^{n\times n}$ is the covariance matrix. Moreover, the notation $|\Si|$ denotes the determinant of $\Si$.

Our goal is to maximize the log-likelihood function which in this case is
$$\begin{aligned}
l(\mu, \Si| \fx_1, \ldots, \fx_N) 
  & := \sum_{i=1}^N \log f(\fx_i | \mu, \Si) \\
  & = \sum_{i=1}^N \li( 
    - \frac{n}{2} \log (2 \pi) 
    - \frac{1}{2} \log |\Si|  
    - \frac{1}{2}  (\fx_i - \mu)^\top \Si^{-1} (\fx_i - \mu) 
  \ri).
\end{aligned}$$
For obtaining the MLE of $\mu$, we can simply take the gradient of $l$ with respect to $\mu$ and set it to zero resulting in:
$$
\nabla_\mu l(\mu, \Si| \fx_1, \ldots, \fx_N) 
 = \sum_{i=1}^N  \Si^{-1} ( \fx_i - \mu )
 = 0 
$$
Since $\Si$ is a covariance matrix, it is positive definite. Thus, we can multiply both sides of the equation by $\Si$ and obtain
$$
\begin{aligned}
0 & = N \mu - \sum_{i=1}^N  \fx_i,
\\
\Rightarrow \hmu_{\mathrm{MLE}} &=  \frac{1}{N} \sum_{i=1}^N \fx_i .
\end{aligned}
$$
Conveniently, $\Si$ disappears and thus we do not have to worry about it.