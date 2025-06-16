First, let's quickly remember that the maximum likelihood estimator (MLE) of a probability distribution from dataapoints $\fx_1, \ldots, \fx_N$ is given by
$$
\hte_{\mathrm{MLE}} = \argmax_{\te \in \Te} \prod_{i=1}^N f(\fx_i | \te),
$$
where $f$ is the probability density function of the considered probability distribution family, $\te$ are the parameters of the distribution, and $\Te$ is the parameter space (a set containing all possible parameters).

As mentioned in our previous exercise, we usually work with the log-likelihood in practice. In this particular case, the log likelihood is given by
$$\begin{aligned}
l(\la | x_1, \ldots, x_N) 
  & := \sum_{i=1}^N \ln f(\fx_i | \la) \\
  & = \sum_{i=1}^N \ln\li(\la \exp(-\la x_i) \ri) \\
  & = \sum_{i=1}^N \ln(\la) + \ln\li( \exp(-\la x_i) \ri) \\
  & =  N \ln(\la) - \sum_{i=1}^N \la x_i .
\end{aligned}$$
The derivative with respect to $\la$ is
$$
 \fr{\partial l(\la | x_1, \ldots, x_N)}{\partial \la}
  = \fr{N}{\la} - \sum_{i=1}^N x_i .
$$
The MLE is obtained by setting it to 0 and solving for $\la$ as
$$
\hla_{MLE} = N \li( \sum_{i=1}^N x_i\ri)^{-1}.
$$
