For the purpose of batch normalization, we can consider each output neuron individually. Thus, we will simplify our notation and write $a_i$, $y_i$, ... instead of $a_i^{(j)}$, $y_i^{(j)}$, ... respectively.

(a) The forward pass is given by the following equations
$$
\begin{aligned}
\mu_B 
  &:= \frac{1}{M}\sum_{i=1}^m a_i, 
  &\text{(mini-batch mean)}  \\
\sigma_B^2 
  &:=\frac{1}{M}\sum_{i=1}^M (a_i-\mu_B)^2,
  &\text{(mini-batch variance)}\\
\ha_i 
  &:= \frac{a_i-\mu_B}{\sqrt{\si_B^2+\epsilon}},   
&\text{(normalize)}\\
  y_i
  &:= BN_{\gamma, \beta}((a_i)_{i=1}^M)
  := \gamma\ha_i + \beta.
  &\text{(scale and shift)}
\end{aligned}
$$
The entire layer is defined as 
$$
BN(\fa_1, \ldots \fa_M)=\big(
  BN_{\gm^{(1)}, \be^{(1)}}\big((a_i^{(1)})_{i=1}^M\big),
  \ldots, 
  BN_{\gm^{(N)}, \be^{(N)}}\big((a_i^{(N)})_{i=1}^M\big)
  \big)
$$
where $\gm^{(1)}, \ldots, \gm^{(N)}$ and $\be^{(1)}, \ldots, \be^{(N)}$ are learnable parameters.


(b) The derivatives can be expressed using the chain rule where we obtain $\mu_B$, $\si_B$, $\ha_i$, and $y_i$ during the forward pass while $\partial \mcL/\partial y_i$ is obtained from earlier steps of the backward pass. The remaining derivatives are:
$$
\begin{aligned}
\frac{\partial\mcL}{\partial \ha_i} &= \fr{\partial \mcL}{y_i}\cdot \gm, \\ 
\frac{\partial\mcL}{\partial \si_B^2}
  &= 
  \sum_{i=1}^M \fr{\partial\mcL}{\partial\ha_i}\cdot(a_i-\mu_B)\cdot
  \frac{-1}{2}(\si_B^2+\epsilon)^{-3/2}, \\ 
\fr{\partial\mcL}{\partial\mu_B} 
  &=
  \bigg(\sum_{i=1}^M \fr{\partial\mcL}{\partial\ha_i}
  \cdot\frac{-1}{\sqrt{\sigma_B^2+\epsilon}}\bigg) 
  +\fr{\partial\mcL}{ \partial \si_B^2}
  \cdot\frac{\sum_{i=1}^M  -2(a_i-\mu_B)}{M},\\
\fr{\partial\mcL}{\partial a_i}
  &=
  \fr{\partial\mcL}{\partial\ha_i}
  \cdot \frac{1}{\sqrt{\sigma_B^2+\epsilon}}  + \fr{\partial \mcL}{\partial \sigma_B^2} \cdot \frac{2(a_i-\mu_B)}{M} + \fr{\partial \mcL}{\partial \mu_B}\cdot \frac{1}{M},\\
\fr{\partial\mcL}{\partial\gamma}&= \sum_{i=1}^M \fr{\partial\mcL}{\partial y_i} \cdot \ha_i, \\ 
\fr{\partial\mcL}{\partial\beta} &= \sum_{i=1}^M \fr{\partial\mcL}{\partial y_i}.
\end{aligned}
$$
Here $\epsilon$ is a small constant which is added in practice to the variance to avoid division by zero. It is actually not part of the derivative.