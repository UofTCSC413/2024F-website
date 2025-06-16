You are an ML Engineer at Googlezon where you are working on an internal ML framework called TorchsorFlow. You are tasked with implementing a new layer known as BatchNormalization. The idea of this layer is as follows:

During training, consider the outputs of the previous layer $\fa_i=(a_i^{(1)}, \ldots, a_i^{(N)})$ for each element $i\in \{1, \ldots, M\}$ of the input batch. Compute the mean $\mu_j$ and variance $\si_j^2$ over each input dimension $j$. Use the resulting statistics to normalize the output of the previous layer. Finally, rescale the resulting vector with a learned constant $\gm$ and shift it by another learned constant $\be$.

(a) Write down the mathematical expression for the BatchNormalization layer. What are its learnable parameters?

(b) Compute the gradient of the loss $\mcL$ with respect to the input of the BatchNormalization $\fa_i$ layer.

(c) At test time, the batch size is usually 1. So, it is not meaningful (or even possible) to compute mean / variance. How would you implement a layer like this?