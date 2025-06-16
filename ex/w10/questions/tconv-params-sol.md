The number of parameters is the product of input channels, output channels and kernel size. For each output channel, there is one bias parameter (if `bias=True` which is default in PyTorch). The stride does not matter for the number of paramters.

(a) $3\cdot 2 \cdot 9 + 2=56$
(b) $3\cdot 10 \cdot 9 + 10=280$
(c) $3\cdot 2 \cdot 16 + 2=98$
(d) $3\cdot 4 \cdot 9 + 4=112$