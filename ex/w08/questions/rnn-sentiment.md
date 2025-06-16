<!-- Based on Week8 Practice exercises -->
Suppose we are training a vanilla RNN like below to determine
whether a sentence expresses positive or negative sentiment.
This RNN will be a character-level RNN where $x^{(1)}, \ldots, x^{(T)}$ is the sequence of input characters. The RNN is given as follows:
\begin{align*}
h^{(t)} &= \tanh\li(U x^{(t)} + W h^{(t-1)} + b\ri) \\
y &= \sigma\li(V h^{(T)} + d\ri)
\end{align*}

(a) How many times do we need to apply the weight matrix $U$, $W$, and $V$?


(b) What are the shapes of the matrices $U$, $W$, and $V$? 

(c) How many addition and multiplication operations are required to make a prediction? 
    You can assume that no addition and multiplications are performed when applying the tanh and sigmoid activation functions.