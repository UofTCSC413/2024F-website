Consider the function $F(x) = f_3(f_2(f_1(x))$ and assume you  also know the derivatives $f_i'$ for all $f_i$. 

(a) Apply the chain rule to express $F'(x)$ in terms of $f_i'$s and $f_i$.

(b) Write down the pseudocode for computing $F'(x)$ using the forward mode and the reverse mode respectively. Assume all functions to be scalar functions of a scalar variable, i.e. $f_i: \R \rightarrow \R$.

(c) If you simply ask your interpreter / compiler to evaluate the expression in (a), will the computation be in forward mode, reverse mode, or neither of the modes? Why? You can assume that your interpreter / compiler does not do any caching or optimization and simply evaluates the expression from left to right. Does anything change if you assume that your interpreter caches results that have been computed before?

