(a) For better readability we will write $(g\circ h)(x)$ for $g(h(x))$. By applying the chain rule, we obtain
$$
\begin{aligned}
F'(x) 
 &= (f_3 \circ f_2 \circ f_1)' (x) \\
 &= (f_2 \circ f_1)'(x) 
    \cdot (f_3' \circ f_2 \circ f_1)(x)\\
&= f_1'(x) \cdot (f_2' \circ f_1)(x) 
    \cdot (f_3' \circ f_2 \circ f_1)(x)
\end{aligned}
$$

(b) First, let's start with the forward mode
    ```
d = f1'(x)
v = f1(x)
d = f2'(x) * d
v = f2(v)
d = f3'(v) * d
```

    Now, for the reverse mode, we first do a "forward pass" before computing gradients:
    ```
    v1 = f1(x)
    v2 = f2(v1)
    d = f3'(v2)
    d = d*f2'(v1)
    d = d*f1'(x)
    ```

(c) Simply evaluating the expression in (a) is not in line with any of the modes. It also involves repeated computations because $f_1(x)$ will be computed twice. 
    Now, if we allow for caching of ingtermediate results, this doubling of compugtaiton disappears. The order written above will then be in line with forward move automatic differentiation. However, this is specific to our example and in general not true.