(a) First, we reformulate (1) as
 \begin{align*}
  \E_{D,x,y}\li[\li[h_{D}(x) - y\ri]^{2}\ri] 
    &= \E_{D,x,y}\li[\li[\li(h_{D}(x) - \hh(x)\ri) + \li(\hh(x) - y\ri)\ri]^{2}\ri] \nonumber \\
    &= \E_{x, D}\li[(\hh_{D}(x) - \hh(x))^{2}\ri] + 2 \mathrm{\;} \E_{x, y, D} \li[\li(h_{D}(x) - \hh(x)\ri)\li(\hh(x) - y\ri)\ri] + \E_{x, y} \li[\li(\hh(x) - y\ri)^{2}\ri]
\end{align*}
  Next, we note that the second term in the above equation is zero because
  \begin{align*}
	\E_{D,x, y} \li[\li(h_{D}(x) - \hh(x)\ri) \li(\hh(x) - y\ri)\ri] &= \E_{x, y} \li[\E_{D} \li[ h_{D}(x) - \hh(x)\ri] \li(\hh(x) - y\ri) \ri] \\
    &= \E_{x, y} \li[ \li( \E_{D} \li[ h_{D}(x) \ri] - \hh(x) \ri) \li(\hh(x) - y \ri)\ri] \\
    &= \E_{x, y} \li[ \li(\hh(x) - \hh(x) \ri) \li(\hh(x) - y \ri)\ri] \\
    &= \E_{x, y} \li[ 0 \ri] \\
    &= 0\ .
\end{align*}

(b) The proof here, is similar. We start by reformulating the second term in (2) as
  \begin{align*}
	\E_{x, y} \li[ \li(\hh(x) - y \ri)^{2}\ri] &= \E_{x, y} \li[ \li(\hh(x) -\bar y(x) )+(\bar y(x) - y \ri)^{2}\ri]  \\
  &=\E_{x, y} \li[\li(\hy(x) - y\ri)^{2}\ri] + \E_{x} \li[\li(\hh(x) - \hy(x)\ri)^{2}\ri] + 2 \mathrm{\;} \E_{x, y} \li[ \li(\hh(x) - \hy(x)\ri)\li(\hy(x) - y\ri)\ri]
  \end{align*}
  Here, the third term is zero which follows from an analogous derivation as in (a). Thus, we have
  \begin{align*}
	\E_{x, y} \li[\li(\hh(x) - \hy(x)\ri)\li(\hy(x) - y\ri)\ri] &= \E_{x}\li[\E_{y \mid x} \li[\hy(x) - y \ri] \li(\hh(x) - \hy(x) \ri) \ri] \\
    &= \E_{x} \li[ \E_{y \mid x} \li[ \hy(x) - y\ri] \li(\hh(x) - \hy(x)\ri)\ri] \\
    &= \E_{x} \li[ \li( \hy(x) - \E_{y \mid x} \li [ y \ri]\ri) \li(\hh(x) - \hy(x)\ri)\ri] \\
    &= \E_{x} \li[ \li( \hy(x) - \hy(x) \ri) \li(\hh(x) - \hy(x)\ri)\ri] \\
    &= \E_{x} \li[ 0 \ri] \\
    &= 0
  \end{align*}
