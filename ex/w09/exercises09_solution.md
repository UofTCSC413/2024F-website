---
solutions: true
format:
  html: default
  pdf:
    geometry: margin=0.75in
    header-includes: |
        \usepackage{../latex_packages/abbreviations}
        \usepackage{fancyhdr}
        \pagestyle{fancy}
        \let\headrule\empty
        \let\footrule\empty
        \lhead{{\bfseries CSC\,413}}
        \chead{{\bfseries Exercises - Week 9}}
        \rhead{{\bfseries Shkurti / Gilitschenski}}
        \lfoot{{}}
        \cfoot{{\thepage}}
        \rfoot{{}}
---



## Exercise 1 - Dot-Product Attention
{{< include questions/attn-dot_product_attention.md >}}

### Solution
{{< include questions/attn-dot_product_attention-sol.md >}}

## Exercise 2 - Attention in Transformers
{{< include questions/attn-transformers.md >}}

### Solution
{{< include questions/attn-transformers-sol.md >}}

## Exercise 3 - Scaled Dot-Product Attention by Hand
{{< include questions/attn-transformers_by_hand.md >}}

### Solution
{{< include questions/attn-transformers_by_hand-sol.md >}}