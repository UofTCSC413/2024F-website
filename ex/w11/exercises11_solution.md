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
        \chead{{\bfseries Exercises - Week 11}}
        \rhead{{\bfseries Shkurti / Gilitschenski}}
        \lfoot{{}}
        \cfoot{{\thepage}}
        \rfoot{{}}
---

<!-- From week 5 -->
## Exercise 1 - Eigenvalues and Eigenvectors
{{< include questions/linalg-evs_to_mat.md >}}

### Solution
{{< include questions/linalg-evs_to_mat-sol.md >}}

<!-- From week 7 -->
## Exercise 2 - Parameter Counting
{{< include questions/pytorch-parameter_count.md >}}

### Solution
{{< include questions/pytorch-parameter_count-sol.md >}}

<!-- From week 7 -->
## Exercise 3 - Convolutional Layers
{{< include questions/cnn-by_hand.md >}}

### Solution
{{< include questions/cnn-by_hand-sol.md >}}

<!-- From week 9 -->
## Exercise 4 - Scaled Dot-Product Attention
{{< include questions/attn-transformers_by_hand.md >}}

### Solution
{{< include questions/attn-transformers_by_hand-sol.md >}}