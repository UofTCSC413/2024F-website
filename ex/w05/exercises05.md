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
        \chead{{\bfseries Exercises - Week 5}}
        \rhead{{\bfseries Shkurti / Gilitschenski}}
        \lfoot{{}}
        \cfoot{{\thepage}}
        \rfoot{{}}
---


## Exercise 1 - Taylor Series
{{< include questions/calc-taylor.md >}}

## Exercise 2 - Eigenvalues, Eigenvectors
{{< include questions/linalg-evs_to_mat.md >}}

## Exercise 3 - SGD with Momentum
{{< include questions/opt-momentum.md >}}