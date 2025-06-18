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
        \chead{{\bfseries Exercises - Week 3}}
        \rhead{{\bfseries Shkurti / Gilitschenski}}
        \lfoot{{}}
        \cfoot{{\thepage}}
        \rfoot{{}}
---


## Exercise 1 - Maximum Likelihood Estimation Refresher
{{< include questions/prob-mle.md >}}

### Solution 
{{< include questions/prob-mle-sol.md >}}

## Exercise 2 - More Gradients
{{< include questions/backprop-batchnorm.md >}}

### Solution
{{< include questions/backprop-batchnorm-sol.md >}}

## Exercise 3 - Autodiff Modes
{{< include questions/autodiff-modes.md >}}

### Solution 
{{< include questions/autodiff-modes-sol.md >}}

## Exercise 4 - GloVe Embeddings
Open the notebook presented in class and work through it by trying some of the ideas presented therein for different word combinations.