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
        \chead{{\bfseries Exercises - Week 7}}
        \rhead{{\bfseries Shkurti / Gilitschenski}}
        \lfoot{{}}
        \cfoot{{\thepage}}
        \rfoot{{}}
---

The exercises this week involve some old material so you can check your learning and understanding.

## Exercise 1 - Maximum Likelihood Estimator
{{< include questions/prob-mle_exp_dist.md >}}

<!-- This exercise is from week 4 -->
## Exercise 2 - Convolutional Layers
{{< include questions/cnn-by_hand.md >}}

## Exercise 3 - Computational Parameter Counting
{{< include questions/pytorch-parameter_count.md >}}

## Exercise 4 - Influence Functions
{{< include questions/opt-influence_functions.md >}}