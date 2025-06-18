---
solutions: true
geometry: margin=0.75in
format:
  html: default
  pdf:
    header-includes: |
        \usepackage{../latex_packages/abbreviations}
        \usepackage{fancyhdr}
        \pagestyle{fancy}
        \let\headrule\empty
        \let\footrule\empty
        \lhead{{\bfseries CSC\,413}}
        \chead{{\bfseries Exercises - Week 10}}
        \rhead{{\bfseries Shkurti / Gilitschenski}}
        \lfoot{{}}
        \cfoot{{\thepage}}
        \rfoot{{}}
---

## Exercise 1 - Transposed Convolution Output Sizes
{{< include questions/tconv-sizes.md >}}

## Exercise 2 - Transposed Convolution Parameter Sizes
{{< include questions/tconv-params.md >}}

## Exercise 3 - Transposed Convolution by Hand
{{< include questions/tconv-by_hand.md >}}