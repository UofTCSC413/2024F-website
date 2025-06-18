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
        \chead{{\bfseries Exercises - Week 4}}
        \rhead{{\bfseries Shkurti / Gilitschenski}}
        \lfoot{{}}
        \cfoot{{\thepage}}
        \rfoot{{}}
---
## Exercise 1 - Co-occurrence Matrix
{{< include questions/nlp-co_occurrence.md >}}

### Solution
{{< include questions/nlp-co_occurrence-sol.md >}}

## Exercise 2 - Convolutional Layers
{{< include questions/cnn-by_hand.md >}}

### Solution
{{< include questions/cnn-by_hand-sol.md >}}

## Exercise 3 - Sizes in MLPs Refresher
{{< include questions/mlp-sizes.md >}}

### Solution
{{< include questions/mlp-sizes-sol.md >}}

## Exercise 4 - Sizes in CNNs
{{< include questions/cnn-sizes.md >}}

### Solution
{{< include questions/cnn-sizes-sol.md >}}
