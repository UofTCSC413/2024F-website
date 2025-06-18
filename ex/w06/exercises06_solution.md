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
        \chead{{\bfseries Exercises - Week 6}}
        \rhead{{\bfseries Shkurti / Gilitschenski}}
        \lfoot{{}}
        \cfoot{{\thepage}}
        \rfoot{{}}
---
## Exercise 1 - Variance
{{< include questions/prob-variance.md >}}

### Solution
{{< include questions/prob-variance-sol.md >}}

## Exercise 2 - Variance / Bias Decomposistion
{{< include questions/ml-variance_bias_decomposition.md >}}

### Solution
{{< include questions/ml-variance_bias_decomposition-sol.md >}}

## Exercise 3 - Ensembling
Download the file `ex06-ensembling.ipynb` from quercus. It contains basic Pytorch code training a classifier on MNIST. Modify that code such that it trains an ensemble of 5-10 neural networks and computes their average prediction once trained.