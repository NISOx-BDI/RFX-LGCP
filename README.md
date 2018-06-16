# RFX-LGCP
Random Effects Log Gaussian Cox Process Model for Neuroimaging Coordinate-Based Meta Analysis

This repository has code for the following paper:

Pantelis Samartsidis,
Claudia R. Eickhoff,
Simon B. Eickhoff
Tor D. Wager,
Lisa Feldman Barrett,
Shir Atzil,
Timothy D. Johnson,
Thomas E. Nichols (2018).
Bayesian log-Gaussian Cox process regression: applications to meta-analysis of neuroimaging working memory studies. _Journal of the Royal Statistical Society. Series C (Applied Statistics)_, _in press_.

# Reproducing the paper's results

To reproduce the results in our paper:

```console
make
./lgcp 7000 15000 5 50 40 15 > lgcp.log 2>&1 disown
```
You will need a computer with GPU and have CUDA installed.

# Manual

_Coming soon!_
