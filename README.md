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
./lgcp 7000 15000 5 50 40 15 > lgcp.log 2>&1 &
```
You will need a computer with GPU and have to [install CUDA](https://developer.nvidia.com/cuda-downloads).

You may need to issue a `disown` command.

# Manual

At present please consult the usage:
```
Usage: lgcp Burnin Iters Adjust AdjustWin Thin Save [GPU]

   Burnin  -- The burn-in period of the HMC
   Iters   -- The total number of iterations AFTER burn-in
   Adjust  -- How often to adjust the stepsize
   AdjustWindow
           -- Chain window when adjusting the stepsize
   Thin    -- How often to save the running sum of the GPs
   Save    -- How often to save snapshots of the GPs
   GPU     -- GPU device number (defaults to 0)

The following files are expected in the ./inputs directory:

   setup.txt: Contains following values, one per line:
       * Total number of elements in the initial grid. The program
         will figure out how many there are in the extended grid
       * Total number of points (foci)
       * Total number of point patterns (contrasts/studies)
       * Total number of covariates
       * Total number of spatially varying covariates
       * Total number of HMC leapfrog steps
       * Seed
       * HMC mass parameters (4 values), if one wants to see between-type
         comparisons

   seed.dat: 3 long integers

   rho.txt: Correlation decay parameters, one for each spatially varying
            covariate

   sigma.txt: Marginal standard deviations, one for each spatially varying
              covariate

   beta.txt: Overall mean parameter, one for each covariate

   gamma.txt: Standard normal variates, 144*192*144=3981312 for each spatially 
              varying covariate.  If missing, random numbers are generated.
```
