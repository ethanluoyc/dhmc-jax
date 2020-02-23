# Discontinuous Hamiltonian Monte Carlo

<img src="https://raw.githubusercontent.com/ethanluoyc/dhmc-jax/master/figures/dhmc.png" width="300">

Discontinuous Hamiltonian Monte Carlo (DHMC) (https://arxiv.org/abs/1705.08510) in JAX.

## Description

The repo includes implementation of a few MCMC algorithms in [JAX](https://github.com/google/jax). The design follows largely from [FunMCMC](https://github.com/tensorflow/probability/blob/master/discussion/fun_mcmc/fun_mcmc_lib.py).

notebooks/HMC.ipynb includes a Jupyter notebook that demonstrates (D)HMC
on a few problems.

## Installation

In order to set up the necessary environment:

1. create an environment `dhmc` with the help of [conda],
   ```
   conda env create -f environment.yaml
   ```
2. activate the new environment with
   ```
   conda activate dhmc
   ```
