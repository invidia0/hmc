# Hamiltonian Monte Carlo in JAX
JAX implementation of the HMC following Betancourt's paper.

Find my notes about the paper [here](hmc.md).

## Requirements

Please follow the [JAX installation instructions](https://docs.jax.dev/en/latest/installation.html) to install JAX with the appropriate backend for your hardware.

## Todo
- [ ] Add NUTS
- [ ] Implement `vmap` version for chain parallelization
- [ ] Benchmark

## References
- [Betancourt's paper](https://arxiv.org/abs/1701.02434)
- [JAX](https://jax.readthedocs.io/en/latest/)