# Project overview

The overarching goal of the project is to correlate Lyapunov exponents with the synchronization of molecular dynamical trajectories.

## What is a molecular dynamical trajectory?

A molecular dynamical trajectory is the trajectory (position and momenta as they evolve in time) of an atom/molecule.

## What is a Lyapunov exponent?

A Lyapunov exponent is a way to measure the divergence of two initially infinitesimally close trajectories in phase space:
  
<p align="center">
  <img src="https://github.com/jwjeffr/lyapunov_synchronization/blob/main/lyapunov_exponent.png?raw=true">
</p>

## What is synchronization in this context?

Molecular systems are often simulated at constant temperature, which is simulated by the Langevin equation:

<p align="center">
  <img src="https://github.com/jwjeffr/lyapunov_synchronization/blob/main/langevin_dynamics.png?raw=true">
</p>

The first term represents the effect of internal forces on the particle (modeled as a sum of Gaussian wells confined to a super-basin in this project), the second term simulates drag, and the last term simulates random "kicks" from the temperature control.

The last term introduces difficulty in the simulation: we need a random force. Simulating this random force requires picking a pseudorandom number generator, which generates an entirely deterministic sequence of numbers.

For long simulations, trajectories can become statistically independent of all other factors but the deterministic sequence of numbers. This means that two initially independent trajectories will become synchronized as they become statistically driven by the same sequence.
