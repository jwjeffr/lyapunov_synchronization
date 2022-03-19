# Project overview

The overarching goal of the project is to correlate Lyapunov exponents with the synchronization of molecular dynamical trajectories.

### What is a molecular dynamical trajectory?

A molecular dynamical trajectory is the trajectory (position and momenta as they evolve in time) of an atom/molecule.

### What is a Lyapunov exponent?

A Lyapunov exponent is a way to measure the divergence of two initially infinitesimally close trajectories in phase space:
  
<p align="center">
  <img src="https://github.com/jwjeffr/lyapunov_synchronization/blob/main/lyapunov_exponent.png?raw=true">
</p>

### What is synchronization in this context?

Molecular systems are often simulated at constant temperature (driven by what's called a thermostat), which is simulated by the Langevin equation:

<p align="center">
  <img src="https://github.com/jwjeffr/lyapunov_synchronization/blob/main/langevin_dynamics.png?raw=true">
</p>

The first term represents the effect of internal forces on the particle (modeled as a sum of Gaussian wells confined to a super-basin in this project), the second term simulates drag, and the last term simulates random "kicks" from the temperature control.

The last term introduces difficulty in the simulation: we need a random force. Simulating this random force requires picking a pseudorandom number generator, which generates an entirely deterministic sequence of numbers.

For long simulations, trajectories can become statistically independent of all other factors but the deterministic sequence of numbers. This means that two initially independent trajectories will become synchronized as they become statistically driven by the same sequence.

# File and code architecture

This project takes an object oriented approach. The fundamental objects defined are:

### Defined in random_numbers.py

#### LCGSequence
Stores the seed, length, multiplier, increment, and modulus of an [LCG-generated pseudorandom number sequence](https://en.wikipedia.org/wiki/Linear_congruential_generator) and generates said sequence

#### GaussianSequence
Stores the two separate LCGSequence objects and generates a sequence of Gaussian-distributed numbers from those sequences

#### RandomForce
Generates an N-dimensional random force from N GaussianSequence objects

### Defined in classes.py

#### Particle
Stores the position (r in the Langevin dynamics equation), momentum, and mass of the particle (m)

#### Thermostat
Stores the friction coefficient (gamma), random force, and thermal energy (kT)

#### Surface
Stores the heights, centers, and widths of all the Gaussian wells

#### SuperBasin
Calculates and stores parameters for the confining super basin

#### Trajectory
Stores a trajectory (time, states, potential energy, kinetic energy), helper methods to visualize a trajectory or dump it to a file

#### SystemParameters
Stores the Thermostat, Surface, SuperBasin, and time, and calculates other useful quantities. Also has a method that helps with random sampling to calculate averages

#### System
Inherits from SystemParameters. Stores information on the particle and a momentum pre-factor. Methods to calculate the potential energy, kinetic energy, and the force on the particle. Other methods to evolve the forward system in time.
