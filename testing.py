import classes
import random_numbers
import numpy as np


def get_testing_system() -> classes.System:

    """
    Function that generates a "reasonable" system
    returns: The system, classes.System type
    """

    seed_pairs = [(2 ** 4, 2 ** 8),
                  (2 ** 12, 2 ** 16),
                  (2 ** 20, 2 ** 24)]

    length = 1000
    time_step = 0.05
    total_time = length * time_step
    all_sequences = []

    for first_seed, second_seed in seed_pairs:
        first_sequence = random_numbers.LCGSequence(length=length, seed=first_seed)
        second_sequence = random_numbers.LCGSequence(length=length, seed=second_seed)
        gaussian_sequence = random_numbers.GaussianSequence(first_lcg=first_sequence, second_lcg=second_sequence)

        all_sequences.append(gaussian_sequence)

    random_force = random_numbers.RandomForce(all_sequences=all_sequences)

    particle = classes.Particle(position=np.array([0.0, 0.0, 0.1]), momentum=np.zeros(3))

    thermostat = classes.Thermostat(friction_coefficient=1e-14, random_force=random_force)

    heights = np.array([1.0, 0.5, 1.2])
    centers = np.array([
        [1.0, 0.0, 3.0],
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    widths = np.array([1.0, 1.0, 1.0])

    surface = classes.Surface(heights=heights, centers=centers, widths=widths)

    super_basin = classes.SuperBasin(thermostat=thermostat, surface=surface)

    time = np.linspace(0, total_time, length)

    return classes.System(particle=particle,
                          thermostat=thermostat,
                          surface=surface,
                          super_basin=super_basin,
                          time=time)
