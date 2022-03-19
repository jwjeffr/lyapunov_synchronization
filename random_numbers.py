import numpy as np
from dataclasses import dataclass
from custom_exceptions import LCGCompatibilityError


@dataclass
class LCGSequence:

    """

    Class responsible for generating PRNGs from the linear congruential generator

    """

    seed: int
    length: int
    multiplier: int = 1664525
    increment: int = 1013904223
    modulus: int = 2 ** 32

    def generate(self, uniform=True, inclusive=False, epsilon=1e-8) -> np.ndarray:

        """
        Generates an LCG sequence, stores into a numpy array

        uniform: Variable dictating whether to normalize the sequence or not
        inclusive: Variable dictating to exclude endpoints or not
        epsilon: Float controlling the shrinking associated with making the sequence inclusive
        returns: numpy array of LCG-generated pseudo random number generators
        """

        if not 0 < epsilon < 0.5:
            raise ValueError('epsilon needs to be between 0.0 and 0.5')

        sequence = np.zeros(self.length)
        sequence[0] = self.seed
        for index in np.arange(1, self.length):
            new_number = self.multiplier * sequence[index - 1] + self.seed
            new_number %= self.modulus
            sequence[index] = new_number

        if uniform:
            sequence /= self.modulus - 1

        if uniform and not inclusive:
            max_value = 1.0 - epsilon
            min_value = epsilon
            sequence = min_value + (max_value - min_value) * sequence

        return sequence


@dataclass
class GaussianSequence:

    """
    Class responsible for generating a Gaussian sequence from LCG sequences
    """

    first_lcg: LCGSequence
    second_lcg: LCGSequence

    def generate(self, mean=0.0, standard_deviation=1.0) -> np.ndarray:

        """
        Generates a Gaussian sequence, stores it in a numpy array

        mean: Mean of the Gaussian distribution
        standard_deviation: Standard deviation of the Gaussian distribution
        returns: numpy array of Gaussian-distributed numbers
        """

        if not self.first_lcg.length == self.second_lcg.length:
            raise LCGCompatibilityError('incompatible LCG sequence lengths')

        gaussian_sequence = np.zeros(self.first_lcg.length)
        first_sequence = self.first_lcg.generate()
        second_sequence = self.second_lcg.generate()

        for index, (first_number, second_number) in enumerate(zip(first_sequence, second_sequence)):

            number = np.sqrt(-2.0 * np.log(first_number)) * np.cos(2.0 * np.pi * second_number)
            gaussian_sequence[index] = number

        if not np.isclose(standard_deviation, 1.0):

            gaussian_sequence *= standard_deviation

        if not np.isclose(mean, 0.0):

            gaussian_sequence += mean

        return gaussian_sequence


@dataclass
class RandomForce:

    """
    Class responsible for generating a random time series that translates into the thermostat's force
    """

    all_sequences: list

    def generate(self) -> np.ndarray:

        """
        Generates the time series, stores it into a numpy array

        returns: numpy array of the component-wise Gaussian distributed numbers
        """

        random_force = []

        for gaussian_sequence in self.all_sequences:

            generated_sequence = gaussian_sequence.generate()
            random_force.append(generated_sequence)

        random_force = np.array(random_force).T
        return random_force
