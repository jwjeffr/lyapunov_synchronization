import numpy as np
from dataclasses import dataclass
import classes
import copy
from scipy import stats
from custom_exceptions import RegressionCompatibilityError, TooManyIterationsError


def find_linear_region(independent_variable: np.ndarray,
                       dependent_variable: np.ndarray,
                       r_squared_threshold: float = 0.95,
                       max_iter: int = 500) -> tuple:

    """
    Function responsible for finding the initial linear region of a signal
    independent_variable: Independent variable
    dependent_variable: Dependent variable
    r_squared_threshold: Threshold for r^2
    max_iter: Maximum allowed regressions
    returns: The slope and standard error of the slope of the regression as well as the truncated signal
    """

    if not independent_variable.shape == dependent_variable.shape:
        raise RegressionCompatibilityError('Independent and dependent variable need to have the same shape')

    end_index = 2
    current_iter = 0
    regression = stats.linregress(independent_variable[:end_index], dependent_variable[:end_index])
    r_squared = regression.rvalue ** 2

    while r_squared >= r_squared_threshold:

        end_index += 1
        regression = stats.linregress(independent_variable[:end_index], dependent_variable[:end_index])
        r_squared = regression.rvalue ** 2
        current_iter += 1

        if not current_iter < max_iter:
            raise TooManyIterationsError('Maximum iterations reached for repeated regression')

    return regression, independent_variable[:end_index], dependent_variable[:end_index]

    # return regression.slope, regression.stderr, independent_variable[:end_index], dependent_variable[:end_index]


@dataclass
class DifferenceTrajectory:

    """
    Helper class to calculate difference trajectories
    """

    time: np.ndarray

    def output_to_file(self, file_name: str) -> None:

        """
        Output the difference trajectory to a .csv
        file_name: File name to write data to
        returns: None
        """

        header = 't, log_delta\n'

        with open(file_name, 'w') as file:

            file.write(header)

            for t, log_delta in zip(self.time, self.log_difference_values):
                file.write(f'{t},{log_delta}\n')

    def generate(self, system: classes.System, pert_magnitude: float = 1e-14) -> None:

        """
        Generates a difference trajectory and stores it
        system: The system
        pert_magnitude: Perturbation magnitude, needs to be small
        returns: None
        """

        self.time = system.time
        self.log_difference_values = np.zeros(self.time.shape)

        perturbed_system = copy.deepcopy(system)
        position_perturbation = np.random.rand(perturbed_system.num_spatial_dimensions)
        xi_momentum_perturbation = np.random.rand(perturbed_system.num_spatial_dimensions)

        position_perturbation *= pert_magnitude / np.linalg.norm(position_perturbation)
        xi_momentum_perturbation *= pert_magnitude / np.linalg.norm(xi_momentum_perturbation)

        perturbed_system.particle.position += position_perturbation
        perturbed_system.particle.momentum += xi_momentum_perturbation / perturbed_system.momentum_pre_factor

        trajectory = system.integrate()
        perturbed_trajectory = perturbed_system.integrate()

        # need to multiply by pre-factor to calculate phase space norms

        trajectory.states[:, system.num_spatial_dimensions:] *= system.momentum_pre_factor
        perturbed_trajectory.states[:, perturbed_system.num_spatial_dimensions] *= perturbed_system.momentum_pre_factor

        for index, (state, perturbed_state) in enumerate(zip(trajectory.states, perturbed_trajectory.states)):

            difference = np.linalg.norm(state - perturbed_state)
            self.log_difference_values[index] = np.log(difference)


@dataclass
class ExponentCalculator:

    """
    Helper class to calculate the Lyapunov exponent at a point
    """

    system: classes.System
    sample_size: int = 30

    def __post_init__(self) -> None:

        """
        Generates mean difference trajectories post initialization
        returns: None
        """

        log_difference_arrays = np.zeros((self.sample_size, self.system.time.shape[0]))
        for sample_number in np.arange(self.sample_size):

            difference_trajectory = DifferenceTrajectory(self.system.time)
            difference_trajectory.generate(self.system)

            log_difference_arrays[sample_number] = difference_trajectory.log_difference_values

        self.log_difference_arrays = log_difference_arrays
        self.mean_log_delta = np.mean(log_difference_arrays, axis=0)
        self.std_log_delta = np.std(log_difference_arrays, axis=0)

    def calculate_exponent(self) -> None:

        """
        Calculates the exponent and stores it (and other information)
        returns: None
        """

        regression, time_linear, log_delta_linear = find_linear_region(self.system.time, self.mean_log_delta)
        self.exponent = regression.slope
        self.error = regression.stderr
        self.intercept = regression.intercept
        self.time_linear = time_linear
        self.log_delta_linear = log_delta_linear
