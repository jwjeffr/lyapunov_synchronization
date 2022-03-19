import numpy as np
from dataclasses import dataclass
import random_numbers
from custom_exceptions import PRNGCompatibilityError
import matplotlib.pyplot as plt


@dataclass
class Particle:

    """
    Class representing the particle
    """

    position: np.ndarray
    momentum: np.ndarray
    mass: float = 1.0

    def __repr__(self) -> str:
        return f'mass = {self.mass}'


@dataclass
class Thermostat:

    """
    Class representing the thermostat
    """

    friction_coefficient: float
    random_force: random_numbers.RandomForce
    thermal_energy: float = 1.0

    def __post_init__(self) -> None:

        """
        Creates the necessary Gaussian sequence after initialization

        returns: None
        """

        self.prng_sequence = self.random_force.generate()

    def __repr__(self) -> str:
        return f'friction coefficient = {self.friction_coefficient}, thermal_energy = {self.thermal_energy}'


@dataclass
class Surface:

    """
    Class representing the Gaussian surface

    """

    heights: np.ndarray
    centers: np.ndarray
    widths: np.ndarray

    def __repr__(self) -> str:
        return f'heights = {self.heights.tolist()}, centers = {self.centers.tolist()}, widths = {self.widths.tolist()}'


@dataclass
class SuperBasin:

    """
    Class representing the super basin
    """

    thermostat: Thermostat
    surface: Surface

    def __post_init__(self) -> None:

        """
        Calculates and stores super basin parameters

        thermostat: the system's thermostat
        surface: the system's PES
        returns: None
        """

        self.stiffness = self.thermostat.thermal_energy / (np.sum(self.surface.widths)) ** 2
        self.center = np.mean(self.surface.centers, axis=0)

    def __repr__(self) -> str:
        return f'stiffness = {self.stiffness}, center = {self.center.tolist()}'


@dataclass
class Trajectory:

    """
    Class representing a particle's trajectory
    """

    time: np.ndarray
    states: np.ndarray
    potential_energies: np.ndarray
    kinetic_energies: np.ndarray

    def output_to_file(self, file_name: str) -> None:

        """
        Helper method that writes a trajectory to a file

        file_name: The name of the file to write to
        :return: None
        """

        num_spatial_dimensions = int(self.states[0].shape[0] / 2)

        header = 't,'
        for dimension in range(num_spatial_dimensions):
            header += f'x_{dimension + 1},'
        for dimension in range(num_spatial_dimensions):
            header += f'p_{dimension + 1},'
        header += f'pot,kin\n'

        with open(file_name, 'w') as file:
            file.write(header)
            for t, state, pot, kin in zip(self.time, self.states, self.potential_energies, self.kinetic_energies):
                line = f'{t},'
                for coord in state:
                    line += f'{coord},'
                line += f'{pot},{kin}\n'
                file.write(line)

    def plot(self, image_name: str) -> None:

        """
        Helper function for plotting the trajectory

        image_name: The file to save the image to
        returns: None
        """

        num_spatial_dims = int(self.states[0].shape[0] / 2)

        fig, axs = plt.subplots(nrows=3, ncols=num_spatial_dims, sharex=True, sharey=True)
        for index in np.arange(num_spatial_dims):
            position_axes = axs[0][index]
            position_axes.grid()
            position = self.states[:, index]
            momentum_axes = axs[1][index]
            momentum_axes.grid()
            momentum = self.states[:, index + num_spatial_dims]
            position_axes.plot(self.time, position)
            momentum_axes.plot(self.time, momentum)
            if index == 0:
                position_axes.set_ylabel('position\n(arb. unit)')
                momentum_axes.set_ylabel('momentum\n(arb. unit)')
                axs[2][index].plot(self.time, self.potential_energies)
                axs[2][index].set_ylabel('potential energy')
                axs[2][index].grid()
            elif index == 1:
                axs[2][index].plot(self.time, self.kinetic_energies)
                axs[2][index].set_ylabel('kinetic energy')
                axs[2][index].grid()
            elif index == 2:
                axs[2][index].plot(self.time, self.potential_energies + self.kinetic_energies)
                axs[2][index].set_ylabel('total energy')
                axs[2][index].grid()

        fig.supxlabel('time (arb. unit)')
        fig.savefig(image_name, bbox_inches='tight')
        plt.close(fig)


@dataclass
class SystemParameters:

    """
    SystemParameters class, makes it easier to average over particles later
    """

    thermostat: Thermostat
    surface: Surface
    super_basin: SuperBasin
    time: np.ndarray

    def __post_init__(self):

        self.surface.heights *= self.thermostat.thermal_energy
        self.num_dimensions = int(2 * np.size(self.surface.centers, axis=0))
        self.num_spatial_dimensions = int(self.num_dimensions / 2)
        self.time_step = self.time[1] - self.time[0]
        self.thermostat_weight = np.sqrt(1.0 - np.exp(-2.0 * self.thermostat.friction_coefficient * self.time_step))
        self.momentum_weight = np.exp(-self.thermostat.friction_coefficient * self.time_step)

    def generate_bounds(self, radius: float = 2.0) -> list:

        """
        Generate bounds for configuration space sampling
        radius: Radius for bounds
        returns: Bounds in each spatial dimension
        """

        vals = [[] for _ in np.arange(self.num_spatial_dimensions)]

        for width, center in zip(self.surface.widths, self.surface.centers):

            for dimension in np.arange(self.num_spatial_dimensions):

                lower = center[dimension] - radius * width
                upper = center[dimension] + radius * width
                vals[dimension].append(lower)
                vals[dimension].append(upper)

        return [[min(vals[dimension]), max(vals[dimension])] for dimension in np.arange(self.num_spatial_dimensions)]


@dataclass
class System(SystemParameters):

    """
    The system class, unfortunately has a lot of responsibilities
    """

    particle: Particle
    momentum_pre_factor: float = 1.0

    def __post_init__(self) -> None:

        """
        Modifies heights to be in terms of kT, also calculates and stores necessary parameters

        returns: None
        """

        super().__post_init__()
        self.thermostat_weight *= np.sqrt(self.particle.mass * self.thermostat.thermal_energy)
        self.momentum_std = 2.0 * np.sqrt(self.particle.mass * self.thermostat.thermal_energy)

    def __repr__(self) -> str:

        string = ''

        for key, value in self.__dict__.items():
            if key == 'time':
                string += f'{key}: initial time = {value[0]}, final time = {value[-1]}\n'
            else:
                string += f'{key}: {value}\n'

        return string

    def compute_potential(self) -> float:

        """
        Computes the potential energy at the current time step

        returns: The potential energy, a float
        """

        total_potential = 0.0

        for height, center, width in zip(self.surface.heights, self.surface.centers, self.surface.widths):

            exponent = -np.linalg.norm(self.particle.position - center) ** 2 / (2.0 * width ** 2)
            total_potential += height * np.exp(exponent)

        stretch = np.linalg.norm(self.particle.position - self.super_basin.center)
        total_potential += 0.5 * self.super_basin.stiffness * stretch ** 2

        return total_potential

    def compute_kinetic(self) -> float:

        """
        Computes the kinetic energy at the current time step

        returns: The kinetic energy, a float
        """

        return np.linalg.norm(self.particle.momentum) ** 2 / (2.0 * self.particle.mass)

    def compute_force(self) -> np.ndarray:

        """
        Computes the force at the current time step

        returns: The force, an array of floats
        """

        force = np.zeros(np.size(self.particle.position))

        for height, center, width in zip(self.surface.heights, self.surface.centers, self.surface.widths):

            exponent = -np.linalg.norm(self.particle.position - center) ** 2 / (2.0 * width ** 2)
            force += height * (self.particle.position - center) / (width ** 2) * np.exp(exponent)

        force += -self.super_basin.stiffness * (self.particle.position - self.super_basin.center)

        return force

    def time_evolve(self, index) -> None:

        """
        Evolves the system forward a time step

        index: Necessary parameter to access the proper random force
        returns: None
        """

        random_number = self.thermostat.prng_sequence[index]
        force = self.compute_force()

        self.particle.momentum += 0.5 * self.time_step * force
        self.particle.position += 0.5 * self.time_step * self.particle.momentum / self.particle.mass
        self.particle.momentum = self.particle.momentum * self.momentum_weight + random_number * self.thermostat_weight

    def integrate(self) -> Trajectory:

        """
        Evolves the system forward multiple time steps

        returns: None
        """

        solution = np.zeros((np.size(self.time), self.num_dimensions))
        potential_energies = np.zeros(np.size(self.time))
        kinetic_energies = np.zeros(np.size(self.time))
        state_init = np.hstack((self.particle.position, self.momentum_pre_factor * self.particle.momentum))
        potential_init = self.compute_potential()
        kinetic_init = self.compute_kinetic()
        solution[0] = state_init
        potential_energies[0] = potential_init
        kinetic_energies[0] = kinetic_init

        if any(not self.time.shape == seq.shape for seq in self.thermostat.prng_sequence.T):
            raise PRNGCompatibilityError('PRNG sequence incompatible with inputted time array')

        for index, time in enumerate(self.time[1:]):

            self.time_evolve(index + 1)
            solution[index + 1] = np.hstack((self.particle.position, self.momentum_pre_factor * self.particle.momentum))
            potential_energies[index + 1] = self.compute_potential()
            kinetic_energies[index + 1] = self.compute_kinetic()

        trajectory = Trajectory(time=self.time,
                                states=solution,
                                potential_energies=potential_energies,
                                kinetic_energies=kinetic_energies)

        return trajectory
