import numpy as np
import lyapunov
from dataclasses import dataclass
import classes
import uuid
import multiprocessing as mp
import pandas as pd
import os


sample_size = 30


def write_line(system: classes.System) -> str:

    """
    Function to help write lines to a dump file
    system: System (note that system.particle.position and system.particle.momentum are changed)
    returns: A string, the line to write
    """

    bounds = system.generate_bounds()

    # state below doesn't have momentum pre-factor

    position = np.zeros(system.num_spatial_dimensions)
    momentum = np.zeros(system.num_spatial_dimensions)

    seed = int(uuid.uuid4().time_low)
    np.random.seed(seed)

    for index, bound in enumerate(bounds):
        lower, upper = bound
        position[index] = np.random.uniform(lower, upper)
        momentum[index] = np.random.normal(scale=system.momentum_std)

    system.particle.position = position
    system.particle.momentum = momentum

    potential_energy = system.compute_potential()
    kinetic_energy = system.compute_kinetic()
    boltzmann_weight = np.exp(-potential_energy / system.thermostat.thermal_energy)
    exponent_calculator = lyapunov.ExponentCalculator(system=system, sample_size=sample_size)
    exponent_calculator.calculate_exponent()

    line = ''
    for coordinate in position:
        line += f'{coordinate},'
    for coordinate in momentum:
        line += f'{coordinate},'
    line += f'{potential_energy},{kinetic_energy},{boltzmann_weight},' \
            f'{exponent_calculator.exponent},{exponent_calculator.error}\n'

    return line


@dataclass
class GlobalExponentInformation:

    """
    Class storing information about a global exponent calculation
    """

    global_exponent: float
    weighted_standard_error: float
    number_of_points: int
    sample_size: int
    time_interval: list
    time_step: float

    def __repr__(self):

        return f'Global Exponent = {self.global_exponent}\n' \
               f'Weighted Standard Error = {self.weighted_standard_error}\n' \
               f'Number of Points = {self.number_of_points}\n' \
               f'Sample Size at Each Point = {self.sample_size}\n' \
               f'Time Interval = {self.time_interval}\n' \
               f'Time Step = {self.time_step}\n'


@dataclass
class GlobalExponentCalculator:

    """
    Helper class for calculating global exponent information
    """

    system: classes.System

    def generate_data(self, dump_file: str, num_points: int = 100, remove_dump: bool = False) -> None:

        """
        Method for generating and storing global exponent data
        dump_file: Name of dump file to write to
        num_points: Number of points to sample
        remove_dump: Option to remove the dump file after storing the data
        returns: None
        """

        num_cpus = mp.cpu_count()

        with mp.Pool(num_cpus - 1) as pool:
            print(f'{num_cpus} cpu pool opened, using {num_cpus - 1}')
            args = [self.system for _ in range(num_points)]
            lines = pool.map(write_line, args)

        header = ''

        for dimension in np.arange(self.system.num_spatial_dimensions):
            header += f'r_{dimension + 1},'
        for dimension in np.arange(self.system.num_spatial_dimensions):
            header += f'p_{dimension + 1},'

        header += 'potential,kinetic,weight,exponent,error\n'

        with open(dump_file, 'w') as file:
            print('dump file opened')
            file.write(header)
            for line in lines:
                file.write(line)
            print('data written to dump file')

        df = pd.read_csv(dump_file)

        if remove_dump:
            os.remove(dump_file)
            print('dump file removed')

        global_exponent = np.dot(df.weight, df.exponent) / np.sum(df.weight)
        weighted_error = np.sqrt(sum([weight * error ** 2 for weight, error in
                                      zip(df.weight, df.error)])) / sum(df.weight)

        self.information = GlobalExponentInformation(global_exponent=global_exponent,
                                                     weighted_standard_error=weighted_error,
                                                     number_of_points=num_points,
                                                     sample_size=sample_size,
                                                     time_interval=[self.system.time[0], self.system.time[-1]],
                                                     time_step=self.system.time_step)

    def write_data(self, file_name: str) -> None:

        """
        Method for writing global data to a file after calculating it
        file_name: File name to write global data to
        returns: None
        """

        with open(file_name, 'w') as file:
            file.write(self.information.__repr__())
