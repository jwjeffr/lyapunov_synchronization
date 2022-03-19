import classes
import random_numbers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import lyapunov
import testing


def main() -> None:

    plt.style.use('seaborn-paper')
    plt.rc('axes', axisbelow=True)
    plt.rc('font', size=20)
    color_map = sns.color_palette('flare', as_cmap=True)
    new_color_map = color_map(np.arange(color_map.N))
    uniform = np.linspace(0, 1, color_map.N)
    growth = 5
    a = 1 / (1 - np.exp(-growth))
    new_color_map[:, -1] = a * (1 - np.exp(-growth * uniform))
    new_color_map = ListedColormap(new_color_map)

    system = testing.get_testing_system()

    sample_size = 30

    exponent_calculator = lyapunov.ExponentCalculator(system=system, sample_size=sample_size)
    exponent_calculator.calculate_exponent()

    length = sample_size * system.time.shape[0]

    overall_log_differences = exponent_calculator.log_difference_arrays.reshape((1, length))[0]

    overall_time = system.time
    for _ in np.arange(1, sample_size):
        overall_time = np.append(overall_time, system.time)

    predicted_log_difference = exponent_calculator.exponent * system.time + exponent_calculator.intercept

    every = 50
    
    plt.plot(system.time, predicted_log_difference, color='black', label='Fit of initial\nlinear region', zorder=3)
    hist = plt.hist2d(overall_time, overall_log_differences, bins=100, cmap=new_color_map, zorder=0)
    eb = plt.errorbar(system.time, exponent_calculator.mean_log_delta, yerr=exponent_calculator.std_log_delta,
                      fmt='none', capsize=5, label=r'$\pm 1$ SD', elinewidth=1, ecolor='black', zorder=1,
                      errorevery=every, markeredgewidth=1)
    eb[-1][0].set_linestyle('--')

    plt.scatter(system.time[::every], exponent_calculator.mean_log_delta[::every], edgecolors='black',
                facecolors='white', label=r'$\ln\;\delta(t)$' + f' averaged over\n{sample_size} trajectories',
                zorder=2, linewidths=1)
    color_bar = plt.colorbar(hist[3])
    color_bar.set_label('Frequency')
    plt.grid()
    plt.xlabel('time (arb. unit)')
    plt.ylabel(r'$\ln\;\delta(t)$')
    plt.legend()
    plt.savefig('exponent_calculation.png', dpi=800, bbox_inches='tight')


if __name__ == '__main__':
    main()
