import tepimport
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    sets = tepimport.import_sets([0, 8, 9, 10, 11, 12])
    idv0 = np.delete(sets[0][1], list(range(22, 41)), axis=0)
    for i in range(5):
        fault = np.delete(sets[i + 1][2], list(range(22, 41)), axis=0)
        mean_err = np.abs((np.mean(idv0, axis=1) - np.mean(fault, axis=1))
                          / np.mean(idv0, axis=1)) * 100
        std_err = np.abs((np.std(idv0, axis=1) - np.std(fault, axis=1))
                         / np.std(idv0, axis=1)) * 100
        max_err = np.abs((np.max(idv0, axis=1) - np.max(fault, axis=1))
                         / np.max(idv0, axis=1)) * 100
        min_err = np.abs((np.min(idv0, axis=1) - np.min(fault, axis=1))
                         / np.min(idv0, axis=1)) * 100
        plt.subplot(5, 1, i + 1)
        plt.plot(mean_err, label='Mean')
        plt.plot(std_err, label='Standard Deviation')
        plt.plot(max_err, label='Maximum')
        plt.plot(min_err, label='Minimum')
        plt.ylabel(f'IDV({i + 8})')
        if i == 0:
            plt.legend()
            plt.title('Percent Change From Training Data')
        elif i == 4:
            plt.xlabel('Variable #')
    idv9 = np.delete(sets[2][2], list(range(22, 41)), axis=0)
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(idv0[8, :], label='Training')
    ax[0].plot(idv9[8, :], label='IDV(9)', color='k')
    ax[0].legend()
    ax[0].set_title('Reactor Temperature')
    ax[0].set_ylabel('Temperature (deg C)')

    ax[1].plot(idv0[8, 1:] - idv0[8, :-1], label='Training')
    ax[1].plot(idv9[8, 1:] - idv9[8, :-1], label='IDV(9)', color='k')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Change in Temperature (deg C)')
    plt.show()
