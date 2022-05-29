import matplotlib.pyplot as plt
import numpy as np
import json
import os
import cirq
import pickle
import seaborn as sns
from scipy.sparse import csc_matrix, load_npz
import cirq_google as cg
from scipy.optimize import curve_fit
import math


# from scipy.sparse.linalg import inv


def main():
    plt.rc('font', size=16)
    # Parameters #
    # Dimensionality of the QCA
    d = 1

    # Max time steps (QCA cycles)
    t_max = 30

    # Number of QCA sites
    size = 19

    no_counts = 100_000  # Number of histogram repetitions per measurement
    exp_reps = 4  # Number of experimental repetitions
    activation_unitaries = ["H"]
    rule = "T6"
    # mod_number = int(size + 1) // int(4)

    initial_conditions = [[(size - 1) // int(2)]]  # ,
    # initial_conditions = [[0, 1, 3, 4]]

    # initial_conditions = [[i for i in range(size) if i % mod_number == mod_number - 1]]

    # initial_conditions = [[5, 11]]
    # initial_conditions = [[3, 8, 13]]

    # initial_conditions = [[3, 8, 13], [2, 6, 11, 14], [2, 5, 8, 11, 14], [1, 4, 7, 9, 12, 15],
    #                      [0, 3, 6, 8, 10, 13, 16],
    #                       [0, 2, 5, 7, 9, 11, 14, 16], [0, 2, 4, 6, 8, 10, 12, 14, 16]]

    sim_mode = 'cirq'  # 'engine' or 'cirq'
    two_qubit_gate = "parasitic_root_iSWAP"  # 'CZ' or 'Sycamore' or 'root_iSWAP'
    processor = 'weber'  # Or 'pacific' or something else ('NA' for 'cirq' sim_mode)
    observables = 'only_x'  # 'only_z' (correlators) or 'mutual_information' or only_x

    ising_mitigation = "PS_False"  # Whether to apply post-selection or not
    readout_correction = "RC_False"  # Whether to apply readout correction or not
    floquet_calibration = False  # Or False
    which_routine = "clustering_and_disparity_plots"

    for i in range(0, len(initial_conditions)):
        initial_site_indices = initial_conditions[i]

        post_processing_invariant = calculate_z_invariant(size=size,
                                                          initial_site_indices=initial_site_indices)

        if which_routine == "output_density_matrices":
            output_density_matrices(dim=d, size=size, t_max=t_max, initial_site_indices=initial_site_indices,
                                    no_counts=no_counts, experimental_repetitions=exp_reps,
                                    rule=rule, activation_unitaries=activation_unitaries, sim_mode=sim_mode,
                                    two_qubit_gate=two_qubit_gate, processor=processor, observables=observables)

        elif which_routine == "population_plots":
            population_plots(dim=d, size=size, t_max=t_max, initial_site_indices=initial_site_indices,
                             no_counts=no_counts, experimental_repetitions=exp_reps,
                             rule=rule, activation_unitaries=activation_unitaries, sim_mode=sim_mode,
                             two_qubit_gate=two_qubit_gate, processor=processor, observables=observables,
                             ising_mitigation=ising_mitigation,
                             invariant=post_processing_invariant, readout_correction=readout_correction,
                             floquet_calibration=floquet_calibration)

        elif which_routine == "clustering_and_disparity_plots":
            clustering_and_disparity_plots(dim=d, size=size, t_max=t_max, initial_site_indices=initial_site_indices,
                                           no_counts=no_counts, experimental_repetitions=exp_reps,
                                           rule=rule, activation_unitaries=activation_unitaries, sim_mode=sim_mode,
                                           two_qubit_gate=two_qubit_gate, processor=processor, observables=observables,
                                           ising_mitigation=ising_mitigation,
                                           invariant=post_processing_invariant, readout_correction=readout_correction,
                                           floquet_calibration=floquet_calibration)

        elif which_routine == "entropy_plot":
            if observables != 'mutual_information':
                continue
            entropy_plot(dim=d, size=size, t_max=t_max, initial_site_indices=initial_site_indices,
                         no_counts=no_counts, experimental_repetitions=exp_reps,
                         rule=rule, activation_unitaries=activation_unitaries, sim_mode=sim_mode,
                         two_qubit_gate=two_qubit_gate, processor=processor, observables=observables)

        else:
            raise ValueError("Not a valid routine.")
    return


def qubits_generator(origin: tuple, steps: list):
    # Want the output of this function to give a qubit list. Origin and steps will be pasted by hand for now.
    qubit_list = [cirq.GridQubit(origin[0], origin[1])]  # Initializes list with origin qubit
    next_coords = origin
    for step in steps:
        if step == 0:
            next_coords = (next_coords[0], next_coords[1] - 1)
        elif step == 1:
            next_coords = (next_coords[0] - 1, next_coords[1])
        elif step == 2:
            next_coords = (next_coords[0], next_coords[1] + 1)
        elif step == 3:
            next_coords = (next_coords[0] + 1, next_coords[1])
        else:
            raise ValueError('Not a valid step!')
        qubit_list.append(cirq.GridQubit(next_coords[0], next_coords[1]))
    # print(qubit_list)
    return qubit_list


def which_chain(config_no: int, size: int):  # , two_qubit_gate: str):
    if size == 5:
        qubit_configs = [qubits_generator((6, 5), [3, 2, 1, 2]),
                         qubits_generator((7, 6), [1, 0, 3, 3]),
                         qubits_generator((4, 5), [1, 2, 3, 3]),
                         qubits_generator((7, 6), [0, 1, 2, 1]),
                         qubits_generator((5, 6), [3, 0, 3, 2]),
                         qubits_generator((7, 5), [1, 2, 1, 1]),
                         qubits_generator((6, 7), [0, 0, 3, 2]),
                         qubits_generator((3, 6), [3, 3, 3, 0]),
                         qubits_generator((6, 5), [2, 3, 0, 3]),
                         qubits_generator((3, 5), [3, 2, 3, 0]),
                         qubits_generator((6, 6), [1, 1, 0, 1]),
                         qubits_generator((6, 7), [0, 0, 3, 2])]
    elif size == 7:
        qubit_configs = [qubits_generator((4, 5), [1, 2, 3, 3, 3, 0]),
                         qubits_generator((6, 5), [2, 1, 1, 1, 0, 3]),
                         qubits_generator((4, 7), [0, 3, 3, 0, 3, 3]),
                         qubits_generator((7, 5), [1, 2, 1, 1, 1, 0]),
                         qubits_generator((8, 5), [1, 1, 2, 1, 1, 0]),
                         qubits_generator((8, 5), [1, 2, 1, 1, 1, 1]),
                         qubits_generator((8, 5), [1, 2, 1, 1, 1, 0]),
                         qubits_generator((7, 4), [2, 2, 1, 1, 1, 2]),
                         qubits_generator((3, 7), [0, 3, 3, 3, 3, 0]),
                         qubits_generator((6, 5), [2, 3, 0, 3, 0, 1]),
                         qubits_generator((3, 7), [0, 3, 3, 0, 1, 1])]
    elif size == 9:
        qubit_configs = [qubits_generator((3, 6), [0, 3, 2, 3, 3, 3, 0, 3]),
                         qubits_generator((6, 5), [3, 2, 1, 1, 1, 1, 2, 2]),
                         qubits_generator((4, 7), [0, 3, 3, 0, 3, 0, 0, 1]),
                         qubits_generator((8, 4), [2, 1, 1, 2, 1, 1, 0, 3]),
                         qubits_generator((3, 4), [2, 3, 3, 2, 3, 0, 3, 2]),
                         qubits_generator((8, 5), [1, 1, 2, 1, 0, 1, 1, 1]),
                         qubits_generator((6, 3), [3, 2, 2, 1, 2, 1, 1, 0]),
                         qubits_generator((5, 6), [1, 0, 1, 1, 0, 0, 3, 3]),
                         qubits_generator((2, 3), [2, 2, 3, 3, 3, 2, 1, 1]),
                         qubits_generator((4, 2), [1, 2, 1, 2, 2, 3, 3, 3]),
                         qubits_generator((5, 4), [3, 2, 3, 2, 1, 1, 1, 1])]
    elif size == 11:
        qubit_configs = [qubits_generator((1, 4), [3, 2, 3, 2, 3, 3, 3, 0, 3, 3]),
                         qubits_generator((1, 4), [3, 2, 3, 2, 3, 3, 3, 0, 3, 0]),
                         qubits_generator((6, 4), [2, 3, 2, 1, 1, 0, 0, 1, 1, 2]),
                         qubits_generator((3, 4), [2, 2, 2, 3, 0, 3, 3, 0, 3, 3]),
                         qubits_generator((4, 4), [1, 1, 2, 2, 3, 3, 3, 3, 3, 0]),
                         qubits_generator((7, 4), [2, 2, 1, 0, 1, 2, 1, 1, 1, 1]),
                         qubits_generator((4, 7), [1, 0, 3, 3, 3, 0, 3, 0, 0, 3]),
                         qubits_generator((8, 4), [2, 1, 2, 1, 1, 0, 0, 1, 1, 1]),
                         qubits_generator((3, 4), [3, 3, 2, 2, 3, 0, 3, 3, 0, 1]),
                         qubits_generator((9, 4), [1, 2, 1, 2, 1, 1, 1, 1, 0, 1]),
                         qubits_generator((6, 4), [0, 1, 2, 2, 2, 3, 0, 3, 0, 3])]
    elif size == 13:
        qubit_configs = [qubits_generator((6, 3), [3, 2, 3, 2, 1, 2, 1, 1, 1, 1, 0, 3]),
                         qubits_generator((7, 6), [0, 1, 2, 1, 1, 1, 0, 1, 0, 1, 2, 1]),
                         qubits_generator((3, 3), [1, 2, 2, 3, 2, 3, 3, 3, 3, 0, 1, 0]),
                         qubits_generator((7, 3), [2, 3, 2, 1, 2, 1, 0, 1, 2, 1, 0, 1]),
                         qubits_generator((6, 4), [1, 1, 2, 1, 2, 3, 3, 0, 3, 3, 2, 1]),
                         qubits_generator((5, 4), [3, 2, 2, 1, 1, 0, 1, 1, 0, 0, 3, 3]),
                         qubits_generator((8, 4), [2, 1, 2, 1, 1, 1, 1, 0, 1, 0, 1, 2]),
                         qubits_generator((4, 3), [3, 2, 1, 2, 2, 3, 3, 0, 3, 0, 0, 3]),
                         qubits_generator((2, 8), [3, 0, 0, 0, 3, 2, 3, 0, 3, 3, 0, 0]),
                         qubits_generator((4, 3), [0, 1, 2, 1, 2, 2, 3, 3, 3, 3, 3, 0]),
                         qubits_generator((9, 4), [1, 1, 2, 1, 2, 1, 0, 1, 0, 3, 0, 1])]
    elif size == 15:
        qubit_configs = [qubits_generator((6, 7), [0, 1, 1, 1, 0, 3, 0, 3, 0, 1, 1, 1, 2, 1]),
                         qubits_generator((4, 3), [0, 1, 2, 2, 2, 3, 2, 3, 3, 0, 3, 0, 3, 2]),
                         qubits_generator((2, 3), [2, 3, 2, 3, 3, 2, 3, 0, 3, 3, 0, 1, 0, 1]),
                         qubits_generator((5, 5), [0, 0, 1, 1, 1, 2, 2, 3, 3, 2, 3, 3, 0, 3]),
                         qubits_generator((6, 7), [0, 3, 0, 1, 0, 1, 1, 2, 3, 2, 1, 2, 1, 2]),
                         qubits_generator((6, 3), [3, 2, 2, 1, 2, 1, 0, 1, 1, 0, 0, 3, 0, 1]),
                         qubits_generator((4, 7), [1, 0, 0, 3, 2, 3, 3, 3, 0, 1, 0, 1, 0, 1]),
                         qubits_generator((2, 4), [0, 3, 3, 2, 2, 3, 2, 3, 3, 0, 3, 0, 1, 0]),
                         qubits_generator((2, 5), [0, 0, 3, 0, 3, 2, 3, 2, 1, 2, 3, 2, 3, 0]),
                         qubits_generator((5, 4), [0, 1, 1, 2, 2, 2, 3, 0, 3, 2, 3, 3, 0, 3]),
                         qubits_generator((2, 5), [3, 3, 2, 3, 0, 0, 0, 1, 1, 0, 3, 3, 0, 1])]
    elif size == 17:
        qubit_configs = [qubits_generator((6, 4), [2, 3, 2, 1, 1, 1, 1, 0, 1, 0, 0, 3, 0, 3, 3, 0]),
                         qubits_generator((7, 3), [2, 2, 1, 2, 1, 1, 1, 0, 3, 0, 0, 0, 1, 2, 1, 2]),
                         qubits_generator((8, 3), [1, 2, 2, 2, 1, 0, 1, 2, 1, 0, 1, 0, 1, 0, 3, 3]),
                         qubits_generator((6, 1), [1, 2, 1, 1, 2, 3, 3, 2, 2, 1, 2, 3, 3, 3, 0, 0]),
                         qubits_generator((8, 5), [1, 1, 1, 1, 2, 1, 0, 0, 1, 0, 3, 0, 3, 3, 0, 0]),
                         qubits_generator((3, 8), [0, 3, 0, 3, 3, 0, 3, 0, 0, 1, 1, 1, 0, 1, 2, 1]),
                         qubits_generator((3, 6), [0, 3, 2, 3, 3, 0, 3, 0, 1, 1, 0, 1, 0, 1, 2, 1]),
                         qubits_generator((7, 2), [2, 2, 2, 1, 2, 1, 1, 1, 0, 1, 0, 0, 3, 3, 2, 3]),
                         qubits_generator((2, 8), [3, 0, 3, 0, 3, 3, 0, 3, 0, 0, 1, 1, 1, 0, 1, 2]),
                         qubits_generator((3, 5), [0, 0, 0, 3, 2, 3, 2, 1, 2, 2, 3, 3, 0, 3, 0, 3]),
                         qubits_generator((7, 3), [2, 2, 1, 2, 1, 1, 1, 0, 3, 0, 1, 0, 1, 2, 1, 2])]
    elif size == 19:
        qubit_configs = [qubits_generator((5, 2), [1, 1, 2, 1, 2, 2, 3, 2, 3, 3, 3, 0, 3, 0, 1, 1, 1, 2]),
                         qubits_generator((3, 2), [2, 3, 2, 3, 3, 0, 3, 2, 3, 2, 1, 1, 2, 1, 1, 0, 1, 2]),
                         qubits_generator((4, 2), [1, 2, 1, 2, 3, 2, 3, 3, 2, 3, 3, 0, 1, 0, 0, 3, 2, 3]),
                         qubits_generator((4, 7), [0, 1, 0, 1, 0, 0, 3, 0, 3, 2, 3, 2, 3, 3, 2, 1, 2, 3]),
                         qubits_generator((4, 1), [3, 2, 2, 3, 3, 2, 2, 1, 2, 1, 1, 1, 0, 3, 0, 0, 1, 0]),
                         qubits_generator((4, 6), [3, 3, 0, 0, 1, 0, 1, 0, 1, 2, 1, 2, 2, 3, 2, 2, 2, 1]),
                         qubits_generator((8, 5), [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 2, 3, 3, 3, 3, 0, 1]),
                         qubits_generator((5, 5), [3, 2, 1, 1, 2, 1, 0, 0, 1, 0, 0, 3, 0, 3, 2, 3, 2, 1]),
                         qubits_generator((3, 4), [3, 3, 3, 3, 3, 2, 1, 1, 2, 1, 0, 1, 2, 1, 0, 1, 0, 0]),
                         qubits_generator((4, 1), [3, 2, 2, 2, 3, 2, 2, 1, 0, 1, 1, 1, 0, 0, 3, 0, 3, 2]),
                         qubits_generator((2, 6), [3, 3, 3, 3, 0, 3, 0, 0, 1, 1, 1, 1, 2, 2, 3, 0, 3, 3])]
    elif size > 19:
        project_id = ''
        engine = cg.Engine(project_id=project_id)
        processor_object = engine.get_processor('weber')
        device = processor_object.get_device([cg.SQRT_ISWAP_GATESET])
        line = cg.line_on_device(device, size)
        qubit_configs = [list(line) for i in range(0, 4)]
    else:
        raise ValueError('Not a valid chain size!')
    return qubit_configs[config_no]


def calculate_z_invariant(size, initial_site_indices):
    initial_state = [0 for i in range(0, size)]
    for i in initial_site_indices:  # range(0, len(initial_site_indices)):
        initial_state[i] = 1
    print("initial state:", initial_state)
    invariant_counter = 0
    padded_state = [0] + initial_state + [0]
    for i in range(0, len(padded_state) - 1):
        if padded_state[i] == padded_state[i + 1]:
            invariant_counter += 1
        else:
            invariant_counter -= 1
    return invariant_counter


def exp_func(x, a, b):
    return a * np.exp(-b * x)


# Next need to code routines to calculate complex network measures and plot !
def clustering_and_disparity_plots(dim, size, t_max, initial_site_indices, no_counts, experimental_repetitions,
                                   rule, activation_unitaries, sim_mode, two_qubit_gate,
                                   processor, observables, ising_mitigation, invariant, readout_correction,
                                   floquet_calibration):
    avg_clusterings = np.zeros(t_max)
    avg_disparities = np.zeros(t_max)
    clustering_fits = []
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', 'black']
    fig, axs = plt.subplots(2, sharex='col')
    for r in range(0, experimental_repetitions):
        print(r)
        # if r == 2:
        #     continue
        # elif r == 5:
        #     continue
        clusterings = np.zeros(t_max)
        disparities = np.zeros(t_max)
        qubits = which_chain(r, size)
        for t in range(2, t_max):
            circuit_parameters = (dim, len(qubits), t_max, initial_site_indices, no_counts,
                                  experimental_repetitions, rule, activation_unitaries, sim_mode, two_qubit_gate,
                                  processor, observables, floquet_calibration, r, t)
            results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
            if observables == 'mutual_information':
                mi_matrix = construct_mutual_information_matrix(qubits, results_dict)
            elif observables == 'only_z':
                mi_matrix = construct_classical_mutual_information_matrix(qubits, results_dict, ising_mitigation,
                                                                          invariant, readout_correction, observables)
            elif observables == 'only_x':
                mi_matrix = construct_classical_mutual_information_matrix(qubits, results_dict, ising_mitigation,
                                                                          invariant, readout_correction, observables)
            else:
                raise ValueError('Not a valid set of observables!')

            avg_clusterings[t] += clustering(mi_matrix=mi_matrix) / float(experimental_repetitions)
            avg_disparities[t] += disparity(mi_matrix=mi_matrix) / float(experimental_repetitions)
            clusterings[t] = clustering(mi_matrix=mi_matrix)
            disparities[t] = disparity(mi_matrix=mi_matrix)

        # Plot individual clusterings and disparities
        axs[0].plot(np.array([i for i in range(0, t_max)]), clusterings, linestyle='solid', marker='', alpha=0.4)
        axs[1].plot(np.array([i for i in range(0, t_max)]), disparities, linestyle='solid', marker='', alpha=0.4)
        # if ising_mitigation == "PS_False":
        #     # Now fit
        #     times = np.array([i for i in range(3, 16)])
        #     popt, pcov = curve_fit(exp_func, times, clusterings[3:16])
        #     print(popt)
        #     clustering_fits.append(1./popt[1])
        #     axs[0].plot(times, exp_func(times, *popt), linestyle='dotted', marker='', color='tab:grey')

        root_dir = os.getcwd()
        top_dir = "{}d".format(circuit_parameters[0])
        bot_dir = "confirmation_data_qb{}_tm{}_isi{}_nc{}_er{}_ru{}_au{}_{}_{}_{}_{}_fc{}".format(
            *circuit_parameters[1:len(circuit_parameters) - 2])
        dir_path = os.path.join(root_dir, top_dir, bot_dir)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        np.savez(dir_path + '/' + str(r) + '_clustering_and_disparity_' + readout_correction + ising_mitigation + '.npz'
                 , np.array([i for i in range(0, t_max)]), clusterings,
                 disparities)
    avg_clustering_decay = np.average(np.array(clustering_fits))
    std_clustering_decay = np.std(np.array(clustering_fits))

    axs[0].plot(np.array([i for i in range(0, t_max)]), avg_clusterings, linestyle='--', marker='o')
    axs[1].plot(np.array([i for i in range(0, t_max)]), avg_disparities, linestyle='--', marker='o')
    # if ising_mitigation == "PS_False":
    #     times = np.array([i for i in range(3, 16)])
    #     popt, pcov = curve_fit(exp_func, times, avg_clusterings[3:16])
    #     print(popt)
    #     clustering_fits.append(1. / popt[1])
    #     axs[0].plot(times, exp_func(times, *popt), linestyle='dotted', marker='', color='k')

    axs[0].set_ylim(1e-6, 2)
    axs[1].set_ylim(-0.05, 1)
    axs[0].set_ylabel(r"Clustering $(\mathcal{C})$")
    axs[1].set_ylabel(r"Disparity $(\mathcal{Y})$")
    axs[1].set_xlabel(r"QCA Cycle $(t)$")
    axs[0].set_yscale('log')
    # axs[1].set_yscale('log')

    fig.tight_layout()

    root_dir = os.getcwd()
    top_dir = "{}d".format(circuit_parameters[0])
    bot_dir = "confirmation_data_qb{}_tm{}_isi{}_nc{}_er{}_ru{}_au{}_{}_{}_{}_{}_fc{}".format(
        *circuit_parameters[1:len(circuit_parameters) - 2])
    dir_path = os.path.join(root_dir, top_dir, bot_dir)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    np.savez(dir_path + '/avg_clustering_and_disparity_' + readout_correction + ising_mitigation + '.npz',
             np.array([i for i in range(0, t_max)]), avg_clusterings,
             avg_disparities)

    # with open(dir_path + '/rep{}_ts{}.json'.format(*circuit_parameters[-2:]), 'r') as f:
    #     results_dict = json.load(f)
    file = open(dir_path + "/clustering_decay.txt", "w+")
    print(f"Qubit configuration averaged clustering decay time: {avg_clustering_decay}", file=file)
    print(f"Qubit coniguration averaged decay time stdev: {std_clustering_decay}", file=file)
    file.close()

    plt.savefig(dir_path + '/clustering_and_disparity' + readout_correction + ising_mitigation + '.pdf',
                bbox_inches="tight")
    plt.close(fig)
    return


def output_density_matrices(dim, size, t_max, initial_site_indices, no_counts, experimental_repetitions,
                            rule, activation_unitaries, sim_mode, two_qubit_gate,
                            processor, observables):
    for r in range(0, experimental_repetitions):
        qubits = which_chain(r, size)
        partitions_dict = construct_hash_partitions(qubits=qubits)
        density_matrix_dictionary = {}
        for t in range(0, t_max):
            circuit_parameters = (dim, len(qubits), t_max, initial_site_indices, no_counts,
                                  experimental_repetitions, rule, activation_unitaries, sim_mode, two_qubit_gate,
                                  processor, observables, r, t)
            results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
            counts_dicts_list = [results_dict["All_X"], results_dict["All_Y"], results_dict["All_Z"]]
            for i in range(0, len(qubits)):
                dens_mat = local_entropy(site_index=i, counts_dicts_list=counts_dicts_list, qubits=qubits)[0]
                density_matrix_dictionary.update({"t" + str(t) + "_r" + str(r) + "_q" + str(i): dens_mat})
            for i in range(0, len(qubits) - 1):
                for j in range(i + 1, len(qubits)):
                    dens_mat = two_point_entropy(site_1_index=i, site_2_index=j, results_dict=results_dict,
                                                 partitions_dict=partitions_dict, qubits=qubits)[0]
                    density_matrix_dictionary.update(
                        {"t" + str(t) + "_r" + str(r) + "_q" + str(i) + "_qq" + str(j): dens_mat})

    root_dir = os.getcwd()
    top_dir = "{}d".format(circuit_parameters[0])
    bot_dir = "confirmation_data_qb{}_tm{}_isi{}_nc{}_er{}_ru{}_au{}_{}_{}_{}_{}".format(
        *circuit_parameters[1:len(circuit_parameters) - 2])
    dir_path = os.path.join(root_dir, top_dir, bot_dir)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(dir_path + '/density_matrices.pickle', 'wb') as handle:
        pickle.dump(density_matrix_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Finished with Density Matrices")
    return


def get_directory_path(circuit_parameters):
    root_dir = os.getcwd()
    top_dir = "{}d".format(circuit_parameters[0])
    bot_dir = "confirmation_data_qb{}_tm{}_isi{}_nc{}_er{}_ru{}_au{}_{}_{}_{}_{}_fc{}".format(
        *circuit_parameters[1:len(circuit_parameters) - 2])
    dir_path = os.path.join(root_dir, top_dir, bot_dir)
    return dir_path


def clustering(mi_matrix):
    num_qubits = mi_matrix.shape[0]
    mi_2 = np.linalg.matrix_power(mi_matrix, 2)
    mi_3 = np.linalg.matrix_power(mi_matrix, 3)
    numerator = np.trace(mi_3)
    denominator = 0
    for j in range(0, num_qubits):
        for k in range(0, num_qubits):
            if not j == k:
                denominator += mi_2[j, k]
    if denominator == 0:
        return float("NaN")
    else:
        return float(numerator) / float(denominator)


def disparity(mi_matrix):
    num_qubits = mi_matrix.shape[0]
    summand_numerator = np.sum(np.square(mi_matrix), axis=0)  # Axis shouldn't matter because MI is symmetric
    summand_denominator = np.square(np.sum(mi_matrix, axis=0))
    summand = np.divide(summand_numerator, summand_denominator)
    return float(np.sum(summand)) / float(num_qubits)


def construct_zz_correlation_matrix(qubits, results_dict, ising_mitigation, invariant, readout_correction):
    zz_matrix = np.zeros((len(qubits), len(qubits)))

    z_counts = results_dict["All_" + "Z"]
    if readout_correction == "RC_True":
        z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
    if ising_mitigation == "PS_True":
        z_counts = post_selection(counts_dict=z_counts, qubits=qubits, invariant=invariant)

    for i in range(0, len(qubits) - 1):
        for j in range(i + 1, len(qubits)):
            bin_fmt = '{0:0' + str(len(qubits)) + 'b}'
            numerator = 0
            denominator = 0
            for k, v in z_counts.items():
                # print(k, v)
                bit_string = bin_fmt.format(int(k))
                numerator += int(v) * (-1) ** (int(bit_string[i]) + int(bit_string[j]))
                denominator += int(v)
            zz_term = float(numerator) / float(denominator)
            zz_matrix[i, j] = zz_term - (local_magnetization(i, z_counts, qubits) * local_magnetization(
                j, z_counts, qubits))
            zz_matrix[j, i] = zz_matrix[i, j]
    return zz_matrix


def construct_classical_mutual_information_matrix(qubits, results_dict, ising_mitigation, invariant,
                                                  readout_correction, observables):
    # Originally intended to compute classical mutual information on just z-basis measurements
    mi_matrix = np.zeros((len(qubits), len(qubits)), dtype=np.float64)

    if observables == "only_z":
        z_counts = results_dict["All_" + "Z"]
    elif observables == "only_x":
        z_counts = results_dict["All_" + "X"]

    if readout_correction == "RC_True":
        z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
    if ising_mitigation == "PS_True" and observables == "only_z":
        z_counts = post_selection(counts_dict=z_counts, qubits=qubits, invariant=invariant)

    bin_fmt = '{0:0' + str(len(qubits)) + 'b}'
    for i in range(0, len(qubits) - 1):
        for j in range(i + 1, len(qubits)):
            joint_distribution = np.zeros((2, 2), dtype=np.float64)
            pi_distribution = np.zeros(2, dtype=np.float64)
            pj_distribution = np.zeros(2, dtype=np.float64)
            denominator = 0
            for k, v in z_counts.items():
                bit_string = bin_fmt.format(int(k))
                joint_distribution[int(bit_string[i]), int(bit_string[j])] += float(v)
                pi_distribution[int(bit_string[i])] += float(v)
                pj_distribution[int(bit_string[j])] += float(v)
                denominator += float(v)
            joint_distribution /= float(denominator)
            pi_distribution = np.sum(joint_distribution, axis=1)
            pj_distribution = np.sum(joint_distribution, axis=0)
            # pi_distribution /= float(denominator)  # Two lines above and these two lines now agree.
            # pj_distribution /= float(denominator)  # Think we're in business
            element = 0
            for ii in range(0, 2):
                for jj in range(0, 2):
                    if not joint_distribution[ii, jj] == 0:
                        element += (joint_distribution[ii, jj] * math.log(joint_distribution[ii, jj] /
                                                                          pi_distribution[ii] / pj_distribution[jj], 2))
            mi_matrix[i, j] = element
            mi_matrix[j, i] = element
    # mi_matrix /= np.amax(mi_matrix)
    #  print(mi_matrix)
    return mi_matrix  # May need to normalize!


def construct_mutual_information_matrix(qubits, results_dict):
    mi_matrix = np.zeros((len(qubits), len(qubits)))
    for i in range(0, len(qubits) - 1):
        for j in range(i + 1, len(qubits)):
            mi_matrix[i, j] = mutual_information(site_1_index=i, site_2_index=j, results_dict=results_dict,
                                                 qubits=qubits)
            mi_matrix[j, i] = mi_matrix[i, j]
    return mi_matrix


def rescale(time_step: int):
    a = 0.0  # 0.025
    b = 1.0
    # return b + a * time_step
    if time_step < 30:
        return 1. / (b - a * time_step)
    else:
        return 1.


def population_plots(dim, size, t_max, initial_site_indices, no_counts, experimental_repetitions,
                     rule, activation_unitaries, sim_mode, two_qubit_gate,
                     processor, observables, ising_mitigation, invariant, readout_correction,
                     floquet_calibration):
    # Allow for an ad hoc rescaling factor in (average?) magnetizations
    scale_factor = 1.
    # Initialize array
    avg_magnetizations = np.zeros((size, t_max))
    retained_fractions = np.ones((t_max, experimental_repetitions))

    figs, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, squeeze=False)
    indices = [(0, 0), (0, 1), (1, 0), (1, 1)]  # , (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]
    for r in range(0, experimental_repetitions):
        # if r == 2 or r==5:
        #     continue
        qubits = which_chain(r, size)
        magnetizations = np.zeros((size, t_max))
        for t in range(0, t_max):
            circuit_parameters = (dim, len(qubits), t_max, initial_site_indices, no_counts,
                                  experimental_repetitions, rule, activation_unitaries, sim_mode, two_qubit_gate,
                                  processor, observables, floquet_calibration, r, t)
            results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
            z_counts = results_dict["All_Z"]
            if readout_correction == "RC_True":
                z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
            if ising_mitigation == "PS_True":
                z_counts = post_selection(counts_dict=z_counts, qubits=qubits, invariant=invariant)
            retained_counts = sum(z_counts.values())
            retained_fractions[t, r] = float(retained_counts) / float(no_counts)
            for j in range(0, len(qubits)):
                avg_magnetizations[j, t] += rescale(t) * 0.5 * (1 - local_magnetization(
                    site_index=j, counts_dict=z_counts,
                    qubits=qubits)) / float(
                    experimental_repetitions)
                magnetizations[j, t] = 0.5 * (1 - local_magnetization(site_index=j, counts_dict=z_counts,
                                                                      qubits=qubits))
        index_pair = indices[r]
        sns.heatmap(magnetizations, vmin=0., vmax=1., cmap="vlag_r", square=False, cbar=True,
                    ax=axs[index_pair[0], index_pair[1]],
                    cbar_kws={'label': r'Local population, $\langle n_i \rangle$'})
        axs[index_pair[0], index_pair[1]].set_xlabel(r"QCA cycle, $t$")
        axs[index_pair[0], index_pair[1]].set_ylabel(r"Qubit site, $i$")

    avg_retained_fractions = np.average(retained_fractions, axis=1)
    std_retained_fractions = np.std(retained_fractions, axis=1)

    dir_path = get_directory_path(circuit_parameters=circuit_parameters)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(dir_path + "/" "populations_" + readout_correction + ising_mitigation + ".pdf", dpi=600)
    plt.close(figs)

    fig, ax1 = plt.subplots()

    # sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(avg_magnetizations, vmin=0., vmax=1., cmap="vlag_r", square=False, cbar=True, ax=ax1,
                xticklabels=[i for i in range(0, 30, 2)], yticklabels=[i for i in range(1, size+1, 2)],
                cbar_kws={'label': r'Local population, $\langle n_i \rangle$'})
    ax1.set_xticks([i+.5 for i in range(0, 30, 2)])
    ax1.set_yticks([i-.5 for i in range(1, size+1, 2)])
    ax1.set_xlabel(r"QCA cycle, $t$")
    ax1.set_ylabel(r"Qubit site, $i$")

    # dir_path = get_directory_path(circuit_parameters=circuit_parameters)

    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)
    plt.tight_layout()
    plt.savefig(dir_path + "/avg_population_" + readout_correction + ising_mitigation + ".pdf", bbox_inches="tight",
                dpi=600)

    plt.close(fig)

    fig = plt.figure()
    plt.errorbar(np.array([i for i in range(0, t_max)]), avg_retained_fractions, yerr=std_retained_fractions,
                 marker=".", linestyle="", capsize=5, color="k")
    plt.xlabel(r"QCA Cycle ($t$)")
    plt.ylabel(r"Retained Count Fraction After PS")
    plt.yscale("log")
    plt.savefig(dir_path + "/retained_fraction_" + readout_correction + ising_mitigation + ".pdf")
    plt.close(fig)
    return


def magnetization_plots(dim, size, t_max, initial_site_indices, no_counts, experimental_repetitions,
                        rule, activation_unitaries, sim_mode, two_qubit_gate,
                        processor, observables, ising_mitigation, invariant, readout_correction,
                        floquet_calibration):
    # Allow for an ad hoc rescaling factor in (average?) magnetizations
    scale_factor = 1.
    # Initialize array
    avg_magnetizations = np.zeros((size, t_max))

    figs, axs = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, squeeze=False)
    indices = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]
    for r in range(0, experimental_repetitions):
        # if r == 2 or r==5:
        #     continue
        qubits = which_chain(r, size)
        magnetizations = np.zeros((size, t_max))
        for t in range(0, t_max):
            circuit_parameters = (dim, len(qubits), t_max, initial_site_indices, no_counts,
                                  experimental_repetitions, rule, activation_unitaries, sim_mode, two_qubit_gate,
                                  processor, observables, floquet_calibration, r, t)
            results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
            z_counts = results_dict["All_Z"]
            if readout_correction == "RC_True":
                z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
            if ising_mitigation == "PS_True":
                z_counts = post_selection(counts_dict=z_counts, qubits=qubits, invariant=invariant)
            for j in range(0, len(qubits)):
                avg_magnetizations[j, t] += rescale(t) * local_magnetization(site_index=j, counts_dict=z_counts,
                                                                             qubits=qubits) / float(
                    experimental_repetitions)
                magnetizations[j, t] = local_magnetization(site_index=j, counts_dict=z_counts,
                                                           qubits=qubits)
        index_pair = indices[r]
        sns.heatmap(magnetizations, vmin=-1., vmax=1., cmap="vlag", square=False, cbar=True,
                    ax=axs[index_pair[0], index_pair[1]])

    dir_path = get_directory_path(circuit_parameters=circuit_parameters)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(dir_path + "/" "magnetizations_" + readout_correction + ising_mitigation + ".pdf")
    plt.close(figs)

    fig, ax1 = plt.subplots()

    # sns.color_palette("vlag", as_cmap=True)
    sns.heatmap(avg_magnetizations, vmin=-1., vmax=1., cmap="vlag", square=False, cbar=True, ax=ax1,
                cbar_kws={'label': r'Local Magnetization ($\langle Z_i \rangle$)'})
    ax1.set_xlabel(r"Time ($t$)")
    ax1.set_ylabel(r"Qubit Site ($i$)")

    dir_path = get_directory_path(circuit_parameters=circuit_parameters)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(dir_path + "/" "avg_magnetization_" + readout_correction + ising_mitigation + ".pdf")

    plt.close(fig)
    return


def entropy_plot(dim, size, t_max, initial_site_indices, no_counts, experimental_repetitions,
                 rule, activation_unitaries, sim_mode, two_qubit_gate,
                 processor, observables):
    # Initialize array
    avg_entropies = np.zeros((size, t_max))
    for r in range(0, experimental_repetitions):
        qubits = which_chain(r, size)
        for t in range(0, t_max):
            circuit_parameters = (dim, len(qubits), t_max, initial_site_indices, no_counts,
                                  experimental_repetitions, rule, activation_unitaries, sim_mode, two_qubit_gate,
                                  processor, observables, r, t)
            results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
            counts_dicts_list = [results_dict["All_X"], results_dict["All_Y"], results_dict["All_Z"]]
            for j in range(0, len(qubits)):
                avg_entropies[j, t] += local_entropy(site_index=j, counts_dicts_list=counts_dicts_list, qubits=qubits,
                                                     order=2)[1] / float(experimental_repetitions)
    fig, ax1 = plt.subplots()

    sns.heatmap(avg_entropies, vmin=0., vmax=1., square=False, cbar=True, ax=ax1,
                cbar_kws={'label': r'Local Entropy ($S^{(2)}_i$)'})
    ax1.set_xlabel(r"Time ($t$)")
    ax1.set_ylabel(r"Qubit Site ($i$)")

    dir_path = get_directory_path(circuit_parameters=circuit_parameters)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(dir_path + "/" "entropy.pdf")
    plt.close(fig)
    return


def read_in_data_file(circuit_parameters):
    root_dir = os.getcwd()
    top_dir = "{}d".format(circuit_parameters[0])
    bot_dir = "data_qb{}_tm{}_isi{}_nc{}_er{}_ru{}_au{}_{}_{}_{}_{}_fc{}".format(*circuit_parameters[1:len(
        circuit_parameters) - 2])
    dir_path = os.path.join(root_dir, top_dir, bot_dir)

    with open(dir_path + '/rep{}_ts{}.json'.format(*circuit_parameters[-2:]), 'r') as f:
        results_dict = json.load(f)
    return results_dict


def apply_correction_matrix(counts_dict, qubits):
    num_states = 2 ** len(qubits)
    correction_matrix = load_npz('./11_qubit_correction_matrix.npz')
    # correction_matrix = inv(correction_matrix)
    # correction_matrix = csc_matrix.transpose(correction_matrix)
    counts_vector = np.array([counts_dict.get(str(x), 0) for x in range(0, num_states)])
    corrected_vector = csc_matrix.dot(correction_matrix, counts_vector)
    new_counts_dict = {}
    for i in range(0, len(corrected_vector)):
        if corrected_vector[i] != 0:
            new_counts_dict[i] = corrected_vector[i]
    return new_counts_dict


def post_selection(counts_dict, qubits, invariant):
    bin_fmt = '{0:0' + str(len(qubits)) + 'b}'
    new_counts_dict = {}
    print("Total Counts:", sum(counts_dict.values()))
    for k, v in counts_dict.items():
        bit_string = bin_fmt.format(int(k))
        new_string = '0' + bit_string + '0'
        invariant_counter = 0
        for i in range(0, len(new_string) - 1):
            if new_string[i] == new_string[i + 1]:
                invariant_counter += 1
            else:
                invariant_counter -= 1
        if invariant_counter == invariant:
            new_counts_dict[k] = v
    print("Total PS Counts:", sum(new_counts_dict.values()))
    return new_counts_dict


def local_magnetization(site_index, counts_dict, qubits):
    # counts_dict is a single Z basis histogram dictionary
    bin_fmt = '{0:0' + str(len(qubits)) + 'b}'
    zeros_counter = 0
    ones_counter = 0
    for k, v in counts_dict.items():
        bit_string = bin_fmt.format(int(k))
        if bit_string[site_index] == '0':
            zeros_counter += int(v)
        else:
            ones_counter += int(v)
    if float(zeros_counter + ones_counter) == 0:
        return float("NaN")
    else:
        return float(zeros_counter - ones_counter) / float(zeros_counter + ones_counter)  # Denominator is tot counts


def local_entropy(site_index, counts_dicts_list, qubits, order=2):
    # counts_dicts_list should contain X, Y, and Z basis histograms
    expectation_values = []  # Ordered X, Y, Z
    for counts_dict in counts_dicts_list:
        bin_fmt = '{0:0' + str(len(qubits)) + 'b}'
        zeros_counter = 0
        ones_counter = 0
        for k, v in counts_dict.items():
            bit_string = bin_fmt.format(int(k))
            if bit_string[site_index] == '0':
                zeros_counter += int(v)
            else:
                ones_counter += int(v)
        expectation_values.append(float(zeros_counter - ones_counter) / float(zeros_counter + ones_counter))
    density_matrix = 0.5 * np.array([[1 + expectation_values[2], expectation_values[0] - 1j * expectation_values[1]],
                                     [expectation_values[0] + 1j * expectation_values[1], 1 - expectation_values[2]]])
    density_matrix_power = np.linalg.matrix_power(density_matrix, order)
    purity = np.trace(density_matrix_power)
    return density_matrix, np.log2(purity).real / (1. - float(order))


def mutual_information(site_1_index, site_2_index, results_dict, qubits):
    counts_dicts_list = [results_dict["All_X"], results_dict["All_Y"], results_dict["All_Z"]]
    partitions_dict = construct_hash_partitions(qubits=qubits)
    if site_1_index == site_2_index:
        m_i = 0.
    else:
        m_i = local_entropy(site_index=site_1_index, counts_dicts_list=counts_dicts_list, qubits=qubits)[1]
        m_i += local_entropy(site_index=site_2_index, counts_dicts_list=counts_dicts_list, qubits=qubits)[1]
        m_i -= two_point_entropy(site_1_index=site_1_index, site_2_index=site_2_index, results_dict=results_dict,
                                 partitions_dict=partitions_dict, qubits=qubits)[1]
    # Yes, need to take absolute value before returning, i.e.
    m_i = np.abs(m_i)
    return m_i / 2.  # For normalization


def two_point_entropy(site_1_index, site_2_index, results_dict, partitions_dict, qubits, order=2):
    # May need to define get a helper function that calculates general two-point correlation functions
    # given an appropriate distribution

    # Also probably want to construct rho with Pauli tensor rep. algorithmically?

    #  Need to find which hash function contains sites in correct partitions
    for k, v in partitions_dict.items():
        if qubits[site_1_index] in v[0] and qubits[site_2_index] in v[1]:
            partition = k
        elif qubits[site_2_index] in v[0] and qubits[site_1_index] in v[1]:
            partition = k
        else:
            continue

    observables = ["I", "X", "Y", "Z"]
    pauli_matrices = [np.array([[1, 0], [0, 1]], dtype=np.complex_), np.array([[0, 1], [1, 0]], dtype=np.complex_),
                      np.array([[0, 0 - 1j], [0 + 1j, 0]], dtype=np.complex_), np.array([[1, 0], [0, -1]],
                                                                                        dtype=np.complex_)]
    density_matrix = np.zeros((4, 4), dtype=np.complex_)
    for i in range(0, len(observables)):
        for j in range(0, len(observables)):
            dm = 0.25 * correlation_function(site_1_obs=observables[i], site_2_obs=observables[j],
                                             site_1_index=site_1_index, site_2_index=site_2_index,
                                             results_dict=results_dict, partition=partition,
                                             qubits=qubits) \
                 * np.kron(pauli_matrices[i], pauli_matrices[j])
            density_matrix += dm
    density_matrix_power = np.linalg.matrix_power(density_matrix, order)
    purity = np.trace(density_matrix_power)
    return density_matrix, np.log2(purity).real / (1. - float(order))


def correlation_function(site_1_obs, site_2_obs, site_1_index, site_2_index, results_dict, partition, qubits):
    if site_1_obs == "I" and site_2_obs == "I":  # "II" factor
        return 1

    elif site_1_obs == "I" and not site_2_obs == "I":  # Site_2 single-body
        counts_dict = results_dict["All_" + site_2_obs]
        bin_fmt = '{0:0' + str(len(qubits)) + 'b}'
        zeros_counter = 0
        ones_counter = 0
        for k, v in counts_dict.items():
            bit_string = bin_fmt.format(int(k))
            if bit_string[site_2_index] == '0':
                zeros_counter += int(v)
            else:
                ones_counter += int(v)
        return float(zeros_counter - ones_counter) / float(
            zeros_counter + ones_counter)  # Denominator should be tot counts

    elif site_2_obs == "I" and not site_1_obs == "I":  # Site_1 single_body
        counts_dict = results_dict["All_" + site_1_obs]
        bin_fmt = '{0:0' + str(len(qubits)) + 'b}'
        zeros_counter = 0
        ones_counter = 0
        for k, v in counts_dict.items():
            bit_string = bin_fmt.format(int(k))
            if bit_string[site_1_index] == '0':
                zeros_counter += int(v)
            else:
                ones_counter += int(v)
        return float(zeros_counter - ones_counter) / float(
            zeros_counter + ones_counter)  # Denominator should be tot counts

    elif site_1_obs == site_2_obs and not site_1_obs == "I":  # Diagonal two-body
        counts_dict = results_dict["All_" + site_1_obs]
        bin_fmt = '{0:0' + str(len(qubits)) + 'b}'
        numerator = 0
        denominator = 0
        for k, v in counts_dict.items():
            bit_string = bin_fmt.format(int(k))
            numerator += int(v) * (-1) ** (int(bit_string[site_1_index]) + int(bit_string[site_2_index]))
            denominator += int(v)
        return float(numerator) / float(denominator)

    else:  # All other off-diagonal two-body
        counts_dict = results_dict[partition + "_" + site_1_obs + site_2_obs]
        bin_fmt = '{0:0' + str(len(qubits)) + 'b}'
        numerator = 0
        denominator = 0
        for k, v in counts_dict.items():
            bit_string = bin_fmt.format(int(k))
            numerator += int(v) * (-1) ** (int(bit_string[site_1_index]) + int(bit_string[site_2_index]))
            denominator += int(v)
        return float(numerator) / float(denominator)


def construct_hash_partitions(qubits):
    n = len(qubits)  # Number of qubits
    q = int(np.ceil(np.log2(n)))  # Number of perfect has functions
    bin_fmt = '{0:0' + str(q) + 'b}'  # Correct binary format

    partitions_dict = {}
    for i in range(0, q):  # i-1 from paper
        red_qubits = []
        blue_qubits = []
        for j in range(0, n):  # This is j-1 in paper
            bit_string = bin_fmt.format(int(j))
            if bit_string[i] == '0':
                red_qubits.append(qubits[j])
            else:
                blue_qubits.append(qubits[j])
        partitions_dict.update({"f" + str(i): [red_qubits, blue_qubits]})
    return partitions_dict


if __name__ == "__main__":
    main()
