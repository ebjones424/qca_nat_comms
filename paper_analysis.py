import cirq
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.sparse import csc_matrix, load_npz
import math
import pickle
from scipy.optimize import curve_fit


def main():
    plt.rc('font', size=16)
    # Parameters #
    # Dimensionality of the QCA
    d = 1

    # Max time steps
    t_max = 30

    # Number of QCA sites.
    # sizes = [11, 13, 15]
    # sizes = [19, 21, 23]
    # sizes = [15, 17, 19]
    # sizes = [i for i in range(5, 25, 2)]
    sizes = [17]  # For population vignettes

    no_counts = 100_000  # Number of histogram repetitions per measurement.
    exp_reps = 4  # Number of experimental repetitions
    activation_unitaries = ["H"]
    rule = "T6"

    sim_modes = ['cirq', 'engine']  # 'engine' or 'cirq'
    two_qubit_gate = "parasitic_root_iSWAP"  # 'CZ' or 'Sycamore', 'root_iSWAP' or 'parasitic_root_iSWAP'
    processor = 'weber'  # 'rainbow' or 'weber' or something else ('NA' for 'cirq' sim_mode)
    observables = 'only_z'  # Or 'only_z' or 'mutual_information' or 'only_x'
    # observables = ['only_x', 'only_z']  # For xx_zz comparison
    ising_mitigations = ["PS_False", "PS_True"]
    readout_correction = "RC_False"
    floquet_calibrations = [False, True]  # Or False
    # which_plot = "initial_condition_variation"
    which_plot = "initial_condition_variation"

    if which_plot == "finite_scaling_analysis":
        plot_finite_scaling_analysis(d, sizes, t_max, no_counts, exp_reps,
                                     rule, activation_unitaries, sim_modes, two_qubit_gate,
                                     processor, observables, ising_mitigations, readout_correction,
                                     floquet_calibrations)

    elif which_plot == "relative_complexity_error":
        plot_relative_complexity_error(d, sizes, t_max, no_counts, exp_reps,
                                       rule, activation_unitaries, sim_modes, two_qubit_gate,
                                       processor, observables, ising_mitigations, readout_correction,
                                       floquet_calibrations)

    elif which_plot == "relative_complexity_error_xx_zz":
        plot_relative_complexity_error_xx_zz(d, sizes, t_max, no_counts, exp_reps,
                                             rule, activation_unitaries, sim_modes, two_qubit_gate,
                                             processor, observables, ising_mitigations, readout_correction,
                                             floquet_calibrations)

    elif which_plot == "initial_condition_variation":
        plot_initial_condition_variation(d, sizes, t_max, no_counts, exp_reps,
                                         rule, activation_unitaries, sim_modes, two_qubit_gate,
                                         processor, observables, ising_mitigations, readout_correction,
                                         floquet_calibrations)

    elif which_plot == "population_vignettes":
        plot_population_vignettes(dim=d, sizes=sizes, t_max=t_max,
                                  no_counts=no_counts, experimental_repetitions=exp_reps, rule=rule,
                                  activation_unitaries=activation_unitaries, sim_modes=sim_modes,
                                  two_qubit_gate=two_qubit_gate, processor=processor, observables=observables,
                                  ising_mitigations=ising_mitigations, readout_correction=readout_correction,
                                  floquet_calibrations=floquet_calibrations)

    elif which_plot == "population_vignettes_long":
        plot_population_vignettes_long(dim=d, sizes=sizes, t_max=t_max,
                                       no_counts=no_counts, experimental_repetitions=exp_reps, rule=rule,
                                       activation_unitaries=activation_unitaries, sim_modes=sim_modes,
                                       two_qubit_gate=two_qubit_gate, processor=processor, observables=observables,
                                       ising_mitigations=ising_mitigations, readout_correction=readout_correction,
                                       floquet_calibrations=floquet_calibrations)

    elif which_plot == "population_error":
        plot_population_error(dim=d, sizes=sizes, t_max=t_max,
                              no_counts=no_counts, experimental_repetitions=exp_reps, rule=rule,
                              activation_unitaries=activation_unitaries, sim_modes=sim_modes,
                              two_qubit_gate=two_qubit_gate, processor=processor, observables=observables,
                              ising_mitigations=ising_mitigations, readout_correction=readout_correction,
                              floquet_calibrations=floquet_calibrations)

    elif which_plot == "retained_fractions":
        plot_retained_fractions(dim=d, sizes=sizes, t_max=t_max,
                                no_counts=no_counts, experimental_repetitions=exp_reps, rule=rule,
                                activation_unitaries=activation_unitaries, sim_modes=sim_modes,
                                two_qubit_gate=two_qubit_gate, processor=processor, observables=observables,
                                ising_mitigations=ising_mitigations, readout_correction=readout_correction,
                                floquet_calibrations=floquet_calibrations)

    elif which_plot == "hilbert_bound_saturation":
        hilbert_bound_saturation(dim=d, sizes=sizes, t_max=t_max,
                                 no_counts=no_counts, experimental_repetitions=exp_reps, rule=rule,
                                 activation_unitaries=activation_unitaries, sim_modes=sim_modes,
                                 two_qubit_gate=two_qubit_gate, processor=processor, observables=observables,
                                 ising_mitigations=ising_mitigations, readout_correction=readout_correction,
                                 floquet_calibrations=floquet_calibrations)

    elif which_plot == "output_mutual_information":
        output_mutual_information(dim=d, sizes=sizes, t_max=t_max,
                                  no_counts=no_counts, experimental_repetitions=exp_reps, rule=rule,
                                  activation_unitaries=activation_unitaries, sim_modes=sim_modes,
                                  two_qubit_gate=two_qubit_gate, processor=processor, observables=observables,
                                  ising_mitigations=ising_mitigations, readout_correction=readout_correction,
                                  floquet_calibrations=floquet_calibrations)
    return


def plot_population_error(dim, sizes, t_max, no_counts, experimental_repetitions,
                          rule, activation_unitaries, sim_modes, two_qubit_gate,
                          processor, observables, ising_mitigations, readout_correction,
                          floquet_calibrations):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 4.8))
    # times = np.array([i for i in range(0, t_max//2+1)], dtype=np.int)
    times = np.array([i for i in range(0, t_max)], dtype=np.int)
    for ii, size in enumerate(sizes):
        initial_conditions = [[int(size - 1) // int(2)]]
        for i in range(0, len(initial_conditions)):
            initial_site_indices = initial_conditions[i]
            invariant = calculate_z_invariant(size=size, initial_site_indices=initial_site_indices)
            qubits = [cirq.LineQubit(i) for i in range(0, size)]
            actual_exp_reps = experimental_repetitions

            absolute_errors = []
            absolute_errors_error = []
            absolute_errors_raw = []
            absolute_errors_error_raw = []
            # for t in range(0, t_max//2+1):
            for t in range(0, t_max):
                raw_magnetizations = np.zeros((size, actual_exp_reps))
                exp_magnetizations = np.zeros((size, actual_exp_reps))
                theory_magnetizations = np.zeros((size, actual_exp_reps))
                for r in range(0, actual_exp_reps):
                    circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                          experimental_repetitions, rule, activation_unitaries, "engine",
                                          two_qubit_gate,
                                          processor, observables, True, r, t)
                    theory_circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                                 experimental_repetitions, rule, activation_unitaries, "cirq",
                                                 two_qubit_gate,
                                                 processor, observables, False, r, t)

                    results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
                    z_counts_raw = results_dict["All_Z"]
                    z_counts = results_dict["All_Z"]
                    z_counts = post_selection(counts_dict=z_counts, qubits=qubits, invariant=invariant)

                    theory_results_dict = read_in_data_file(circuit_parameters=theory_circuit_parameters)
                    theory_z_counts = theory_results_dict["All_Z"]

                    for j in range(0, size):
                        raw_magnetizations[j, r] += 0.5 * (1 - local_magnetization(site_index=j,
                                                                                   counts_dict=z_counts_raw,
                                                                                   qubits=qubits))
                        exp_magnetizations[j, r] += 0.5 * (1 - local_magnetization(site_index=j,
                                                                                   counts_dict=z_counts,
                                                                                   qubits=qubits))
                        theory_magnetizations[j, r] += 0.5 * (1 - local_magnetization(site_index=j,
                                                                                      counts_dict=theory_z_counts,
                                                                                      qubits=qubits))
                avg_raw_magnetizations = np.average(raw_magnetizations, axis=1)
                std_raw_magnetizations = np.std(raw_magnetizations, axis=1)

                avg_magnetizations = np.average(exp_magnetizations, axis=1)
                std_magnetizations = np.std(exp_magnetizations, axis=1)

                avg_theory_magnetizations = np.average(theory_magnetizations, axis=1)

                # Now I need to calculate site-averaged relative error.
                # raw_error = np.average(np.divide(np.absolute(np.subtract(avg_theory_magnetizations, avg_raw_magnetizations)), np.add(avg_theory_magnetizations, avg_raw_magnetizations)/2.))
                # error = np.average(np.divide(np.absolute(np.subtract(avg_theory_magnetizations, avg_magnetizations)), np.add(avg_theory_magnetizations, avg_raw_magnetizations)/2.))
                # avg_theory_magnetizations))

                # Now I need to calculate site-averaged weird fidelity thing.
                raw_error = np.average(
                    np.divide(np.absolute(np.subtract(avg_theory_magnetizations, avg_raw_magnetizations)), 1.))
                error = np.average(
                    np.divide(np.absolute(np.subtract(avg_theory_magnetizations, avg_magnetizations)), 1.))

                # Now I need to calculate site-averaged weird fidelity thing.
                # raw_error = np.average(np.divide(np.absolute(np.subtract(0.5, avg_raw_magnetizations)), np.absolute(np.subtract(avg_theory_magnetizations, 0.5))))
                # error = np.average(np.divide(np.absolute(np.subtract(0.5, avg_magnetizations)), np.absolute(np.subtract(avg_theory_magnetizations, 0.5))))

                raw_error_error = np.average(std_raw_magnetizations)
                error_error = np.average(std_magnetizations)
                absolute_errors.append(error)
                absolute_errors_error.append(error_error)
                absolute_errors_raw.append(raw_error)
                absolute_errors_error_raw.append(raw_error_error)
            absolute_errors = np.array(absolute_errors)
            absolute_errors_error = np.array(absolute_errors_error)
            absolute_errors_raw = np.array(absolute_errors_raw)
            absolute_errors_error_raw = np.array(absolute_errors_error_raw)
            # plt.plot(times, absolute_errors, marker=".", label=r"$L=%s$" % size)
            axs[1].errorbar(times, absolute_errors, xerr=None, yerr=absolute_errors_error,
                            marker=".", capsize=5,
                            label=r"$L=%s$" % size, color=colors[ii])
            axs[0].errorbar(times, absolute_errors_raw, xerr=None, yerr=absolute_errors_error_raw,
                            marker=".", capsize=5, color=colors[ii])
    #                         label=r"$L=%s$" % size)
    # plt.axhline(y=.2, color='k', linestyle="dashed")
    axs[0].axhline(y=0.18, color="k", linestyle="dashed", label="0.18")
    axs[1].axhline(y=0.18, color="k", linestyle="dashed")
    # axs[0].axhline(y=0.4, color="k", linestyle="dotted", label="0.4")
    # axs[1].axhline(y=0.4, color="k", linestyle="dotted")
    axs[0].text(0, 0.35, "Raw")
    axs[1].text(0, 0.35, "Post-Selected")
    # axs[0].set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    # axs[1].set_xticks([0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    axs[0].set_xticks([0, 4, 8, 12, 16, 20, 24, 28])
    axs[1].set_xticks([0, 4, 8, 12, 16, 20, 24, 28])
    axs[0].set_xlabel(r"QCA Cycle ($t$)")
    axs[1].set_xlabel(r"QCA Cycle ($t$)")
    axs[0].set_ylabel(r"$\overline{\epsilon(\langle n_i \rangle_{th.}, \langle n_i \rangle_{exp.})}$")
    # plt.yscale("log")
    fig.legend(loc="upper center", ncol=6, columnspacing=.5, borderaxespad=0., labelspacing=0.25,
               borderpad=0.2,
               handletextpad=0.2)
    plt.tight_layout()
    plt.savefig("./paper_analysis/population_error.pdf", bbox_inches="tight", dpi=600)
    plt.close(fig)
    return


def plot_finite_scaling_analysis(dim, sizes, t_max, no_counts, experimental_repetitions,
                                 rule, activation_unitaries, sim_modes, two_qubit_gate,
                                 processor, observables, ising_mitigations, readout_correction,
                                 floquet_calibrations):
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, squeeze=False)

    for sim_mode in sim_modes:
        for ising_mitigation in ising_mitigations:
            if sim_mode == "cirq" and ising_mitigation == "PS_True":
                continue
            for floquet_calibration in floquet_calibrations:
                if sim_mode == "cirq" and floquet_calibration:
                    continue

                size_xaxis = []
                clustering_yaxis = []
                disparity_yaxis = []
                for ii, size in enumerate(sizes):
                    initial_conditions = [[int(size - 1) // int(2)]]
                    for i in range(0, len(initial_conditions)):
                        initial_site_indices = initial_conditions[i]
                        qubits = [cirq.LineQubit(i) for i in range(0, size)]
                        actual_exp_reps = experimental_repetitions

                        clusterings = []
                        disparities = []
                        for r in range(0, actual_exp_reps):
                            circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                                  experimental_repetitions, rule, activation_unitaries, sim_mode,
                                                  two_qubit_gate,
                                                  processor, observables, floquet_calibration, r, 0)
                            time, clustering, disparity = read_in_confirmation_data_file(
                                circuit_parameters=circuit_parameters,
                                readout_correction=readout_correction,
                                ising_mitigation=ising_mitigation,
                                which=r
                            )
                            size_xaxis.append(size)
                            clustering_yaxis.append(np.nanmean(clustering))
                            disparity_yaxis.append(np.nanmean(np.absolute(np.gradient(disparity))))
                            # disparity_yaxis.append(np.nanmean(disparity))
                            # print(np.gradient(disparity))

                            clusterings.append(clustering)
                            disparities.append(disparity)

                        clusterings = np.array(clusterings)
                        disparities = np.array(disparities)
                        time_avg_clusterings = np.average(clusterings, axis=1)
                        # std_clustering = np.std(clusterings, axis=0)
                        time_avg_disparities = np.average(disparities, axis=1)
                        # std_disparity = np.std(disparities, axis=0)

                size_xaxis = np.array(size_xaxis)
                print(size_xaxis.shape, size_xaxis)
                clustering_yaxis = np.array(clustering_yaxis)
                print(clustering_yaxis.shape, clustering_yaxis)
                disparity_yaxis = np.array(disparity_yaxis)
                print(disparity_yaxis.shape, disparity_yaxis)
                axs[0, 0].scatter(size_xaxis, clustering_yaxis, alpha=0.5)
                axs[0, 1].scatter(size_xaxis, disparity_yaxis, alpha=0.5)

    axs[0, 0].set_xlabel(r"System Size ($L$)")
    axs[0, 1].set_xlabel(r"System Size ($L$)")
    axs[0, 0].set_ylabel("Avg. Clustering")
    axs[0, 1].set_ylabel("Avg. Disparity")

    # axs[0, 0].set_yscale("log")
    # axs[0, 1].set_yscale("log")

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return


def plot_initial_condition_variation(dim, sizes, t_max, no_counts, experimental_repetitions,
                                     rule, activation_unitaries, sim_modes, two_qubit_gate,
                                     processor, observables, ising_mitigations, readout_correction,
                                     floquet_calibrations):
    # This function is intended to plot both cluastering and disparity vignettes and relative errors.
    data_dict = np.load('./QL-3_C-Y.npy', allow_pickle=True).item()
    pt_clusterings = data_dict["Cs_PT"][-3:]
    iur_clusterings = data_dict["Cs_inc"][-3:]
    ps_pt_clusterings = data_dict["Cs_PT-PS"][-3:]
    ps_iur_clusterings = data_dict["Cs_inc-PS"][-3:]
    # ps_iur_clusterings = [0.067, 0.066, 0.078, 0.09, 0.099, 0.108, 0.114]
    iur_clusterings = [0., 0., 0.]
    ps_iur_clusterings = [0.07980962, 0.00687779, 0.00250996]

    initial_conditions = [[5, 11], [3, 8, 13], [2, 6, 11, 14]]
    indices = [(0, i) for i in range(0, len(initial_conditions))]
    # indices = [(0, 0), (0, 1), (0, 2), (0, 3)]

    figs_c, axs_c = plt.subplots(nrows=1, ncols=len(indices), sharex=True, sharey=True, squeeze=False,
                                 figsize=(6 * len(indices), 4.8))
    figs_d, axs_d = plt.subplots(nrows=1, ncols=len(indices), sharex=True, sharey=True, squeeze=False)

    for ii, size in enumerate(sizes):
        # initial_conditions = [[int(size - 1) // int(2)]]
        for i in range(0, len(initial_conditions)):
            initial_site_indices = initial_conditions[i]
            qubits = [cirq.LineQubit(i) for i in range(0, size)]
            actual_exp_reps = experimental_repetitions
            for sim_mode in sim_modes:
                for ising_mitigation in ising_mitigations:
                    if sim_mode == "cirq" and ising_mitigation == "PS_True":
                        continue
                    for floquet_calibration in floquet_calibrations:
                        if sim_mode == "cirq" and floquet_calibration:
                            continue
                        elif not floquet_calibration and sim_mode == "engine":
                            continue
                        # clusterings = np.zeros((t_max, actual_exp_reps))
                        # disparities = np.zeros((t_max, actual_exp_reps))
                        clusterings = []
                        disparities = []
                        for r in range(0, actual_exp_reps):
                            if initial_site_indices == [8]:
                                processor = "weber"
                            else:
                                processor = "rainbow"
                            circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                                  experimental_repetitions, rule, activation_unitaries, sim_mode,
                                                  two_qubit_gate,
                                                  processor, observables, floquet_calibration, r, 0)
                            time, clustering, disparity = read_in_confirmation_data_file(
                                circuit_parameters=circuit_parameters,
                                readout_correction=readout_correction,
                                ising_mitigation=ising_mitigation,
                                which=r
                            )
                            clusterings.append(clustering)
                            disparities.append(disparity)
                            # disparities.append(np.gradient(disparity))  # May need to remove gradient.
                        clusterings = np.array(clusterings)
                        disparities = np.array(disparities)
                        avg_clustering = np.nanmean(clusterings, axis=0)
                        std_clustering = np.nanstd(clusterings, axis=0)
                        avg_disparity = np.nanmean(disparities, axis=0)
                        std_disparity = np.nanstd(disparities, axis=0)
                        index_pair = indices[i]

                        if sim_mode == "cirq":
                            if i == 0:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None, yerr=std_clustering,
                                                           marker=".", linestyle="solid", capsize=5,
                                                           label="Emulated", color="tab:blue")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None, yerr=std_disparity,
                                                           marker=".", linestyle="solid", capsize=5,
                                                           label="Emulated", color="tab:blue")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                            else:
                                axs_c[index_pair].plot(time, avg_clustering, marker=".", color="tab:blue")
                                axs_d[index_pair].plot(time, avg_disparity, marker=".", color="tab:blue")
                        elif sim_mode == "engine" and (ising_mitigation == "PS_True" and floquet_calibration):
                            # print(std_magnetizations)
                            if i == 0:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None, yerr=std_clustering,
                                                           marker=".", linestyle="", capsize=5,
                                                           label="Post-selected", color="tab:green")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None, yerr=std_disparity,
                                                           marker=".", linestyle="", capsize=5,
                                                           label="Post-selected", color="tab:green")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                            else:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None,
                                                           yerr=std_clustering,
                                                           capsize=5, marker=".", linestyle="", color="tab:green")
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None,
                                                           yerr=std_disparity,
                                                           capsize=5, marker=".", linestyle="", color="tab:green")
                        # elif sim_mode == "engine" and (ising_mitigation == "PS_True" and not floquet_calibration):
                        #    if ii == 0:
                        #        axs_c[index_pair].plot(time, avg_clustering, marker=".", linestyle="",
                        #                               label="PS")
                        #        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                        #        #       str(floquet_calibration))
                        #        axs_d[index_pair].plot(time, avg_disparity, marker=".", linestyle="",
                        #                               label="PS")
                        #        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                        #        #       str(floquet_calibration))
                        #    else:
                        #        axs_c[index_pair].plot(time, avg_clustering, marker=".", linestyle="")
                        #        axs_d[index_pair].plot(time, avg_disparity, marker=".", linestyle="")
                        elif sim_mode == "engine" and (ising_mitigation == "PS_False" and floquet_calibration):
                            if i == 0:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None,
                                                           yerr=std_clustering, capsize=5, marker=".", linestyle="",
                                                           label="Raw", color="tab:red")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None,
                                                           yerr=std_disparity, capsize=5, marker=".", linestyle="",
                                                           label="Raw", color="tab:red")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                            else:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None,
                                                           yerr=std_clustering, capsize=5, marker=".", linestyle="",
                                                           color="tab:red")
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None,
                                                           yerr=std_disparity, capsize=5, marker=".", linestyle="",
                                                           color="tab:red")
                            # else:
                            #    if ii == 0:
                            #        axs_c[index_pair].plot(time, avg_clustering, marker=".", linestyle="",
                            #                               label="Raw")
                            #        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                            #        #       str(floquet_calibration))
                            #        axs_d[index_pair].plot(time, avg_disparity, marker=".", linestyle="",
                            #                               label="Raw")
                            #        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                            #        #       str(floquet_calibration))
                            #    else:
                            #        axs_c[index_pair].plot(time, avg_clustering, marker=".", linestyle="")
                            #        axs_d[index_pair].plot(time, avg_disparity, marker=".", linestyle="")

                            if size <= 11:
                                axs_c[index_pair].set_xticks([i for i in range(0, len(time), 6)])
                                axs_d[index_pair].set_xticks([i for i in range(0, len(time), 6)])
                            else:
                                axs_c[index_pair].set_xticks([i for i in range(0, len(time), 4)])
                                axs_d[index_pair].set_xticks([i for i in range(0, len(time), 6)])
                            if index_pair[0] == 0:
                                axs_c[index_pair].set_xlabel(r"QCA cycle, $t$")
                                axs_d[index_pair].set_xlabel(r"QCA Cycle, $t$")
                            if index_pair[1] == 0:
                                axs_c[index_pair].set_ylabel(r"Clustering, $\mathcal{C}$")  # ($\langle Z_i \rangle$)")
                                axs_d[index_pair].set_ylabel(r"Disparity")  # ($\langle Z_i \rangle$)")

            axs_c[index_pair].text(0, .32, r'$%s/%s$ Filling' % (len(initial_site_indices), size))
            axs_d[index_pair].text(0, .4, r'$%s/%s$ Filling' % (len(initial_site_indices), size))

            # axs_c[index_pair].set_yscale("log")

            if i == 2:
                axs_c[index_pair].axhline(y=iur_clusterings[i], color='k', linestyle='dotted',
                                          label='Incoherent uniform random')
                axs_c[index_pair].axhline(y=ps_iur_clusterings[i], color='k', linestyle='dashed',
                                          label='Post-selected incoherent uniform random')
            else:
                axs_c[index_pair].axhline(y=iur_clusterings[i], color='k', linestyle='dotted')
                axs_c[index_pair].axhline(y=ps_iur_clusterings[i], color='k', linestyle='dashed')

        # else:
        #     axs_c[index_pair].text(0, 1., r'$L=%s$' % size)
        #     axs_d[index_pair].text(16, 0.7, r'$L=%s$' % size)
        # if i == 2:
        #     axs_c[index_pair].axhline(y=iur_clusterings[ii], color='tab:grey', linestyle='dotted',
        #                               label='IUR')
        #     axs_c[index_pair].axhline(y=ps_iur_clusterings[ii], color='tab:grey', linestyle='dashed',
        #                               label='IUR + PS')
        #     axs_c[index_pair].axhline(y=pt_clusterings[ii], color='k', linestyle='dotted',
        #                               label='PT')
        #     axs_c[index_pair].axhline(y=ps_pt_clusterings[ii], color='k', linestyle='dashed',
        #                               label='PT + PS')
        # else:
        #     axs_c[index_pair].axhline(y=iur_clusterings[ii], color='tab:grey', linestyle='dotted')
        #     axs_c[index_pair].axhline(y=ps_iur_clusterings[ii], color='tab:grey', linestyle='dashed')
        #     axs_c[index_pair].axhline(y=pt_clusterings[ii], color='k', linestyle='dotted')
        #     axs_c[index_pair].axhline(y=ps_pt_clusterings[ii], color='k', linestyle='dashed')

    axs_c[indices[0]].text(-8, .33, 'a', fontweight='bold')
    axs_c[indices[1]].text(-4, .33, 'b', fontweight='bold')
    axs_c[indices[2]].text(-4, .33, 'c', fontweight='bold')

    figs_c.legend(loc='upper center', ncol=7)
    figs_d.legend(loc='upper center', ncol=4)
    # figs_c.subplots_adjust(hspace=None, wspace=0.05)
    figs_c.savefig("./paper_analysis/clustering_ic_vignettes.pdf", bbox_inches="tight", dpi=600)
    figs_d.savefig("./paper_analysis/disparity_ic_vignettes.pdf")
    plt.close(figs_c)
    plt.close(figs_d)

    return


def plot_relative_complexity_error(dim, sizes, t_max, no_counts, experimental_repetitions,
                                   rule, activation_unitaries, sim_modes, two_qubit_gate,
                                   processor, observables, ising_mitigations, readout_correction,
                                   floquet_calibrations):
    # This function is intended to plot both cluastering and disparity vignettes and relative errors.
    data_dict = np.load('./QL-3_C-Y_shannon.npy', allow_pickle=True).item()
    # pt_clusterings = data_dict["Cs_PT"][-5:-2]
    # iur_clusterings = data_dict["Cs_inc"][-5:-2]
    if sizes[-1] == 23:
        iur_clusterings = data_dict["Cs_inc"][-int((23 - sizes[-1]) / 2) - 3:]
        ps_iur_clusterings = data_dict["Cs_inc-PS"][-int((23 - sizes[-1]) / 2) - 3:]
    else:
        iur_clusterings = data_dict["Cs_inc"][-int((23 - sizes[-1]) / 2) - 3:-int((23 - sizes[-1]) / 2)]
        # ps_pt_clusterings = data_dict["Cs_PT-PS"][-5:-2]
        ps_iur_clusterings = data_dict["Cs_inc-PS"][-int((23 - sizes[-1]) / 2) - 3:-int((23 - sizes[-1]) / 2)]
        # ps_iur_clusterings = [0.067, 0.066, 0.078, 0.09, 0.099, 0.108, 0.114]

    figs_c, axs_c = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, squeeze=False, figsize=(18, 4.8))
    figs_d, axs_d = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, squeeze=False)

    indices = [(0, 0), (0, 1), (0, 2)]

    for ii, size in enumerate(sizes):
        initial_conditions = [[int(size - 1) // int(2)]]
        for i in range(0, len(initial_conditions)):
            initial_site_indices = initial_conditions[i]
            qubits = [cirq.LineQubit(i) for i in range(0, size)]
            actual_exp_reps = experimental_repetitions
            for sim_mode in sim_modes:
                for ising_mitigation in ising_mitigations:
                    if sim_mode == "cirq" and ising_mitigation == "PS_True":
                        continue
                    for floquet_calibration in floquet_calibrations:
                        if sim_mode == "cirq" and floquet_calibration:
                            continue
                        elif not floquet_calibration and sim_mode == "engine":
                            continue
                        # clusterings = np.zeros((t_max, actual_exp_reps))
                        # disparities = np.zeros((t_max, actual_exp_reps))
                        clusterings = []
                        disparities = []
                        for r in range(0, actual_exp_reps):
                            circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                                  experimental_repetitions, rule, activation_unitaries, sim_mode,
                                                  two_qubit_gate,
                                                  processor, observables, floquet_calibration, r, 0)
                            time, clustering, disparity = read_in_confirmation_data_file(
                                circuit_parameters=circuit_parameters,
                                readout_correction=readout_correction,
                                ising_mitigation=ising_mitigation,
                                which=r
                            )
                            clusterings.append(clustering)
                            disparities.append(disparity)
                            # disparities.append(np.gradient(disparity))  # May need to remove gradient.
                        clusterings = np.array(clusterings)
                        disparities = np.array(disparities)
                        avg_clustering = np.nanmean(clusterings, axis=0)
                        std_clustering = np.nanstd(clusterings, axis=0)
                        avg_disparity = np.nanmean(disparities, axis=0)
                        std_disparity = np.nanstd(disparities, axis=0)
                        index_pair = indices[ii]

                        if sim_mode == "cirq":
                            if ii == 0:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None, yerr=std_clustering,
                                                           marker=".", linestyle="solid", capsize=5,
                                                           label="Emulated", color="tab:blue")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None, yerr=std_disparity,
                                                           marker=".", linestyle="solid", capsize=5,
                                                           label="Emulated", color="tab:blue")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                            else:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None, yerr=std_clustering,
                                                           marker=".", linestyle="solid", capsize=5,
                                                           color="tab:blue")
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None, yerr=std_disparity,
                                                           marker=".", linestyle="solid", capsize=5,
                                                           color="tab:blue")
                        elif sim_mode == "engine" and (ising_mitigation == "PS_True" and floquet_calibration):
                            # print(std_magnetizations)
                            if ii == 0:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None, yerr=std_clustering,
                                                           marker=".", linestyle="", capsize=5,
                                                           label="Post-selected", color="tab:green")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None, yerr=std_disparity,
                                                           marker=".", linestyle="", capsize=5,
                                                           label="Post-selected", color="tab:green")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                            else:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None,
                                                           yerr=std_clustering,
                                                           capsize=5, marker=".", linestyle="", color="tab:green")
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None,
                                                           yerr=std_disparity,
                                                           capsize=5, marker=".", linestyle="", color="tab:green")
                        # elif sim_mode == "engine" and (ising_mitigation == "PS_True" and not floquet_calibration):
                        #    if ii == 0:
                        #        axs_c[index_pair].plot(time, avg_clustering, marker=".", linestyle="",
                        #                               label="PS")
                        #        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                        #        #       str(floquet_calibration))
                        #        axs_d[index_pair].plot(time, avg_disparity, marker=".", linestyle="",
                        #                               label="PS")
                        #        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                        #        #       str(floquet_calibration))
                        #    else:
                        #        axs_c[index_pair].plot(time, avg_clustering, marker=".", linestyle="")
                        #        axs_d[index_pair].plot(time, avg_disparity, marker=".", linestyle="")
                        elif sim_mode == "engine" and (ising_mitigation == "PS_False" and floquet_calibration):
                            if ii == 0:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None, yerr=std_clustering,
                                                           marker=".", linestyle="", capsize=5,
                                                           label="Raw", color="tab:red")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None, yerr=std_clustering,
                                                           marker=".", linestyle="", capsize=5,
                                                           label="Raw", color="tab:red")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                            else:
                                axs_c[index_pair].errorbar(time, avg_clustering, xerr=None, yerr=std_clustering,
                                                           marker=".", linestyle="", capsize=5,
                                                           color="tab:red")
                                axs_d[index_pair].errorbar(time, avg_disparity, xerr=None, yerr=std_clustering,
                                                           marker=".", linestyle="", capsize=5,
                                                           color="tab:red")
                            # else:
                            #    if ii == 0:
                            #        axs_c[index_pair].plot(time, avg_clustering, marker=".", linestyle="",
                            #                               label="Raw")
                            #        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                            #        #       str(floquet_calibration))
                            #        axs_d[index_pair].plot(time, avg_disparity, marker=".", linestyle="",
                            #                               label="Raw")
                            #        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                            #        #       str(floquet_calibration))
                            #    else:
                            #        axs_c[index_pair].plot(time, avg_clustering, marker=".", linestyle="")
                            #        axs_d[index_pair].plot(time, avg_disparity, marker=".", linestyle="")

                            if size <= 11:
                                axs_c[index_pair].set_xticks([i for i in range(0, len(time), 6)])
                                axs_d[index_pair].set_xticks([i for i in range(0, len(time), 6)])
                            else:
                                axs_c[index_pair].set_xticks([i for i in range(0, len(time), 4)])
                                axs_d[index_pair].set_xticks([i for i in range(0, len(time), 6)])
                            if index_pair[0] == 0:
                                axs_c[index_pair].set_xlabel(r"QCA cycle, $t$")
                                axs_d[index_pair].set_xlabel(r"QCA cycle, $t$")
                            if index_pair[1] == 0:
                                axs_c[index_pair].set_ylabel(r"Clustering, $\mathcal{C}$")  # ($\langle Z_i \rangle$)")
                                axs_d[index_pair].set_ylabel(r"Disparity")  # ($\langle Z_i \rangle$)")
        if size < 11:
            axs_c[index_pair].text(0, .7, r'$L=%s$' % size)
            axs_d[index_pair].text(16, 0.0, r'$L=%s$' % size)
        else:
            axs_c[index_pair].text(0, .5, r'$L=%s$' % size)  # 1. If 23 included
            axs_d[index_pair].text(16, 0.7, r'$L=%s$' % size)
        if ii == 2:
            axs_c[index_pair].axhline(y=iur_clusterings[ii], color='k', linestyle='dotted',
                                      label='Incoherent uniform random')
            axs_c[index_pair].axhline(y=ps_iur_clusterings[ii], color='k', linestyle='dashed',
                                      label='Post-selected incoherent uniform random')
            # axs_c[index_pair].axhline(y=pt_clusterings[ii], color='k', linestyle='dotted',
            #                           label='PT')
            # axs_c[index_pair].axhline(y=ps_pt_clusterings[ii], color='k', linestyle='dashed',
            #                           label='PT + PS')
        else:
            axs_c[index_pair].axhline(y=iur_clusterings[ii], color='k', linestyle='dotted')
            axs_c[index_pair].axhline(y=ps_iur_clusterings[ii], color='k', linestyle='dashed')
            # axs_c[index_pair].axhline(y=pt_clusterings[ii], color='k', linestyle='dotted')
            # axs_c[index_pair].axhline(y=ps_pt_clusterings[ii], color='k', linestyle='dashed')

        # axs_c[index_pair].set_yscale("log")
        # axs_c[index_pair].set_ylim((1e-5, 1e1))

    axs_c[indices[0]].text(-7, .51, 'a', fontweight='bold')
    axs_c[indices[1]].text(-4, .51, 'b', fontweight='bold')
    axs_c[indices[2]].text(-4, .51, 'c', fontweight='bold')

    figs_c.legend(loc='upper center', ncol=7)
    figs_d.legend(loc='upper center', ncol=4)
    figs_c.savefig("./paper_analysis/clustering_vignettes_%s-%s.pdf" % (sizes[0], sizes[-1]), bbox_inches="tight",
                   dpi=600)
    figs_d.savefig("./paper_analysis/disparity_vignettes.pdf")
    plt.close(figs_c)
    plt.close(figs_d)

    return


def plot_relative_complexity_error_xx_zz(dim, sizes, t_max, no_counts, experimental_repetitions,
                                         rule, activation_unitaries, sim_modes, two_qubit_gate,
                                         processor, observables, ising_mitigations, readout_correction,
                                         floquet_calibrations):
    # This function is intended to plot both cluastering and disparity vignettes and relative errors.
    data_dict = np.load('./QL-3_C-Y_shannon.npy', allow_pickle=True).item()
    # pt_clusterings = data_dict["Cs_PT"][-5:-2]
    # iur_clusterings = data_dict["Cs_inc"][-5:-2]
    if sizes[-1] == 23:
        iur_clusterings = data_dict["Cs_inc"][-int((23 - sizes[-1]) / 2) - 3:]
        ps_iur_clusterings = data_dict["Cs_inc-PS"][-int((23 - sizes[-1]) / 2) - 3:]
    else:
        iur_clusterings = data_dict["Cs_inc"][-int((23 - sizes[-1]) / 2) - 3:-int((23 - sizes[-1]) / 2)]
        # ps_pt_clusterings = data_dict["Cs_PT-PS"][-5:-2]
        ps_iur_clusterings = data_dict["Cs_inc-PS"][-int((23 - sizes[-1]) / 2) - 3:-int((23 - sizes[-1]) / 2)]
        # ps_iur_clusterings = [0.067, 0.066, 0.078, 0.09, 0.099, 0.108, 0.114]

    figs_c, axs_c = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, squeeze=False, figsize=(18, 4.8))
    figs_d, axs_d = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, squeeze=False)

    indices = [(0, 0), (0, 1), (0, 2)]

    for ii, size in enumerate(sizes):
        initial_conditions = [[int(size - 1) // int(2)]]
        for i in range(0, len(initial_conditions)):
            initial_site_indices = initial_conditions[i]
            plot_labels_dict = {"only_x": "X-basis",
                                "only_z": "Z-basis"}
            colors = ["tab:blue", "tab:orange"]
            for ll, observable in enumerate(observables):
                qubits = [cirq.LineQubit(i) for i in range(0, size)]
                actual_exp_reps = experimental_repetitions
                sim_mode = "cirq"
                ising_mitigation = "PS_False"
                floquet_calibration = False

                clusterings = []
                for r in range(0, actual_exp_reps):
                    circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                          experimental_repetitions, rule, activation_unitaries, sim_mode,
                                          two_qubit_gate,
                                          processor, observable, floquet_calibration, r, 0)
                    time, clustering, disparity = read_in_confirmation_data_file(
                        circuit_parameters=circuit_parameters,
                        readout_correction=readout_correction,
                        ising_mitigation=ising_mitigation,
                        which=r
                    )
                    clusterings.append(clustering)
                clusterings = np.array(clusterings)
                avg_clustering = np.nanmean(clusterings, axis=0)
                std_clustering = np.nanstd(clusterings, axis=0)
                index_pair = indices[ii]

                if ii == 0:
                    axs_c[index_pair].errorbar(time, avg_clustering, xerr=None, yerr=std_clustering,
                                               marker=".", linestyle="solid", capsize=5,
                                               label=plot_labels_dict[observable], color=colors[ll])

                else:
                    axs_c[index_pair].errorbar(time, avg_clustering, xerr=None, yerr=std_clustering,
                                               marker=".", linestyle="solid", capsize=5,
                                               color=colors[ll])

                if size <= 11:
                    axs_c[index_pair].set_xticks([i for i in range(0, len(time), 6)])
                    axs_d[index_pair].set_xticks([i for i in range(0, len(time), 6)])
                else:
                    axs_c[index_pair].set_xticks([i for i in range(0, len(time), 4)])
                    axs_d[index_pair].set_xticks([i for i in range(0, len(time), 6)])
                if index_pair[0] == 0:
                    axs_c[index_pair].set_xlabel(r"QCA cycle, $t$")
                    axs_d[index_pair].set_xlabel(r"QCA cycle, $t$")
                if index_pair[1] == 0:
                    axs_c[index_pair].set_ylabel(r"Clustering, $\mathcal{C}$")  # ($\langle Z_i \rangle$)")
                    axs_d[index_pair].set_ylabel(r"Disparity")  # ($\langle Z_i \rangle$)")
        if size < 11:
            axs_c[index_pair].text(0, .7, r'$L=%s$' % size)
            axs_d[index_pair].text(16, 0.0, r'$L=%s$' % size)
        else:
            axs_c[index_pair].text(0, .5, r'$L=%s$' % size)  # 1. If 23 included
            axs_d[index_pair].text(16, 0.7, r'$L=%s$' % size)
        if ii == 2:
            axs_c[index_pair].axhline(y=iur_clusterings[ii], color='k', linestyle='dotted',
                                      label='Incoherent uniform random')
            axs_c[index_pair].axhline(y=ps_iur_clusterings[ii], color='k', linestyle='dashed',
                                      label='Z-basis post-selected incoherent uniform random')
            # axs_c[index_pair].axhline(y=pt_clusterings[ii], color='k', linestyle='dotted',
            #                           label='PT')
            # axs_c[index_pair].axhline(y=ps_pt_clusterings[ii], color='k', linestyle='dashed',
            #                           label='PT + PS')
        else:
            axs_c[index_pair].axhline(y=iur_clusterings[ii], color='k', linestyle='dotted')
            axs_c[index_pair].axhline(y=ps_iur_clusterings[ii], color='k', linestyle='dashed')
            # axs_c[index_pair].axhline(y=pt_clusterings[ii], color='k', linestyle='dotted')
            # axs_c[index_pair].axhline(y=ps_pt_clusterings[ii], color='k', linestyle='dashed')

        # axs_c[index_pair].set_yscale("log")
        # axs_c[index_pair].set_ylim((1e-5, 1e1))

    axs_c[indices[0]].text(-7, .51, 'a', fontweight='bold')
    axs_c[indices[1]].text(-4, .51, 'b', fontweight='bold')
    axs_c[indices[2]].text(-4, .51, 'c', fontweight='bold')

    figs_c.legend(loc='upper center', ncol=4, borderaxespad=0.5)
    # figs_c.tight_layout()
    figs_c.savefig("./paper_analysis/clustering_vignettes_xx_zz%s-%s.pdf" % (sizes[0], sizes[-1]), bbox_inches="tight",
                   dpi=600)
    plt.close(figs_c)

    return


def construct_classical_mutual_information_matrix(qubits, results_dict, ising_mitigation, invariant,
                                                  readout_correction):
    # Originally intended to compute classical mutual information on just z-basis measurements
    mi_matrix = np.zeros((len(qubits), len(qubits)), dtype=np.float64)

    z_counts = results_dict["All_" + "Z"]
    if readout_correction == "RC_True":
        z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
    if ising_mitigation == "PS_True":
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


def construct_classical_mutual_information_matrix_xx(qubits, results_dict, ising_mitigation, invariant,
                                                     readout_correction):
    # Originally intended to compute classical mutual information on just z-basis measurements
    mi_matrix = np.zeros((len(qubits), len(qubits)), dtype=np.float64)

    z_counts = results_dict["All_" + "X"]
    if readout_correction == "RC_True":
        z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
    if ising_mitigation == "PS_True":
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


def output_mutual_information(dim, sizes, t_max, no_counts, experimental_repetitions,
                              rule, activation_unitaries, sim_modes, two_qubit_gate,
                              processor, observables, ising_mitigations, readout_correction,
                              floquet_calibrations):
    # mutual_information_dict = {}
    for jj, size in enumerate(sizes):
        qubits = [cirq.LineQubit(i) for i in range(0, size)]
        initial_conditions = [[int(size - 1) // int(2)]]
        for i in range(0, len(initial_conditions)):
            initial_site_indices = initial_conditions[i]
            invariant = calculate_z_invariant(size=size, initial_site_indices=initial_site_indices)

            exp_param_dict = {}
            for sim_mode in sim_modes:
                for ising_mitigation in ising_mitigations:
                    if sim_mode == "cirq" and ising_mitigation == "PS_True":
                        continue
                    for floquet_calibration in floquet_calibrations:
                        if sim_mode == "cirq" and floquet_calibration:
                            continue
                        elif not floquet_calibration and ising_mitigation == "PS_True":
                            continue

                        exp_param_key = (sim_mode, ising_mitigation, floquet_calibration)

                        exp_rep_dict = {}
                        for r in range(0, experimental_repetitions):

                            t_dict = {}
                            for t in range(0, t_max):
                                circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                                      experimental_repetitions, rule, activation_unitaries, sim_mode,
                                                      two_qubit_gate,
                                                      processor, observables, floquet_calibration, r, t)
                                results_dict = read_in_data_file(circuit_parameters=circuit_parameters)

                                mi_matrix = construct_classical_mutual_information_matrix(qubits, results_dict,
                                                                                          ising_mitigation,
                                                                                          invariant,
                                                                                          readout_correction)
                                t_dict[t] = mi_matrix

                            exp_rep_dict[r] = t_dict

                        exp_param_dict[exp_param_key] = exp_rep_dict

            with open("./paper_analysis/mutual_information_%s.pickle" % size, "wb") as handle:
                pickle.dump(exp_param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


def plot_population_vignettes_long(dim, sizes, t_max, no_counts, experimental_repetitions,
                                   rule, activation_unitaries, sim_modes, two_qubit_gate,
                                   processor, observables, ising_mitigations, readout_correction,
                                   floquet_calibrations):
    fig_0 = plt.figure()
    for jj, size in enumerate(sizes):
        initial_conditions = [[int(size - 1) // int(2)]]
        for i in range(0, len(initial_conditions)):
            initial_site_indices = initial_conditions[i]
            invariant = calculate_z_invariant(size=size, initial_site_indices=initial_site_indices)

            qubits = [cirq.LineQubit(j) for j in range(0, size)]
            t_actual_max = t_max
            actual_exp_reps = experimental_repetitions
            figs, axs = plt.subplots(nrows=3, ncols=10, sharex=True, sharey=True, squeeze=False, figsize=(18, 4.8))
            indices = [(i, j) for i in range(0, 3) for j in range(0, 10)]

            retained_fractions = np.ones((t_max, experimental_repetitions))
            for t in range(0, t_max):
                for sim_mode in sim_modes:
                    for ising_mitigation in ising_mitigations:
                        if sim_mode == "cirq" and ising_mitigation == "PS_True":
                            continue
                        for floquet_calibration in floquet_calibrations:
                            if sim_mode == "cirq" and floquet_calibration:
                                continue
                            elif not floquet_calibration and ising_mitigation == "PS_True":
                                continue
                            magnetizations = np.zeros((size, actual_exp_reps))
                            for r in range(0, actual_exp_reps):
                                circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                                      experimental_repetitions, rule, activation_unitaries, sim_mode,
                                                      two_qubit_gate,
                                                      processor, observables, floquet_calibration, r, t)
                                results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
                                z_counts = results_dict["All_Z"]
                                if readout_correction == "RC_True":
                                    z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
                                if ising_mitigation == "PS_True":
                                    z_counts = post_selection(counts_dict=z_counts, qubits=qubits, invariant=invariant)
                                    retained_counts = sum(z_counts.values())
                                    retained_fractions[t, r] = float(retained_counts) / float(no_counts)
                                for j in range(0, size):
                                    if sim_mode == "engine":
                                        magnetizations[j, r] += rescale(t) * 0.5 * (
                                                1 - local_magnetization(site_index=j,
                                                                        counts_dict=z_counts,
                                                                        qubits=qubits))
                                    else:
                                        magnetizations[j, r] += 0.5 * (1 - local_magnetization(site_index=j,
                                                                                               counts_dict=z_counts,
                                                                                               qubits=qubits))
                            avg_magnetizations = np.average(magnetizations, axis=1)
                            std_magnetizations = np.std(magnetizations, axis=1)

                            if t < t_actual_max:
                                index_pair = indices[t]

                                if sim_mode == "cirq":
                                    if t == 0:
                                        axs[index_pair].errorbar(qubits, avg_magnetizations, xerr=None,
                                                                 yerr=std_magnetizations,
                                                                 marker=".", linestyle="solid", capsize=5,
                                                                 label="Emulated", color="tab:blue")
                                        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                        #       str(floquet_calibration))
                                    else:
                                        axs[index_pair].errorbar(qubits, avg_magnetizations, xerr=None,
                                                                 yerr=std_magnetizations,
                                                                 marker=".", linestyle="solid", capsize=5,
                                                                 color="tab:blue")
                                elif sim_mode == "engine" and (ising_mitigation == "PS_True" and floquet_calibration):
                                    # print(std_magnetizations)
                                    if t == 0:
                                        axs[index_pair].errorbar(qubits, avg_magnetizations, xerr=None,
                                                                 yerr=std_magnetizations,
                                                                 marker=".", linestyle="", capsize=5,
                                                                 label="Post-selected", color="tab:green")
                                        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                        #      str(floquet_calibration))
                                    else:
                                        axs[index_pair].errorbar(qubits, avg_magnetizations, xerr=None,
                                                                 yerr=std_magnetizations,
                                                                 capsize=5, marker=".", linestyle="", color="tab:green")
                                # elif sim_mode == "engine" and (
                                #         ising_mitigation == "PS_True" and not floquet_calibration):
                                #     if t == 0:
                                #         axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="",
                                #                              label="Expt. PS")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                                #     else:
                                #         axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="")
                                elif sim_mode == "engine" and (ising_mitigation == "PS_False" and floquet_calibration):
                                    if t == 0:
                                        axs[index_pair].errorbar(qubits, avg_magnetizations, xerr=None,
                                                                 yerr=std_magnetizations,
                                                                 marker=".", linestyle="", capsize=5,
                                                                 label="Raw", color="tab:red")
                                        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                        #       str(floquet_calibration))
                                    else:
                                        axs[index_pair].errorbar(qubits, avg_magnetizations, xerr=None,
                                                                 yerr=std_magnetizations,
                                                                 marker=".", linestyle="", capsize=5,
                                                                 color="tab:red")
                                else:
                                    #     if t == 0:
                                    #         axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="",
                                    #                              label="Expt. Raw")
                                    # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                    #       str(floquet_calibration))
                                    #     else:
                                    #         axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="")
                                    if size <= 11:
                                        # axs[index_pair].set_xticks([i for i in range(0, size, 2)])
                                        axs[index_pair].set_xticks([1, (size // 2) + 1, size])
                                        axs[index_pair].set_xticklabels([1, (size // 2) + 1, size])
                                    else:
                                        # axs[index_pair].set_xticks([i for i in range(0, size, 4)])
                                        axs[index_pair].set_xticks([1, (size // 2) + 1, size])
                                        axs[index_pair].set_xticklabels([1, (size // 2) + 1, size])
                                    if index_pair[0] == 2:
                                        axs[index_pair].set_xlabel(r"Site, $i$")
                                    if index_pair[1] == 0:
                                        axs[index_pair].set_ylabel(r"Pop., $\langle n_i \rangle$")

                if t < 10:
                    # axs[index_pair].text(0.7 * (size - 1), .85, r'$t=%s$' % t)
                    axs[index_pair].text(0., .8, r'$t=%s$' % t, fontsize=14)
                elif t in [i for i in range(10, t_actual_max)]:
                    axs[index_pair].text(0., .8, r'$t=%s$' % t, fontsize=14)

            figs.legend(loc='upper center', ncol=3, columnspacing=.5, labelspacing=0.25,
                        borderpad=0.2,
                        handletextpad=0.2)
            # plt.tight_layout()
            plt.savefig("./paper_analysis/population_vignettes_long_%s.pdf" % size, bbox_inches="tight", dpi=600)
            plt.close(figs)

            avg_retained_fractions = np.average(retained_fractions, axis=1)
            std_retained_fractions = np.std(retained_fractions, axis=1)
            plt.errorbar(np.array([i for i in range(0, t_max)]), avg_retained_fractions, yerr=std_retained_fractions,
                         marker=".", linestyle="", capsize=5, label=r"$L=%s$" % size, figure=fig_0)
    plt.axhline(y=1. / float(no_counts), color='k', linestyle='dashed', label=r"$1/(No. Counts)$")
    plt.xlabel(r"QCA Cycle ($t$)")
    plt.ylabel(r"Retained Count Fraction After PS")
    plt.legend(loc="upper right")
    plt.yscale("log")
    # plt.savefig("./paper_analysis/retained_fractions_" + readout_correction + ising_mitigation + ".pdf",
    #             bbox_inches="tight")
    plt.close(fig_0)

    return


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


def func2(x, a, b):
    return a * np.exp(-b * x)


def plot_retained_fractions(dim, sizes, t_max, no_counts, experimental_repetitions,
                            rule, activation_unitaries, sim_modes, two_qubit_gate,
                            processor, observables, ising_mitigations, readout_correction,
                            floquet_calibrations):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', 'b', 'g', 'r', 'c', 'm']
    # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    amplitudes = []
    decay_constants = []
    asymptotes = []
    all_avg_ret_fracs = []
    all_std_ret_fracs = []
    fig_0, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5.4))
    for jj, size in enumerate(sizes):
        initial_conditions = [[int(size - 1) // int(2)]]
        for i in range(0, len(initial_conditions)):
            initial_site_indices = initial_conditions[i]
            invariant = calculate_z_invariant(size=size, initial_site_indices=initial_site_indices)

            qubits = [cirq.LineQubit(j) for j in range(0, size)]
            t_actual_max = t_max
            actual_exp_reps = experimental_repetitions

            retained_fractions = np.ones((t_max, experimental_repetitions))
            for t in range(0, t_max):
                for sim_mode in sim_modes:
                    for ising_mitigation in ising_mitigations:
                        if sim_mode == "cirq" and ising_mitigation == "PS_True":
                            continue
                        for floquet_calibration in floquet_calibrations:
                            if sim_mode == "cirq" and floquet_calibration:
                                continue
                            elif not floquet_calibration and ising_mitigation == "PS_True":
                                continue
                            for r in range(0, actual_exp_reps):
                                circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                                      experimental_repetitions, rule, activation_unitaries, sim_mode,
                                                      two_qubit_gate,
                                                      processor, observables, floquet_calibration, r, t)
                                results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
                                z_counts = results_dict["All_Z"]
                                if readout_correction == "RC_True":
                                    z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
                                if ising_mitigation == "PS_True":
                                    z_counts = post_selection(counts_dict=z_counts, qubits=qubits, invariant=invariant)
                                    retained_counts = sum(z_counts.values())
                                    retained_fractions[t, r] = float(retained_counts) / float(no_counts)

            avg_retained_fractions = np.average(retained_fractions, axis=1)
            all_avg_ret_fracs.append(avg_retained_fractions.tolist())
            std_retained_fractions = np.std(retained_fractions, axis=1)
            all_std_ret_fracs.append(std_retained_fractions.tolist())
            ax1.errorbar(np.array([i for i in range(0, t_max)]), avg_retained_fractions, yerr=std_retained_fractions,
                         marker=".", linestyle="", capsize=5, label=r"$L=%s$" % size, figure=fig_0, color=colors[jj])
            # Now do some perfunctory fitting
            if size == 21:
                bounds = ([-np.inf, -np.inf, 1.5e-4], [np.inf, np.inf, np.inf])
            elif size == 23:
                bounds = ([-np.inf, -np.inf, 5.e-5], [np.inf, np.inf, np.inf])
            else:
                bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(func, np.array([i for i in range(0, 14)]), avg_retained_fractions[0:14],
                                   bounds=bounds)
            amplitudes.append(popt[0])
            decay_constants.append(popt[1])
            asymptotes.append(popt[2])
            ax1.plot(np.array([i for i in range(0, 14)]), func(np.array([i for i in range(0, 14)]), *popt),
                     linestyle='solid', color=colors[jj])

    all_avg_ret_fracs = np.array(all_avg_ret_fracs)
    all_std_ret_fracs = np.array(all_std_ret_fracs)
    print(all_avg_ret_fracs, all_std_ret_fracs)

    print(amplitudes)
    print(decay_constants)
    print(asymptotes)
    ax1.axhline(y=1. / float(no_counts), color='k', linestyle='dashed', label=r"$1/N_c$")
    ax1.set_xlabel(r"QCA cycle, $t$")
    ax1.set_ylabel(r"Retained count fraction after PS")
    ax1.legend(loc="upper right", borderaxespad=0)
    ax1.set_yscale("log")
    ax1.text(-10, 5.e0, "a", weight="bold")

    b_size_constants = []
    shift = 0
    for ii in range(0+shift, 14+shift):  # Goes to some max number of timesteps
        ax2.errorbar(np.array(sizes), all_avg_ret_fracs[:, ii], yerr=all_std_ret_fracs[:, ii],
                     marker=".", linestyle="", capsize=5, label=r"$t=%s$" % ii, color=colors[ii-shift])
        popt, pcov = curve_fit(func2, np.array(sizes[2:]), all_avg_ret_fracs[2:, ii])#, p0=[1., 1., 0.])
                               # bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, 0.]))
        b_size_constants.append(popt[1])
        ax2.plot(np.array(sizes), func2(np.array(sizes), *popt),
                 linestyle='solid', color=colors[ii-shift])
    print("bL", b_size_constants)

    ax2.set_xlabel(r"System size, $L$")
    ax2.set_ylabel(r"Retained count fraction after PS")
    ax2.legend(loc="bottom left", borderaxespad=0, ncol=2, columnspacing=1.0)
    # plt.xscale("log")
    ax2.set_yscale("log")
    ax1.text(32, 5.e0, "b", weight="bold")
    plt.tight_layout()
    plt.savefig("./paper_analysis/retained_fractions_transpose_" + readout_correction + ising_mitigation + ".pdf",
                bbox_inches="tight")
    plt.close(fig_0)

    fig_1 = plt.figure()
    plt.plot([i for i in range(0, 14)], b_size_constants, color='k', marker='.', linestyle='', )
    plt.xlabel(r"QCA cycle, $t$")
    plt.ylabel(r"$b_L$")
    # plt.yscale("log")
    plt.tight_layout()
    plt.savefig("./paper_analysis/ret_frac_decays_" + readout_correction + ising_mitigation + ".pdf",
                bbox_inches="tight")
    plt.close(fig_1)

    return


def func_power(x, a, b):
    return a * np.power(x, b)


def hilbert_bound_saturation(dim, sizes, t_max, no_counts, experimental_repetitions,
                             rule, activation_unitaries, sim_modes, two_qubit_gate,
                             processor, observables, ising_mitigations, readout_correction,
                             floquet_calibrations):
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan', 'b', 'g', 'r', 'c', 'm']

    fig_1, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5.4))

    relative_bound_saturation = []
    protected_hilbert_dimensions = []
    relative_hilbert_dimensions = []
    for jj, size in enumerate(sizes):
        initial_conditions = [[int(size - 1) // int(2)]]  # This line fixes regardless of input
        for ii in range(0, len(initial_conditions)):
            initial_site_indices = initial_conditions[ii]
            invariant = calculate_z_invariant(size=size, initial_site_indices=initial_site_indices)

            # Generate theoretical bound for protected Hilbert space
            # This will serve to check Logan's result too
            # Loop through full Hilbert space
            bin_fmt = '{0:0' + str(size) + 'b}'
            state_counter = 0
            for s in range(0, 2**size):
                bit_string = bin_fmt.format(int(s))
                new_string = '0' + bit_string + '0'
                invariant_counter = 0
                for i in range(0, len(new_string) - 1):
                    if new_string[i] == new_string[i + 1]:
                        invariant_counter += 1
                    else:
                        invariant_counter -= 1
                if invariant_counter == invariant:  # If a given state shares the protected invariant ...
                    state_counter += 1  # Tabulate that it belongs to the protected Hilbert space.
            protected_hilbert_dimensions.append(state_counter)
            relative_hilbert_dimensions.append(float(state_counter)/float(2**size))

            # Next, read in data emulations and check to what extent this bound is
            # saturated over the cycle window we look at
            sim_mode = "cirq"
            r = 0
            floquet_calibration = False
            time_sliced_dimensions = []
            for t in range(0, t_max):
                circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                      experimental_repetitions, rule, activation_unitaries, sim_mode,
                                      two_qubit_gate,
                                      processor, observables, floquet_calibration, r, t)
                results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
                z_counts = results_dict["All_Z"]
                time_sliced_dimensions.append(len(z_counts.keys()))
            relative_time_sliced_dimensions = np.array(time_sliced_dimensions) / float(state_counter)
            ax3.plot([t for t in range(0, t_max)], time_sliced_dimensions, color=colors[jj], label=r"L=%s" % size)
            ax3.axhline(state_counter, linestyle="dashed", color=colors[jj])
            ax4.plot(size, np.amax(relative_time_sliced_dimensions), marker="o", linestyle="", color=colors[jj])

    ax3.text(-10, 300, "a", weight="bold")
    ax3.text(32, 300, "b", weight="bold")
    ax3.legend(loc="right")
    ax3.set_xlabel(r"QCA cycle, $t$")
    ax3.set_ylabel(r"Occupied number of states in $\mathcal{H}_{\mathcal{O}}$")
    ax4.set_xlabel(r"System size, $L$")
    ax4.set_ylabel(r"Occupied fraction of $\mathcal{H}_{O}$")
    # ax4.set_yscale("log")
    plt.tight_layout()
    plt.savefig("./paper_analysis/hilbert_saturation.pdf", bbox_inches="tight")
    plt.close(fig_1)

    # Now fit and back out exponent base
    popt, pcov = curve_fit(func_power, np.array(sizes), np.array(protected_hilbert_dimensions))
    fig_0, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5.4))
    ax1.plot(sizes, protected_hilbert_dimensions, marker='o', linestyle='', label="Emulated")
    ax1.plot(np.array(sizes), func_power(np.array(sizes), *popt), label=r"$f = %.2f L^{%.2f}$" % (popt[0], popt[1]))
    ax1.legend()
    ax1.set_xlabel(r"System size, $L$")
    ax1.set_ylabel(r"Absolute dimension, dim($\mathcal{H}_{\mathcal{O}}$)")
    ax1.set_yscale("log")
    ax1.set_xscale("log")

    ax2.plot(sizes, relative_hilbert_dimensions, marker='o', linestyle='', label="Emulated")
    ax2.plot(np.array(sizes), np.divide(func_power(np.array(sizes), *popt), np.power(2, sizes)),
             label=r"$f = %.2f L^{%.2f} / 2^L$" % (popt[0], popt[1]))
    ax2.legend()
    ax2.set_xlabel(r"System size, $L$")
    ax2.set_ylabel(r"Relative dimension, dim($\mathcal{H}_{\mathcal{O}}$)$/$dim($\mathcal{H}$)")
    ax2.set_yscale("log")
    plt.tight_layout()
    plt.savefig("./paper_analysis/hilbert_dim.pdf",
                bbox_inches="tight")
    plt.close(fig_0)



    '''
            qubits = [cirq.LineQubit(j) for j in range(0, size)]
            t_actual_max = t_max
            actual_exp_reps = experimental_repetitions

            retained_fractions = np.ones((t_max, experimental_repetitions))
            for t in range(0, t_max):
                for sim_mode in sim_modes:
                    for ising_mitigation in ising_mitigations:
                        if sim_mode == "cirq" and ising_mitigation == "PS_True":
                            continue
                        for floquet_calibration in floquet_calibrations:
                            if sim_mode == "cirq" and floquet_calibration:
                                continue
                            elif not floquet_calibration and ising_mitigation == "PS_True":
                                continue
                            for r in range(0, actual_exp_reps):
                                circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                                      experimental_repetitions, rule, activation_unitaries, sim_mode,
                                                      two_qubit_gate,
                                                      processor, observables, floquet_calibration, r, t)
                                results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
                                z_counts = results_dict["All_Z"]
                                if readout_correction == "RC_True":
                                    z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
                                if ising_mitigation == "PS_True":
                                    z_counts = post_selection(counts_dict=z_counts, qubits=qubits, invariant=invariant)
                                    retained_counts = sum(z_counts.values())
                                    retained_fractions[t, r] = float(retained_counts) / float(no_counts)

            avg_retained_fractions = np.average(retained_fractions, axis=1)
            all_avg_ret_fracs.append(avg_retained_fractions.tolist())
            std_retained_fractions = np.std(retained_fractions, axis=1)
            all_std_ret_fracs.append(std_retained_fractions.tolist())
            ax1.errorbar(np.array([i for i in range(0, t_max)]), avg_retained_fractions, yerr=std_retained_fractions,
                         marker=".", linestyle="", capsize=5, label=r"$L=%s$" % size, figure=fig_0, color=colors[jj])
            # Now do some perfunctory fitting
            if size == 21:
                bounds = ([-np.inf, -np.inf, 1.5e-4], [np.inf, np.inf, np.inf])
            elif size == 23:
                bounds = ([-np.inf, -np.inf, 5.e-5], [np.inf, np.inf, np.inf])
            else:
                bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(func, np.array([i for i in range(0, 14)]), avg_retained_fractions[0:14],
                                   bounds=bounds)
            amplitudes.append(popt[0])
            decay_constants.append(popt[1])
            asymptotes.append(popt[2])
            ax1.plot(np.array([i for i in range(0, 14)]), func(np.array([i for i in range(0, 14)]), *popt),
                     linestyle='solid', color=colors[jj])

    all_avg_ret_fracs = np.array(all_avg_ret_fracs)
    all_std_ret_fracs = np.array(all_std_ret_fracs)
    print(all_avg_ret_fracs, all_std_ret_fracs)

    print(amplitudes)
    print(decay_constants)
    print(asymptotes)
    ax1.axhline(y=1. / float(no_counts), color='k', linestyle='dashed', label=r"$1/N_c$")
    ax1.set_xlabel(r"QCA cycle, $t$")
    ax1.set_ylabel(r"Retained count fraction after PS")
    ax1.legend(loc="upper right", borderaxespad=0)
    ax1.set_yscale("log")
    ax1.text(-10, 5.e0, "a", weight="bold")

    b_size_constants = []
    shift = 0
    for ii in range(0+shift, 14+shift):  # Goes to some max number of timesteps
        ax2.errorbar(np.array(sizes), all_avg_ret_fracs[:, ii], yerr=all_std_ret_fracs[:, ii],
                     marker=".", linestyle="", capsize=5, label=r"$t=%s$" % ii, color=colors[ii-shift])
        popt, pcov = curve_fit(func2, np.array(sizes[2:]), all_avg_ret_fracs[2:, ii])#, p0=[1., 1., 0.])
                               # bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, 0.]))
        b_size_constants.append(popt[1])
        ax2.plot(np.array(sizes), func2(np.array(sizes), *popt),
                 linestyle='solid', color=colors[ii-shift])
    print("bL", b_size_constants)

    ax2.set_xlabel(r"System size, $L$")
    ax2.set_ylabel(r"Retained count fraction after PS")
    ax2.legend(loc="bottom left", borderaxespad=0, ncol=2, columnspacing=1.0)
    # plt.xscale("log")
    ax2.set_yscale("log")
    ax1.text(32, 5.e0, "b", weight="bold")
    plt.tight_layout()
    plt.savefig("./paper_analysis/retained_fractions_transpose_" + readout_correction + ising_mitigation + ".pdf",
                bbox_inches="tight")
    plt.close(fig_0)

    fig_1 = plt.figure()
    plt.plot([i for i in range(0, 14)], b_size_constants, color='k', marker='.', linestyle='', )
    plt.xlabel(r"QCA cycle, $t$")
    plt.ylabel(r"$b_L$")
    # plt.yscale("log")
    plt.tight_layout()
    plt.savefig("./paper_analysis/ret_frac_decays_" + readout_correction + ising_mitigation + ".pdf",
                bbox_inches="tight")
    plt.close(fig_1)
    '''
    return


def plot_population_vignettes(dim, sizes, t_max, no_counts, experimental_repetitions,
                              rule, activation_unitaries, sim_modes, two_qubit_gate,
                              processor, observables, ising_mitigations, readout_correction,
                              floquet_calibrations):
    fig_0 = plt.figure()
    for jj, size in enumerate(sizes):
        initial_conditions = [[int(size - 1) // int(2)]]
        for i in range(0, len(initial_conditions)):
            initial_site_indices = initial_conditions[i]
            invariant = calculate_z_invariant(size=size, initial_site_indices=initial_site_indices)

            qubits = [cirq.LineQubit(j) for j in range(0, size)]
            t_actual_max = 16
            actual_exp_reps = experimental_repetitions
            figs, axs = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, squeeze=False)
            indices = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3),
                       (3, 0), (3, 1), (3, 2), (3, 3)]
            retained_fractions = np.ones((t_max, experimental_repetitions))
            for t in range(0, t_max):
                for sim_mode in sim_modes:
                    for ising_mitigation in ising_mitigations:
                        if sim_mode == "cirq" and ising_mitigation == "PS_True":
                            continue
                        for floquet_calibration in floquet_calibrations:
                            if sim_mode == "cirq" and floquet_calibration:
                                continue
                            elif not floquet_calibration and ising_mitigation == "PS_True":
                                continue
                            magnetizations = np.zeros((size, actual_exp_reps))
                            for r in range(0, actual_exp_reps):
                                circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                                      experimental_repetitions, rule, activation_unitaries, sim_mode,
                                                      two_qubit_gate,
                                                      processor, observables, floquet_calibration, r, t)
                                results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
                                z_counts = results_dict["All_Z"]
                                if readout_correction == "RC_True":
                                    z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
                                if ising_mitigation == "PS_True":
                                    z_counts = post_selection(counts_dict=z_counts, qubits=qubits, invariant=invariant)
                                    retained_counts = sum(z_counts.values())
                                    retained_fractions[t, r] = float(retained_counts) / float(no_counts)
                                for j in range(0, size):
                                    if sim_mode == "engine":
                                        magnetizations[j, r] += rescale(t) * 0.5 * (
                                                1 - local_magnetization(site_index=j,
                                                                        counts_dict=z_counts,
                                                                        qubits=qubits))
                                    else:
                                        magnetizations[j, r] += 0.5 * (1 - local_magnetization(site_index=j,
                                                                                               counts_dict=z_counts,
                                                                                               qubits=qubits))
                            avg_magnetizations = np.average(magnetizations, axis=1)
                            std_magnetizations = np.std(magnetizations, axis=1)

                            if t < t_actual_max:
                                index_pair = indices[t]

                                if sim_mode == "cirq":
                                    if t == 0:
                                        axs[index_pair].plot(qubits, avg_magnetizations, marker=".",
                                                             label="Exact", color="tab:blue")
                                        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                        #       str(floquet_calibration))
                                    else:
                                        axs[index_pair].plot(qubits, avg_magnetizations, marker=".", color="tab:blue")
                                elif sim_mode == "engine" and (ising_mitigation == "PS_True" and floquet_calibration):
                                    # print(std_magnetizations)
                                    if t == 0:
                                        axs[index_pair].errorbar(qubits, avg_magnetizations, xerr=None,
                                                                 yerr=std_magnetizations,
                                                                 marker=".", linestyle="", capsize=5,
                                                                 label="Post-Selected", color="tab:green")
                                        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                        #      str(floquet_calibration))
                                    else:
                                        axs[index_pair].errorbar(qubits, avg_magnetizations, xerr=None,
                                                                 yerr=std_magnetizations,
                                                                 capsize=5, marker=".", linestyle="", color="tab:green")
                                # elif sim_mode == "engine" and (
                                #         ising_mitigation == "PS_True" and not floquet_calibration):
                                #     if t == 0:
                                #         axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="",
                                #                              label="Expt. PS")
                                # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                #       str(floquet_calibration))
                                #     else:
                                #         axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="")
                                elif sim_mode == "engine" and (ising_mitigation == "PS_False" and floquet_calibration):
                                    if t == 0:
                                        axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="",
                                                             label="Raw", color="tab:red")
                                        # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                        #       str(floquet_calibration))
                                    else:
                                        axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="",
                                                             color="tab:red")
                                else:
                                    #     if t == 0:
                                    #         axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="",
                                    #                              label="Expt. Raw")
                                    # label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                    #       str(floquet_calibration))
                                    #     else:
                                    #         axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="")
                                    if size <= 11:
                                        # axs[index_pair].set_xticks([i for i in range(0, size, 2)])
                                        axs[index_pair].set_xticks([0, (size - 1) // 2, size - 1])
                                    else:
                                        # axs[index_pair].set_xticks([i for i in range(0, size, 4)])
                                        axs[index_pair].set_xticks([0, (size - 1) // 2, size - 1])
                                    if index_pair[0] == 3:
                                        axs[index_pair].set_xlabel(r"Site ($i$)")
                                    if index_pair[1] == 0:
                                        axs[index_pair].set_ylabel(r"$\langle n_i \rangle$")

                if t < 10:
                    # axs[index_pair].text(0.7 * (size - 1), .85, r'$t=%s$' % t)
                    axs[index_pair].text(0., .8, r'$%s$' % t)
                elif t in [i for i in range(10, t_actual_max)]:
                    axs[index_pair].text(0., .8, r'$%s$' % t)

            figs.legend(loc='upper center', ncol=3, columnspacing=.5, labelspacing=0.25,
                        borderpad=0.2,
                        handletextpad=0.2)
            # plt.tight_layout()
            plt.savefig("./paper_analysis/population_vignettes_%s.pdf" % size, bbox_inches="tight", dpi=600)
            plt.close(figs)

            avg_retained_fractions = np.average(retained_fractions, axis=1)
            std_retained_fractions = np.std(retained_fractions, axis=1)
            plt.errorbar(np.array([i for i in range(0, t_max)]), avg_retained_fractions, yerr=std_retained_fractions,
                         marker=".", linestyle="", capsize=5, label=r"$L=%s$" % size, figure=fig_0)
    plt.axhline(y=1. / float(no_counts), color='k', linestyle='dashed', label=r"$1/(No. Counts)$")
    plt.xlabel(r"QCA Cycle ($t$)")
    plt.ylabel(r"Retained Count Fraction After PS")
    plt.legend(loc="upper right")
    plt.yscale("log")
    plt.savefig("./paper_analysis/retained_fractions_" + readout_correction + ising_mitigation + ".pdf")
    plt.close(fig_0)

    return


def plot_magnetization_vignettes(dim, size, t_max, initial_site_indices, no_counts, experimental_repetitions,
                                 rule, activation_unitaries, sim_modes, two_qubit_gate,
                                 processor, observables, ising_mitigations, invariant, readout_correction,
                                 floquet_calibrations):
    qubits = [cirq.LineQubit(i) for i in range(0, size)]
    t_actual_max = 16
    actual_exp_reps = experimental_repetitions
    figs, axs = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True, squeeze=False)
    indices = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3),
               (3, 0), (3, 1), (3, 2), (3, 3)]
    for t in range(0, t_actual_max):
        for sim_mode in sim_modes:
            for ising_mitigation in ising_mitigations:
                if sim_mode == "cirq" and ising_mitigation == "PS_True":
                    continue
                for floquet_calibration in floquet_calibrations:
                    if sim_mode == "cirq" and floquet_calibration:
                        continue
                    magnetizations = np.zeros((size, actual_exp_reps))
                    for r in range(0, actual_exp_reps):
                        circuit_parameters = (dim, size, t_max, initial_site_indices, no_counts,
                                              experimental_repetitions, rule, activation_unitaries, sim_mode,
                                              two_qubit_gate,
                                              processor, observables, floquet_calibration, r, t)
                        results_dict = read_in_data_file(circuit_parameters=circuit_parameters)
                        z_counts = results_dict["All_Z"]
                        if readout_correction == "RC_True":
                            z_counts = apply_correction_matrix(counts_dict=z_counts, qubits=qubits)
                        if ising_mitigation == "PS_True":
                            z_counts = post_selection(counts_dict=z_counts, qubits=qubits, invariant=invariant)
                        for j in range(0, size):
                            if sim_mode == "engine":
                                magnetizations[j, r] += rescale(t) * local_magnetization(site_index=j,
                                                                                         counts_dict=z_counts,
                                                                                         qubits=qubits)
                            else:
                                magnetizations[j, r] += local_magnetization(site_index=j,
                                                                            counts_dict=z_counts,
                                                                            qubits=qubits)
                    avg_magnetizations = np.average(magnetizations, axis=1)
                    std_magnetizations = np.std(magnetizations, axis=1)
                    index_pair = indices[t]

                    if sim_mode == "cirq":
                        if t == 0:
                            axs[index_pair].plot(qubits, avg_magnetizations, marker=".",
                                                 label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                                       str(floquet_calibration))
                        else:
                            axs[index_pair].plot(qubits, avg_magnetizations, marker=".")
                    elif sim_mode == "engine" and (ising_mitigation == "PS_True" and floquet_calibration):
                        print(std_magnetizations)
                        if t == 0:
                            axs[index_pair].errorbar(qubits, avg_magnetizations, xerr=None, yerr=std_magnetizations,
                                                     marker=".", linestyle="", capsize=5,
                                                     label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                                           str(floquet_calibration))
                        else:
                            axs[index_pair].errorbar(qubits, avg_magnetizations, xerr=None, yerr=std_magnetizations,
                                                     capsize=5, marker=".", linestyle="")
                    else:
                        if t == 0:
                            axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="",
                                                 label=sim_mode + "_" + ising_mitigation + "_FC_" +
                                                       str(floquet_calibration))
                        else:
                            axs[index_pair].plot(qubits, avg_magnetizations, marker=".", linestyle="")
                        if size <= 11:
                            # axs[index_pair].set_xticks([i for i in range(0, size, 2)])
                            axs[index_pair].set_xticks([0, (size - 1) // 2, size - 1])
                        else:
                            axs[index_pair].set_xticks([0, (size - 1) // 2, size - 1])
                            # axs[index_pair].set_xticks([i for i in range(0, size, 4)])
                        if index_pair[0] == 3:
                            axs[index_pair].set_xlabel(r"Qubit Site ($i$)")
                        if index_pair[1] == 0:
                            axs[index_pair].set_ylabel(r"Avg. Mag.")  # ($\langle Z_i \rangle$)")

        if t < 10:
            axs[index_pair].text(0.7 * (size - 1), -1, r'$t=%s$' % t)
        else:
            axs[index_pair].text(0.6 * (size - 1), -1, r'$t=%s$' % t)

    figs.legend(loc='upper left', ncol=3, fontsize="small")
    plt.savefig("./paper_analysis/magnetization_vignettes_%s.pdf" % size)
    plt.close(figs)
    return


# def dir_path(which_dir_path, circuit_parameters):
#     if which_dir_path == "unprocessed":
#         print("")
#     return


def rescale(time_step: int):
    a = 0.0  # 0.025
    b = 1.0
    # return b + a * time_step
    return 1. / (b - a * time_step)


def calculate_z_invariant(size, initial_site_indices):
    initial_state = [0 for i in range(0, size)]
    for i in range(0, len(initial_site_indices)):
        initial_state[i] = 1
    invariant_counter = 0
    padded_state = [0] + initial_state + [0]
    for i in range(0, len(padded_state) - 1):
        if padded_state[i] == padded_state[i + 1]:
            invariant_counter += 1
        else:
            invariant_counter -= 1
    return invariant_counter


def read_in_confirmation_data_file(circuit_parameters, readout_correction, ising_mitigation, which):
    # which field should be "0"-"3" or "avg"
    root_dir = os.getcwd()
    top_dir = "{}d".format(circuit_parameters[0])
    bot_dir = "confirmation_data_qb{}_tm{}_isi{}_nc{}_er{}_ru{}_au{}_{}_{}_{}_{}_fc{}".format(*circuit_parameters[1:len(
        circuit_parameters) - 2])
    dir_path = os.path.join(root_dir, top_dir, bot_dir)

    data = np.load(dir_path + "/" + str(which) + "_" + "clustering_and_disparity" + '_' + readout_correction +
                   ising_mitigation +
                   ".npz")
    time = data['arr_0']
    clustering = data['arr_1']
    disparity = data['arr_2']
    return time, clustering, disparity


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


if __name__ == "__main__":
    main()
