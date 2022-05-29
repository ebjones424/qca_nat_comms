import numpy as np
import json
import time
import cirq
import cirq_google as cg
import os
import pickle


def main():
    t0 = time.time()
    # Parameters #
    # Dimensionality of the QCA
    d = 1

    # Max time steps
    t_max = 30

    # Number of QCA sites.
    sizes = [15, 17, 19]
    # sizes = [11, 13, 15, 19]

    no_counts = 100_000  # Number of histogram repetitions per measurement.
    exp_reps = 4  # Number of experimental repetitions
    activation_unitaries = ["H"]
    rule = "T6"

    sim_mode = 'cirq'  # 'engine' or 'cirq'
    two_qubit_gate = "parasitic_root_iSWAP"  # 'CZ' or 'Sycamore', 'root_iSWAP' or 'parasitic_root_iSWAP'
    processor = 'weber'  # 'rainbow' or 'weber' or something else ('NA' for 'cirq' sim_mode)
    observables = 'only_x'  # Or 'only_z' or 'mutual_information'
    floquet_calibration = False  # Or False

    for size in sizes:
        # Just one for the central site and one for every (L+1)/4 sites according to Eliot
        # mod_number = int(size+1)//int(4)
        initial_conditions = [[int(size - 1) // int(2)]]  # ,
        # initial_conditions = [[0, 1, 3, 4]]
        # [i for i in range(len(qubits)) if i % mod_number == mod_number - 1]]
        for i in range(0, len(initial_conditions)):
            initial_site_indices = initial_conditions[i]

            generate_data_1d(dim=d, size=size, t_max=t_max, initial_site_indices=initial_site_indices,
                             no_counts=no_counts, experimental_repetitions=exp_reps, rule=rule,
                             activation_unitaries=activation_unitaries, sim_mode=sim_mode,
                             two_qubit_gate=two_qubit_gate, processor=processor, observables=observables,
                             floquet_calibration=floquet_calibration)

    t1 = time.time()
    print("Time:", t1 - t0)
    return


def which_chain(size: int, config_no: int):
    project_id = ''
    engine = cg.Engine(project_id=project_id)
    processor_object = engine.get_processor('weber')
    device = processor_object.get_device([cg.SQRT_ISWAP_GATESET])
    if size == 5:
        qubit_sets_indices = [
            [(1, 6), (2, 6), (2, 7), (3, 7), (4, 7)],
            [(1, 6), (2, 6), (2, 7), (3, 7), (3, 8)],
            [(3, 5), (3, 6), (3, 7), (2, 7), (1, 7)],
            [(0, 6), (1, 6), (2, 6), (2, 7), (1, 7)]
        ]
        # Convert indices to grid qubits.
        qubit_sets = [[cirq.GridQubit(*idx) for idx in qubit_indices]
                      for qubit_indices in qubit_sets_indices]
        # line_length = 45
        # line = cg.line_on_device(device, line_length)
        # segment_length = size
        # qubit_sets = [line[i: i + segment_length]
        #               for i in range(0, line_length - segment_length + 1, segment_length)]
        return qubit_sets[config_no]
    elif size == 7:  # This is the test size from the cirq tutorial.
        qubit_sets_indices = [
            [(0, 6), (1, 6), (2, 6), (2, 7), (3, 7), (4, 7), (4, 8)],
            [(0, 6), (1, 6), (2, 6), (2, 7), (3, 7), (3, 8), (2, 8)],
            [(3, 5), (3, 6), (3, 7), (2, 7), (1, 7), (1, 6), (2, 6)],
            [(0, 6), (1, 6), (2, 6), (2, 7), (3, 7), (3, 6), (3, 5)]
        ]
        # Convert indices to grid qubits.
        qubit_sets = [[cirq.GridQubit(*idx) for idx in qubit_indices]
                      for qubit_indices in qubit_sets_indices]
        return qubit_sets[config_no]
    elif size == 9:  # This is the test size from the cirq tutorial.
        qubit_sets_indices = [
            [(0, 5), (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (4, 8), (5, 8)],
            [(0, 5), (1, 5), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5), (4, 4), (5, 4)],
            [(1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
            [(1, 4), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 6), (3, 5), (3, 4)]
        ]
        # Convert indices to grid qubits.
        qubit_sets = [[cirq.GridQubit(*idx) for idx in qubit_indices]
                      for qubit_indices in qubit_sets_indices]
        return qubit_sets[config_no]
    elif size == 11:  # This is the test size from the cirq tutorial.
        qubit_sets_indices = [
            [(0, 5), (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (4, 8), (4, 7), (3, 7), (3, 6)],
            [(0, 5), (1, 5), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5), (4, 4), (5, 4), (5, 5), (6, 5)],
            [(1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (4, 5), (3, 5)],
            [(1, 4), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4)]
        ]
        # Convert indices to grid qubits.
        qubit_sets = [[cirq.GridQubit(*idx) for idx in qubit_indices]
                      for qubit_indices in qubit_sets_indices]
        return qubit_sets[config_no]
    elif size == 13:  # This is the test size from the cirq tutorial.
        qubit_sets_indices = [
            [(0, 5), (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (4, 8), (4, 7), (3, 7), (3, 6), (3, 5), (3, 4)],
            [(0, 5), (1, 5), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6)],
            [(1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (4, 5), (3, 5), (3, 6), (3, 7)],
            [(1, 4), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5)]
        ]
        # Convert indices to grid qubits.
        qubit_sets = [[cirq.GridQubit(*idx) for idx in qubit_indices]
                      for qubit_indices in qubit_sets_indices]
        return qubit_sets[config_no]
    elif size == 15:  # This is the test size from the cirq tutorial.
        qubit_sets_indices = [
            [(0, 5), (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (4, 8), (4, 7), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4)],
            [(0, 6), (0, 5), (1, 5), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6), (7, 5)],
            [(1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (4, 5), (3, 5), (3, 6), (3, 7), (3, 8), (4, 8)],
            [(1, 4), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6)]
        ]
        # Convert indices to grid qubits.
        qubit_sets = [[cirq.GridQubit(*idx) for idx in qubit_indices]
                      for qubit_indices in qubit_sets_indices]
        return qubit_sets[config_no]
    elif size == 17:  # This is the test size from the cirq tutorial.
        qubit_sets_indices = [
            [(0, 5), (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (4, 8), (4, 7), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5)],
            [(2, 6), (1, 6), (0, 6), (0, 5), (1, 5), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6), (7, 5)],
            [(2, 6), (2, 7), (1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (4, 5), (3, 5), (3, 6), (3, 7), (3, 8), (4, 8)],
            [(2, 4), (1, 4), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6), (7, 5)]
        ]
        # Convert indices to grid qubits.
        qubit_sets = [[cirq.GridQubit(*idx) for idx in qubit_indices]
                      for qubit_indices in qubit_sets_indices]
        return qubit_sets[config_no]
    elif size == 19:  # This is the test size from the cirq tutorial.
        qubit_sets_indices = [
            [(0, 5), (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (4, 8), (4, 7), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6)],
            [(2, 6), (1, 6), (0, 6), (0, 5), (1, 5), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6), (7, 5), (8, 5), (8, 4)],
            [(2, 6), (2, 7), (1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (4, 5), (3, 5), (3, 6), (3, 7), (3, 8), (4, 8), (5, 8), (5, 7)],
            [(2, 4), (1, 4), (1, 5), (1, 6), (2, 6), (2, 7), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (7, 6), (7, 5), (8, 5), (8, 4)]
        ]

        # Convert indices to grid qubits.
        qubit_sets = [[cirq.GridQubit(*idx) for idx in qubit_indices]
                      for qubit_indices in qubit_sets_indices]
        return qubit_sets[config_no]
    elif size == 21:  # This is the test size from the cirq tutorial.
        qubit_sets_indices = [
            [(0, 5), (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (4, 8), (4, 7), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (5, 6), (5, 7), (6, 7)],
            [(2, 8), (2, 7), (2, 6), (1, 6), (0, 6), (0, 5), (1, 5), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (5, 6), (5, 7), (4, 7), (4, 8)],
            [(2, 6), (2, 7), (1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (4, 5), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 9), (4, 8), (4, 7), (5, 7)],
            [(2, 4), (1, 4), (1, 5), (1, 6), (2, 6), (2, 7), (2, 8), (3, 8), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 8)]
        ]

        # Convert indices to grid qubits.
        qubit_sets = [[cirq.GridQubit(*idx) for idx in qubit_indices]
                      for qubit_indices in qubit_sets_indices]
        return qubit_sets[config_no]
    elif size == 23:  # This is the test size from the cirq tutorial.
        qubit_sets_indices = [
            [(1, 4), (1, 5), (0, 5), (0, 6), (1, 6), (1, 7), (2, 7), (2, 8), (3, 8), (4, 8), (4, 7), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (5, 6), (5, 7), (6, 7)],
            [(3, 9), (3, 8), (2, 8), (2, 7), (2, 6), (1, 6), (0, 6), (0, 5), (1, 5), (1, 4), (2, 4), (3, 4), (3, 5), (4, 5), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (5, 6), (5, 7), (4, 7), (4, 8)],
            [(2, 6), (2, 7), (1, 7), (1, 6), (0, 6), (0, 5), (1, 5), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (5, 5), (4, 5), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (4, 9), (4, 8), (4, 7), (5, 7)],
            [(2, 4), (1, 4), (1, 5), (0, 5), (0, 6), (1, 6), (2, 6), (2, 7), (2, 8), (3, 8), (3, 7), (3, 6), (3, 5), (3, 4), (4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 8)]
        ]

        # Convert indices to grid qubits.
        qubit_sets = [[cirq.GridQubit(*idx) for idx in qubit_indices]
                      for qubit_indices in qubit_sets_indices]
        return qubit_sets[config_no]
    elif size > 30:
        line = cg.line_on_device(device, size)
        qubit_sets = [list(line) for i in range(0, 4)]
        return qubit_sets[config_no]
    else:
        raise ValueError('Not a valid chain!')


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


def pull_calibration_data(circuit_parameters):
    engine = cg.Engine(project_id='')
    processor = engine.get_processor('rainbow')
    latest_calibration = processor.get_current_calibration()

    root_dir = os.getcwd()
    top_dir = "{}d".format(circuit_parameters[0])
    bot_dir = "data_qb{}_tm{}_isi{}_nc{}_er{}_ru{}_au{}_{}_{}_{}_{}".format(*circuit_parameters[1:len(circuit_parameters
                                                                                                      ) - 2])
    dir_path = os.path.join(root_dir, top_dir, bot_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(dir_path + '/calibration_rep{}_ts{}.pickle'.format(*circuit_parameters[-2:]), 'wb') as output:
        pickle.dump(latest_calibration, output)

    return


def generate_data_1d(dim, size, t_max, initial_site_indices, no_counts, experimental_repetitions,
                     rule, activation_unitaries, sim_mode, two_qubit_gate, processor, observables,
                     floquet_calibration):
    # dim, no_qubits, t_max, initial_site_indices, no_counts, experimental_repetitions, rule, activation_unitaries
    # will be directory-level descriptors.
    # t and r will be file-level descriptors.

    # Create an Engine object.  This uses the project id of your
    # Google cloud project.
    # project_id = ''
    # engine = cg.Engine(project_id=project_id)
    # processor_object = engine.get_processor(processor)
    # device = processor_object.get_device([cg.SQRT_ISWAP_GATESET])

    sycamore_circuits = []

    for r in range(0, experimental_repetitions):
        # Can also just fix this to a single good chain or two
        # good_chain_index = 6
        # qubits = which_chain(good_chain_index, size)
        print("Experimental Repetition:", r)
        # qubits = which_chain(size, r)
        qubits = [cirq.LineQubit(i) for i in range(0, size)]

        measurement_circuit = cirq.Circuit()
        measurement_circuit.append(cirq.measure(*qubits, key='x'))
        if observables == "only_x":
            rot_from_x = cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5)
            tmp_circuit_x = cirq.Circuit()
            tmp_circuit_x.append(rot_from_x(q) for q in qubits)
            measurement_circuit = tmp_circuit_x + measurement_circuit

        circuits_list = []
        # immediate_t_max = 20
        for t in range(0, 30):  # t_max):
            print("Time step:", t)
            base_circuit = construct_1d_base_circuit_instance(qubits=qubits, evolution_steps=t,
                                                              initial_site_indices=initial_site_indices,
                                                              rule=rule, activation_unitaries=activation_unitaries,
                                                              two_qubit_gate=two_qubit_gate, parasitic_cphase=0.,
                                                              characterization_data=None)

            # First pass optimization for sycamore architecture
            # syc_circuit = cg.optimized_for_sycamore(circuit=base_circuit, new_device=device)
            syc_circuit = base_circuit

            if t == 1:
                print("\nFully bare circuit\n", base_circuit)
                print("\nFirst pass sycamore optimized\n", syc_circuit)

            # Perform spin echo insertion for arbitrary parasitic_cphase including zero.
            echo_circuit = cirq.Circuit()
            for jj, moment in enumerate(syc_circuit):
                new_moment = []
                for operation in moment:
                    new_moment.append(operation)
                if cirq.ISwapPowGate(exponent=-0.5)(qubits[0], qubits[1]) in syc_circuit[jj - 1]:
                    new_moment.append([cirq.PhasedXZGate(axis_phase_exponent=0., x_exponent=2., z_exponent=0.)(
                        qubits[i]) for i in range(1, len(qubits), 2)
                        if cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1., z_exponent=0.)(qubits[i - 1])
                           in moment])
                    new_moment.append(cirq.PhasedXZGate(axis_phase_exponent=-1., x_exponent=1., z_exponent=0.)(
                        qubits[-1]))
                elif cirq.ISwapPowGate(exponent=-0.5)(qubits[-2], qubits[-1]) in syc_circuit[jj - 1]:
                    new_moment.append([cirq.PhasedXZGate(axis_phase_exponent=0., x_exponent=2., z_exponent=0.)(
                        qubits[i]) for i in range(2, len(qubits), 2)
                        if cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1., z_exponent=0.)(qubits[i - 1])
                           in moment])
                    new_moment.append(cirq.PhasedXZGate(axis_phase_exponent=-1., x_exponent=1., z_exponent=0.)(
                        qubits[0]))
                echo_circuit.append(new_moment, strategy=cirq.InsertStrategy.INLINE)

            if t == 1:
                print("\nSpin echoes inserted\n", echo_circuit)

            # Now we do Floquet Calibration. Extract parasitic cphase first, reconstitute circuit and then
            # Do dummy calibration on other moments.
            if floquet_calibration:
                if t < 2:
                    print("Calibrating circuit %s" % t)
                    """Step 1: Find moments in the circuit that need to be characterized."""
                    (characterized_circuit, characterization_requests
                     ) = cg.prepare_floquet_characterization_for_moments(
                        echo_circuit
                    )
                    """Show an example characterization request."""
                    # print(f"Total {len(characterization_requests)} moment(s) to characterize.")

                    # print("\nExample request")
                    # for jj in range(0, len(characterization_requests)):
                    #     request = characterization_requests[jj]
                    #     print("Gate:", request.gate)
                    #     print("Qubit pairs:", request.pairs)
                    #     print("Options: ", request.options)
                    """Step 2: Characterize moments on the engine."""
                    characterizations = cg.run_calibrations(
                        characterization_requests,
                        engine,
                        processor_id=processor,
                        gate_set=cg.SQRT_ISWAP_GATESET,
                        max_layers_per_request=1,
                    )
                    # print(f"Total: {len(characterizations)} characterizations.")
                    # for ii in range(0, len(characterizations)):
                    #     for (pair, parameters) in characterizations[ii].parameters.items():
                    #         print(f"Example pair: {pair}")
                    #         print(f"Example parameters: {parameters}")

                # Reconstruct base circuit with known parasitic cphases
                base_circuit = construct_1d_base_circuit_instance(qubits=qubits, evolution_steps=t,
                                                                  initial_site_indices=initial_site_indices,
                                                                  rule=rule, activation_unitaries=activation_unitaries,
                                                                  two_qubit_gate=two_qubit_gate, parasitic_cphase=0.,
                                                                  characterization_data=characterizations)
                # Reoptimize for sycamore
                syc_circuit = cg.optimized_for_sycamore(circuit=base_circuit, new_device=device)

                # Reinsert required spin echoes
                echo_circuit = cirq.Circuit()
                for jj, moment in enumerate(syc_circuit):
                    new_moment = []
                    for operation in moment:
                        new_moment.append(operation)
                    if cirq.ISwapPowGate(exponent=-0.5)(qubits[0], qubits[1]) in syc_circuit[jj - 1]:
                        # new_moment.append([cirq.PhasedXZGate(axis_phase_exponent=0., x_exponent=2., z_exponent=0.)(
                        #     qubits[i]) for i in range(1, len(qubits), 2)
                        #     if cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1., z_exponent=0.)(qubits[i - 1])
                        #        in moment])
                        new_moment.append(cirq.PhasedXZGate(axis_phase_exponent=-1., x_exponent=1., z_exponent=0.)(
                            qubits[-1]))
                    elif cirq.ISwapPowGate(exponent=-0.5)(qubits[-2], qubits[-1]) in syc_circuit[jj - 1]:
                        # new_moment.append([cirq.PhasedXZGate(axis_phase_exponent=0., x_exponent=2., z_exponent=0.)(
                        #     qubits[i]) for i in range(2, len(qubits), 2)
                        #     if cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1., z_exponent=0.)(qubits[i - 1])
                        #        in moment])
                        new_moment.append(cirq.PhasedXZGate(axis_phase_exponent=-1., x_exponent=1., z_exponent=0.)(
                            qubits[0]))
                    echo_circuit.append(new_moment, strategy=cirq.InsertStrategy.INLINE)

                # Finally, add gamma and zeta compensations through Floquet utility
                # First run all Floquet steps again to check that phi angles are now zero.
                (characterized_circuit, characterization_requests
                 ) = cg.prepare_floquet_characterization_for_moments(
                    echo_circuit
                )
                # characterizations = cg.run_calibrations(
                #    characterization_requests,
                #    engine,
                #    processor_id=processor,
                #    gate_set=cg.SQRT_ISWAP_GATESET,
                #    max_layers_per_request=1,
                # )
                # print(f"Total: {len(characterizations)} characterizations.")
                # for ii in range(0, len(characterizations)):
                #    for (pair, parameters) in characterizations[ii].parameters.items():
                #        print(f"Example pair: {pair}")
                #        print(f"Example parameters: {parameters}")
                """Step 3: Apply previoiusly derived corrections to the new circuit to get a calibrated circuit."""
                calibrated_circuit = cg.make_zeta_chi_gamma_compensation_for_moments(
                    characterized_circuit,
                    characterizations
                )

                if t == 1:
                    print("\nNew base circuit\n", base_circuit)
                    print("\nNew sycamore circuit\n", syc_circuit)
                    print("\nNew spin echo circuit\n", echo_circuit)
                    print("\nFully optimized and calibrated circuit\n", calibrated_circuit.circuit)

                full_circuit = calibrated_circuit.circuit + measurement_circuit
                circuits_list.append(full_circuit)

                # calibrated_circuit, *_ = cg.run_zeta_chi_gamma_compensation_for_moments(
                #    circuit=echo_circuit,
                #    engine=engine,
                #    processor_id=processor,
                #    gate_set=cg.SQRT_ISWAP_GATESET
                # )

                # print("\nFloquet calibrated\n", calibrated_circuit.circuit)
                # echo_circuit = calibrated_circuit.circuit
                # floquet_syc_circuit = cg.optimized_for_sycamore(circuit=calibrated_circuit.circuit,
                #                                                          new_device=device)
            else:
                full_circuit = echo_circuit + measurement_circuit
                circuits_list.append(full_circuit)

        results_list = run_circuit(circuits_list, no_counts, sim_mode, two_qubit_gate, processor, floquet_calibration,
                                   qubits)
        for t in range(0, 30):  # t_max):
            results = results_list[t]
            if sim_mode == 'engine':
                z_counts = results.histogram(key='x')
            elif sim_mode == 'cirq':
                z_counts = results[0].histogram(key='x')
            else:
                raise ValueError('Not a valid sim mode!')

            results_dict = {}
            if observables == "only_x":
                results_dict.update({"All_X": z_counts})
            elif observables == "only_z":
                results_dict.update({"All_Z": z_counts})
            else:
                raise ValueError("Not a valid measurement basis!")

            circuit_parameters = (dim, len(qubits), t_max, initial_site_indices, no_counts,
                                  experimental_repetitions, rule, activation_unitaries, sim_mode, two_qubit_gate,
                                  processor, observables, floquet_calibration, r, t)

            root_dir = os.getcwd()
            top_dir = "{}d".format(circuit_parameters[0])
            bot_dir = "data_qb{}_tm{}_isi{}_nc{}_er{}_ru{}_au{}_{}_{}_{}_{}_fc{}".format(
                *circuit_parameters[1:len(circuit_parameters
                                          ) - 2])
            dir_path = os.path.join(root_dir, top_dir, bot_dir)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            with open(dir_path + '/rep{}_ts{}.json'.format(*circuit_parameters[-2:]), 'w') as f:
                json.dump(results_dict, f)

            ### PROBABLY DON'T NEED FULL QOT!!
            ### CAN JUST CONSTRUCT ALL MEASUREMENT CIRCUITS IN HERE FOR Z-BASIS
            # quantum_overlapping_tomography(qubits=qubits, base_circuit=base_circuit,
            #                                circuit_parameters=circuit_parameters, no_counts=no_counts,
            #                                sim_mode=sim_mode, two_qubit_gate=two_qubit_gate, processor=processor,
            #                                observables=observables)


def run_circuit(circuits: list, rep: int, sim_mode: str, two_qubit_gate: str, processor: str,
                floquet_calibration: bool, qubits: list):
    for ii, circuit in enumerate(circuits):
        if ii == 1:
            print('\nOriginal circuit\n', circuit)
        if two_qubit_gate == "CZ":
            cg.ConvertToXmonGates().optimize_circuit(circuit)  # Use for CH
            gate_set = cg.XMON
        elif two_qubit_gate == "Sycamore":
            cg.ConvertToSycamoreGates().optimize_circuit(circuit)  # Use for C\sqrt{H}
            gate_set = cg.SYC_GATESET
        elif two_qubit_gate == "root_iSWAP":
            cg.ConvertToSqrtIswapGates().optimize_circuit(circuit)  # Use for EAP tests
            # Need to be able to insert spin echo

            # cirq.MergeSingleQubitGates().optimize_circuit(circuit)
            # cirq.ExpandComposite().optimize_circuit(circuit)
            # cirq.EjectZ().optimize_circuit(circuit)
            # cg.ConvertToSqrtIswapGates().optimize_circuit(circuit)  # Use for EAP tests
            gate_set = cg.SQRT_ISWAP_GATESET
        elif two_qubit_gate == "parasitic_root_iSWAP":
            # cg.ConvertToSqrtIswapGates().optimize_circuit(circuit)  # Use for EAP tests
            # Need to be able to insert spin echo

            # cirq.MergeSingleQubitGates().optimize_circuit(circuit)
            # cirq.ExpandComposite().optimize_circuit(circuit)
            # cirq.EjectZ().optimize_circuit(circuit)
            # cg.ConvertToSqrtIswapGates().optimize_circuit(circuit)  # Use for EAP tests
            gate_set = cg.SQRT_ISWAP_GATESET
        else:
            raise ValueError("Not a valid gate set.")

    print("\nChecking this is still correct circuit\n", circuits[1])

    if sim_mode == 'engine':
        # Create an Engine object.  This uses the project id of your
        # Google cloud project.
        project_id = ''
        engine = cg.Engine(project_id=project_id)
        processor_object = engine.get_processor(processor)
        device = processor_object.get_device([cg.SQRT_ISWAP_GATESET])

        print("Uploading program and scheduling job on Quantum Engine...\n")

        results = engine.run_batch(
            programs=circuits,
            repetitions=rep,
            processor_ids=[processor],
            gate_set=gate_set)  # Other option XMON for CZ gates, SYC_GATESET

        # print("Scheduled. View the job at: https://console.cloud.google.com/quantum/"
        #       f"programs/{results.program_id}/jobs/{results.job_id}"
        #       f"/overview?project={project_id}")

    elif sim_mode == 'cirq':  # and processor == 'NA':
        results = cirq.Simulator().run_batch(
            circuits, repetitions=rep)
    else:
        raise ValueError('No such simulation mode')
    return results


def quantum_overlapping_tomography(qubits, base_circuit, circuit_parameters,
                                   no_counts, sim_mode, two_qubit_gate, processor, observables):
    # NOTE THAT THIS FUNCTION COULD BE MORE EFFICIENT!!
    # CAN USE SWEEPS WITH PARAMETRIZED PHXPOW GATES TO GET MEASUREMENTS IN DIFFERENT BASES

    n = len(qubits)  # Number of qubits
    q = int(np.ceil(np.log2(n)))  # Number of perfect has functions
    # print('q', q)
    bin_fmt = '{0:0' + str(q) + 'b}'  # Correct binary format

    results_dict = {}

    # Reusable Items
    rot_from_x = cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=0.5)
    rot_from_y = cirq.PhasedXPowGate(phase_exponent=0.0, exponent=0.5)
    tmp_circuit_2 = cirq.Circuit()
    tmp_circuit_2.append(cirq.measure(*qubits, key='x'))

    # First need to measure all qubits in X, Y, and Z basis

    if observables == 'only_z':
        # Z
        result = run_circuit(base_circuit + tmp_circuit_2, no_counts, sim_mode, two_qubit_gate, processor, qubits)
        results_dict.update({"All_Z": result.histogram(key='x')})

    elif observables == 'mutual_information':

        # X
        tmp_circuit_1 = cirq.Circuit()
        tmp_circuit_1.append(rot_from_x(q) for q in qubits)
        result = run_circuit(base_circuit + tmp_circuit_1 + tmp_circuit_2, no_counts, sim_mode, two_qubit_gate,
                             processor, qubits)
        results_dict.update({"All_X": result.histogram(key='x')})

        # Y
        tmp_circuit_1 = cirq.Circuit()
        tmp_circuit_1.append(rot_from_y(q) for q in qubits)
        result = run_circuit(base_circuit + tmp_circuit_1 + tmp_circuit_2, no_counts, sim_mode, two_qubit_gate,
                             processor, qubits)
        results_dict.update({"All_Y": result.histogram(key='x')})

        # Z
        result = run_circuit(base_circuit + tmp_circuit_2, no_counts, sim_mode, two_qubit_gate, processor, qubits)
        results_dict.update({"All_Z": result.histogram(key='x')})

        # Now with ze perfect hash functions.
        # Will probably need to reconstruct hash partitions in analysis script.
        for i in range(0, q):  # i-1 from paper
            red_qubits = []
            blue_qubits = []
            for j in range(0, n):  # This is j-1 in paper
                bit_string = bin_fmt.format(int(j))
                if bit_string[i] == '0':
                    red_qubits.append(qubits[j])
                else:
                    blue_qubits.append(qubits[j])

            # Now have a partition for given hash function.
            # Need to measure all 6 off-diagonal correlation functions.

            # XY
            tmp_circuit_1 = cirq.Circuit()
            tmp_circuit_1.append(rot_from_x(q) for q in red_qubits)
            tmp_circuit_1.append(rot_from_y(q) for q in blue_qubits)
            result = run_circuit(base_circuit + tmp_circuit_1 + tmp_circuit_2, no_counts, sim_mode, two_qubit_gate,
                                 processor, qubits)
            results_dict.update({"f" + str(i) + "_XY": result.histogram(key='x')})

            # YX
            tmp_circuit_1 = cirq.Circuit()
            tmp_circuit_1.append(rot_from_x(q) for q in blue_qubits)
            tmp_circuit_1.append(rot_from_y(q) for q in red_qubits)
            result = run_circuit(base_circuit + tmp_circuit_1 + tmp_circuit_2, no_counts, sim_mode, two_qubit_gate,
                                 processor, qubits)
            results_dict.update({"f" + str(i) + "_YX": result.histogram(key='x')})

            # XZ
            tmp_circuit_1 = cirq.Circuit()
            tmp_circuit_1.append(rot_from_x(q) for q in red_qubits)
            result = run_circuit(base_circuit + tmp_circuit_1 + tmp_circuit_2, no_counts, sim_mode, two_qubit_gate,
                                 processor, qubits)
            results_dict.update({"f" + str(i) + "_XZ": result.histogram(key='x')})

            # ZX
            tmp_circuit_1 = cirq.Circuit()
            tmp_circuit_1.append(rot_from_x(q) for q in blue_qubits)
            result = run_circuit(base_circuit + tmp_circuit_1 + tmp_circuit_2, no_counts, sim_mode, two_qubit_gate,
                                 processor, qubits)
            results_dict.update({"f" + str(i) + "_ZX": result.histogram(key='x')})

            # YZ
            tmp_circuit_1 = cirq.Circuit()
            tmp_circuit_1.append(rot_from_y(q) for q in red_qubits)
            result = run_circuit(base_circuit + tmp_circuit_1 + tmp_circuit_2, no_counts, sim_mode, two_qubit_gate,
                                 processor, qubits)
            results_dict.update({"f" + str(i) + "_YZ": result.histogram(key='x')})

            # ZY
            tmp_circuit_1 = cirq.Circuit()
            tmp_circuit_1.append(rot_from_y(q) for q in blue_qubits)
            result = run_circuit(base_circuit + tmp_circuit_1 + tmp_circuit_2, no_counts, sim_mode, two_qubit_gate,
                                 processor, qubits)
            results_dict.update({"f" + str(i) + "_ZY": result.histogram(key='x')})

    else:
        raise ValueError("Not a valid set of observables.")

    # Now need to save everything, probably as json, maybe pickle?
    # Also test to see if works.
    # Note that it will save as dictionary of Counters but will
    # subsequently load as a dictionary of dictionaries.

    root_dir = os.getcwd()
    top_dir = "{}d".format(circuit_parameters[0])
    bot_dir = "data_qb{}_tm{}_isi{}_nc{}_er{}_ru{}_au{}_{}_{}_{}_{}".format(*circuit_parameters[1:len(circuit_parameters
                                                                                                      ) - 2])
    dir_path = os.path.join(root_dir, top_dir, bot_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(dir_path + '/rep{}_ts{}.json'.format(*circuit_parameters[-2:]), 'w') as f:
        json.dump(results_dict, f)


def construct_1d_base_circuit_instance(qubits, evolution_steps, initial_site_indices, rule, activation_unitaries,
                                       two_qubit_gate, parasitic_cphase, characterization_data):
    if rule == "T6" and two_qubit_gate == "CZ":
        layer = one_1d_t6_step
    elif rule == "T6" and two_qubit_gate == "root_iSWAP":
        layer = one_1d_t6_root_iswap_step
    elif rule == "T6" and two_qubit_gate == "parasitic_root_iSWAP":
        layer = one_1d_t6_parasitic_root_iswap_step
    else:
        raise ValueError("Not a valid rule number.")

    c0 = cirq.Circuit()
    c0.append(cirq.X(qubits[index]) for index in initial_site_indices)
    circuit = cirq.Circuit()
    for i in range(0, evolution_steps):
        circuit.append(layer(qubits=qubits, characterization_data=characterization_data))
    return c0 + circuit


def one_1d_t6_parasitic_root_iswap_step(qubits, characterization_data):
    # This is a CZ -> \sqrt{iSWAP} decomposition assuming a parasitic \np.pi/23 cphase in the \sqrt{iSWAP}
    # I guess start with phi = 0 as base instantiation and then modify after floquet?
    # Parasitic cphase is nominally ~pi/23.
    if characterization_data is None:
        correct_cz = CorrCPhase(np.pi, np.pi / 4., 0., 0., 0., 0.)
        yield [cirq.ry(rads=-np.pi / 4.)(qubits[i]) for i in range(0, len(qubits), 2)] + \
              [cirq.I(qubits[i]) for i in range(1, len(qubits), 2)]
        yield [correct_cz._decompose_([qubits[i], qubits[i + 1]]) for i in range(0, len(qubits) - 1, 2)]
        yield [correct_cz._decompose_([qubits[i], qubits[i + 1]]) for i in range(1, len(qubits), 2)]
        yield [cirq.ry(rads=np.pi / 4.)(qubits[i]) for i in range(0, len(qubits), 2)] + \
              [cirq.ry(rads=-np.pi / 4.)(qubits[i]) for i in range(1, len(qubits), 2)]
        yield [correct_cz._decompose_([qubits[i], qubits[i + 1]]) for i in range(1, len(qubits), 2)]
        yield [correct_cz._decompose_([qubits[i], qubits[i + 1]]) for i in range(0, len(qubits) - 1, 2)]
        yield [cirq.ry(rads=np.pi / 4.)(qubits[i]) for i in range(1, len(qubits), 2)] + \
              [cirq.I(qubits[i]) for i in range(0, len(qubits), 2)]

    elif characterization_data is not None:
        all_pair_calibration_dictionary = {}
        for ii in range(0, len(characterization_data)):
            for (pair, parameters) in characterization_data[ii].parameters.items():
                all_pair_calibration_dictionary[pair] = parameters
                all_pair_calibration_dictionary[(pair[1], pair[0])] = parameters
        # print(all_pair_calibration_dictionary)
        yield [cirq.ry(rads=-np.pi / 4.)(qubits[i]) for i in range(0, len(qubits), 2)] + \
              [cirq.I(qubits[i]) for i in range(1, len(qubits), 2)]
        yield [
            CorrCPhase(np.pi, np.pi / 4., 0., 0., 0., all_pair_calibration_dictionary[(qubits[i], qubits[i + 1])].phi).
                _decompose_([qubits[i], qubits[i + 1]]) for i in range(0, len(qubits) - 1, 2)]
        yield [
            CorrCPhase(np.pi, np.pi / 4., 0., 0., 0., all_pair_calibration_dictionary[(qubits[i], qubits[i + 1])].phi).
                _decompose_([qubits[i], qubits[i + 1]]) for i in range(1, len(qubits), 2)]
        yield [cirq.ry(rads=np.pi / 4.)(qubits[i]) for i in range(0, len(qubits), 2)] + \
              [cirq.ry(rads=-np.pi / 4.)(qubits[i]) for i in range(1, len(qubits), 2)]
        yield [
            CorrCPhase(np.pi, np.pi / 4., 0., 0., 0., all_pair_calibration_dictionary[(qubits[i], qubits[i + 1])].phi).
                _decompose_([qubits[i], qubits[i + 1]]) for i in range(1, len(qubits), 2)]
        yield [
            CorrCPhase(np.pi, np.pi / 4., 0., 0., 0., all_pair_calibration_dictionary[(qubits[i], qubits[i + 1])].phi).
                _decompose_([qubits[i], qubits[i + 1]]) for i in range(0, len(qubits) - 1, 2)]
        yield [cirq.ry(rads=np.pi / 4.)(qubits[i]) for i in range(1, len(qubits), 2)] + \
              [cirq.I(qubits[i]) for i in range(0, len(qubits), 2)]
    else:
        raise ValueError("Not valid characterization data!")
    return


def one_1d_t6_root_iswap_step(qubits, activation_unitaries):
    # yield [cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.75, z_exponent=-0.5)(qubits[i])
    #        for i in range(0, len(qubits) - 1, 2)] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=0.5)(qubits[i])
    #        for i in range(1, len(qubits) - 1, 2)] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.25, z_exponent=0.)(qubits[len(qubits) - 1])]

    yield [cirq.ry(rads=-np.pi / 4.)(qubits[i]) for i in range(0, len(qubits), 2)] + \
          [cirq.I(qubits[i]) for i in range(1, len(qubits), 2)]
    yield [cirq.rz(rads=np.pi / 2.)(qubits[i]) for i in range(0, len(qubits) - 1)] + [cirq.X(qubits[len(qubits) - 1])]
    yield [cirq.rx(rads=-np.pi / 2.)(qubits[i]) for i in range(0, len(qubits) - 1, 2)] + \
          [cirq.rx(rads=np.pi / 2.)(qubits[i]) for i in range(1, len(qubits) - 1, 2)] + [
              cirq.X(qubits[len(qubits) - 1])]

    yield [cirq.ISwapPowGate(exponent=-0.5)(qubits[i], qubits[i + 1]) for i in range(0, len(qubits) - 1, 2)] + \
          [cirq.I(qubits[len(qubits) - 1])]
    # yield [cirq.PhasedXZGate(axis_phase_exponent=-1., x_exponent=1., z_exponent=0.)(qubits[i])
    #        for i in range(1, len(qubits) - 1, 2)] + [cirq.I(qubits[len(qubits) - 1])]
    yield [cirq.rx(rads=-np.pi)(qubits[i]) for i in range(1, len(qubits) - 1, 2)] + [cirq.X(qubits[len(qubits) - 1])]
    yield [cirq.ISwapPowGate(exponent=0.5)(qubits[i], qubits[i + 1]) for i in range(0, len(qubits) - 1, 2)] + \
          [cirq.I(qubits[len(qubits) - 1])]

    # yield [cirq.PhasedXZGate(axis_phase_exponent=0., x_exponent=0.5, z_exponent=0.)(qubits[0])] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.)(qubits[i])
    #        for i in range(1, len(qubits)-1, 2)] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0.)(qubits[i])
    #        for i in range(2, len(qubits) - 1, 2)] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0.5)(qubits[len(qubits)-1])]

    yield [cirq.rx(rads=np.pi / 2.)(qubits[i]) for i in range(0, len(qubits) - 1)] + [cirq.X(qubits[len(qubits) - 1])]
    yield [cirq.rz(rads=np.pi / 2.)(qubits[i]) for i in range(1, len(qubits))] + [cirq.X(qubits[0])]
    yield [cirq.rx(rads=-np.pi / 2.)(qubits[i]) for i in range(2, len(qubits), 2)] + \
          [cirq.rx(rads=np.pi / 2.)(qubits[i]) for i in range(1, len(qubits), 2)] + [cirq.X(qubits[0])]

    yield [cirq.ISwapPowGate(exponent=-0.5)(qubits[i], qubits[i + 1]) for i in range(1, len(qubits), 2)] + \
          [cirq.I(qubits[0])]
    # yield [cirq.PhasedXZGate(axis_phase_exponent=-1., x_exponent=1., z_exponent=0.)(qubits[i])
    #        for i in range(1, len(qubits) - 1, 2)] + [cirq.I(qubits[0])]
    yield [cirq.rx(rads=-np.pi)(qubits[i]) for i in range(1, len(qubits), 2)] + [cirq.X(qubits[0])]
    yield [cirq.ISwapPowGate(exponent=0.5)(qubits[i], qubits[i + 1]) for i in range(1, len(qubits), 2)] + \
          [cirq.I(qubits[0])]

    # yield [cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.25, z_exponent=0.)(qubits[0])] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=0.25, x_exponent=0.5, z_exponent=0.25)(qubits[i])
    #        for i in range(1, len(qubits), 2)] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=-0.25, x_exponent=0.5, z_exponent=0.75)(qubits[i])
    #        for i in range(2, len(qubits), 2)]

    yield [cirq.rx(rads=np.pi / 2.)(qubits[i]) for i in range(1, len(qubits))] + [cirq.X(qubits[0])]
    yield [cirq.ry(rads=np.pi / 4.)(qubits[i]) for i in range(0, len(qubits), 2)] + \
          [cirq.ry(rads=-np.pi / 4.)(qubits[i]) for i in range(1, len(qubits), 2)]
    yield [cirq.rz(rads=np.pi / 2.)(qubits[i]) for i in range(1, len(qubits))] + [cirq.X(qubits[0])]
    yield [cirq.rx(rads=np.pi / 2.)(qubits[i]) for i in range(2, len(qubits), 2)] + \
          [cirq.rx(rads=-np.pi / 2.)(qubits[i]) for i in range(1, len(qubits), 2)] + [cirq.X(qubits[0])]

    yield [cirq.ISwapPowGate(exponent=-0.5)(qubits[i], qubits[i + 1]) for i in range(1, len(qubits), 2)] + \
          [cirq.I(qubits[0])]
    # yield [cirq.PhasedXZGate(axis_phase_exponent=-1., x_exponent=1., z_exponent=0.)(qubits[i])
    #        for i in range(2, len(qubits), 2)] + [cirq.I(qubits[0])]
    yield [cirq.rx(rads=-np.pi)(qubits[i]) for i in range(2, len(qubits), 2)] + [cirq.X(qubits[0])]
    yield [cirq.ISwapPowGate(exponent=0.5)(qubits[i], qubits[i + 1]) for i in range(1, len(qubits), 2)] + \
          [cirq.I(qubits[0])]

    # yield [cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=0.5)(qubits[0])] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=0.5, z_exponent=0.5)(qubits[i])
    #        for i in range(1, len(qubits), 2)] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0.5, z_exponent=-1.)(qubits[i])
    #        for i in range(2, len(qubits)-1, 2)] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=0., x_exponent=0.5, z_exponent=0.)(qubits[len(qubits)-1])]

    yield [cirq.rx(rads=np.pi / 2.)(qubits[i]) for i in range(1, len(qubits))] + [cirq.X(qubits[0])]
    yield [cirq.rz(rads=np.pi / 2.)(qubits[i]) for i in range(0, len(qubits) - 1)] + [cirq.X(qubits[len(qubits) - 1])]
    yield [cirq.rx(rads=np.pi / 2.)(qubits[i]) for i in range(0, len(qubits) - 1, 2)] + \
          [cirq.rx(rads=-np.pi / 2.)(qubits[i]) for i in range(1, len(qubits) - 1, 2)] + \
          [cirq.X(qubits[len(qubits) - 1])]

    yield [cirq.ISwapPowGate(exponent=-0.5)(qubits[i], qubits[i + 1]) for i in range(0, len(qubits) - 1, 2)] + \
          [cirq.I(qubits[len(qubits) - 1])]
    # yield [cirq.PhasedXZGate(axis_phase_exponent=-1., x_exponent=1., z_exponent=0.)(qubits[i])
    #        for i in range(0, len(qubits)-1, 2)] + [cirq.I(qubits[len(qubits)-1])]
    yield [cirq.rx(rads=-np.pi)(qubits[i]) for i in range(0, len(qubits) - 1, 2)] + [cirq.X(qubits[len(qubits) - 1])]
    yield [cirq.ISwapPowGate(exponent=0.5)(qubits[i], qubits[i + 1]) for i in range(0, len(qubits) - 1, 2)] + \
          [cirq.I(qubits[len(qubits) - 1])]

    # yield [cirq.PhasedXZGate(axis_phase_exponent=0., x_exponent=0.5, z_exponent=0.)(qubits[i])
    #        for i in range(0, len(qubits)-1, 2)] + \
    #       [cirq.PhasedXZGate(axis_phase_exponent=0.25, x_exponent=0.5, z_exponent=-0.25)(qubits[i])
    #        for i in range(1, len(qubits), 2)] + [cirq.I(qubits[len(qubits) - 1])]

    yield [cirq.rx(rads=np.pi / 2.)(qubits[i]) for i in range(0, len(qubits) - 1)] + [cirq.X(qubits[len(qubits) - 1])]
    yield [cirq.ry(rads=np.pi / 4.)(qubits[i]) for i in range(1, len(qubits), 2)] + \
          [cirq.I(qubits[i]) for i in range(0, len(qubits), 2)]
    return


def efficient_hadamard_t6_gate_root_iswap_sequence(ctrl1, tgt, ctrl2):
    # After simplifying both CH together
    yield cirq.ry(rads=-np.pi / 4.)(tgt)
    yield cirq.CZ(ctrl2, tgt)
    yield cirq.CZ(ctrl1, tgt)
    yield cirq.ry(rads=np.pi / 4.)(tgt)
    return


def efficient_controlled_hadamard_root_iswap(ctrl, tgt):
    # From Zhang's hint.
    yield cirq.ry(rads=-np.pi / 4.)(tgt)
    yield cirq.CZ(ctrl, tgt)
    yield cirq.ry(rads=np.pi / 4.)(tgt)
    return


def one_1d_t6_step(qubits, activation_unitaries):
    if activation_unitaries[0] == "H":
        neighborhood_gate_1 = efficient_hadamard_t6_gate_sequence
        half_neighborhood_gate_1 = efficient_controlled_hadamard
    else:
        raise ValueError("Not a valid activation operator")

    for i in range(0, len(qubits), 2):  # Even qubits
        if i == 0:
            yield half_neighborhood_gate_1(ctrl=qubits[i + 1], tgt=qubits[i])
        elif i == len(qubits) - 1:
            yield half_neighborhood_gate_1(ctrl=qubits[i - 1], tgt=qubits[i])
        else:
            yield neighborhood_gate_1(ctrl1=qubits[i - 1], tgt=qubits[i], ctrl2=qubits[i + 1])

    for i in range(1, len(qubits), 2):  # Odd qubits
        if i == 0:
            yield half_neighborhood_gate_1(ctrl=qubits[i + 1], tgt=qubits[i])
        elif i == len(qubits) - 1:
            yield half_neighborhood_gate_1(ctrl=qubits[i - 1], tgt=qubits[i])
        else:
            yield neighborhood_gate_1(ctrl1=qubits[i - 1], tgt=qubits[i], ctrl2=qubits[i + 1])
    return


def efficient_controlled_controlled_hadamard(ctrl1, ctrl2, tgt):
    yield cirq.ry(rads=-np.pi / 4.)(tgt)
    yield cirq.CCZ(ctrl1, ctrl2, tgt)
    yield cirq.ry(rads=np.pi / 4.)(tgt)


def efficient_controlled_hadamard(ctrl, tgt):
    # From Zhang's hint.
    yield cirq.ry(rads=-np.pi / 4.)(tgt)
    yield cirq.CZ(ctrl, tgt)
    yield cirq.ry(rads=np.pi / 4.)(tgt)


def efficient_hadamard_t6_gate_sequence(ctrl1, tgt, ctrl2):
    # After simplifying both CH together
    yield cirq.ry(rads=-np.pi / 4.)(tgt)
    yield cirq.CZ(ctrl2, tgt)
    yield cirq.CZ(ctrl1, tgt)
    yield cirq.ry(rads=np.pi / 4.)(tgt)


class CorrCPhase(cirq.ops.gate_features.TwoQubitGate):
    r"""Implement a C-Phase gate using two fsim gates. This only works for
    the sqrt_iswap gate

    Input unitary (fsim):
    [[1,  0,   0,   0],
     [0,  c, -is,   0],
     [0, -is,  c,   0],
     [0,  0,   0,   e^{-i phi}]]
    where c = cos(theta) and s = sin(theta)

    Output unitary (c-phase):
    [[1   0   0   0],
     [0   1   0   0],
     [0   0   1   0],
     [0   0   0   e^{-i cphase}]]

    """

    def __init__(self,
                 cphase: float,
                 theta: float,
                 delta: float,
                 chi: float,
                 gamma: float,
                 phi: float,
                 sin_alpha_tol: float = 0.0,
                 test: bool = False):
        self.cphase = cphase
        self.theta = theta
        self.delta = delta
        self.chi = chi
        self.gamma = gamma
        self.phi = phi
        self.sin_alpha_tol = sin_alpha_tol
        self.test = test

    def _decompose_(self, qubits) -> cirq.OP_TREE:
        a, b = qubits

        temp = (np.sin(self.cphase / 4) ** 2 - np.sin(self.phi / 2) ** 2) / \
               (np.sin(self.theta) ** 2 - np.sin(self.phi / 2) ** 2)
        if temp < 0.0 and np.isclose(temp, 0.0, atol=1e-3):
            temp = 0.0
        elif 1.0 < temp < 1.0 + self.sin_alpha_tol:
            temp = 1.0

        if 0 <= temp <= 1:
            alpha = np.arcsin(temp ** 0.5)
        else:
            raise RuntimeError(
                'Cannot decompose the C-phase gate on qubits {qubits} into '
                'the given fSim gates (cphase={self.cphase}, '
                'theta={self.theta}, delta={self.delta}, gamma={self.gamma}, '
                'phi={self.phi}, sin_alpha={temp})')
        beta = 0.5 * np.pi * (1 - np.sign(np.cos(0.5 * self.phi)))
        gamma = 0.5 * np.pi * (1 - np.sign(np.sin(0.5 * self.phi)))

        xi = np.arctan(np.tan(alpha) * np.cos(self.theta)
                       / np.cos(0.5 * self.phi)) + beta
        if self.cphase < 0:
            xi += np.pi

        if self.phi == 0:
            eta = 0.5 * np.sign(np.tan(alpha) * np.sin(self.theta)) * np.pi
        else:
            eta = np.arctan(np.tan(alpha) * np.sin(self.theta)
                            / np.sin(0.5 * self.phi)) + gamma

        if self.test:
            two_qubit_gate = cirq.FSimGate(self.theta, self.phi)
        else:
            two_qubit_gate = cirq.ISwapPowGate(exponent=-0.5)
            # two_qubit_gate = CorrSqrtISWP(self.delta, self.chi, self.gamma)

        # This must match the structure of EchoCPhaseGate below
        yield cirq.rz(-0. * self.cphase).on(a)
        yield cirq.rz(-0. * self.cphase).on(b)

        yield cirq.rx(xi).on(a)
        yield cirq.rx(eta).on(b)

        yield cirq.rz(0.5 * self.phi).on(a)
        yield cirq.rz(0.5 * self.phi).on(b)

        yield two_qubit_gate.on(*qubits)

        yield cirq.rx(-2 * alpha).on(a)

        yield cirq.rz(0.5 * self.phi + np.pi).on(a)
        yield cirq.rz(0.5 * self.phi).on(b)

        yield two_qubit_gate.on(*qubits)

        yield cirq.rx(-xi).on(a)
        yield cirq.rx(-eta).on(b)

        yield cirq.rz(-0.5 * self.cphase + np.pi).on(a)
        yield cirq.rz(-0.5 * self.cphase).on(b)


if __name__ == "__main__":
    main()
