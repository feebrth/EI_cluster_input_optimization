# Importing necessary libraries
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

# Importing custom modules
import network_EI as network
from helper_EI import raster_plot
from network_params_EI import net_dict
from sim_params_EI import sim_dict
from stimulus_params_EI import stim_dict
from Exp_data import get_exp_data_ff

# Section 1: Data Generation Functions
def generate_trials(num_trials, direction_range, start=2000, step=3000, variation=200):
    timepoints = []
    direction = []
    time = start

    for _ in range(num_trials):
        timepoints.append(time)
        direction.append(random.choice(direction_range))
        random_step = step + random.randint(0, variation)
        time += random_step

    return timepoints, direction

def stim_amplitudes(timepoints, direction, kernel, kernel_step):
    stim_dicts = {}
    for dir_value in set(direction):
        stim_times = []
        stim_amps = []

        for i, time in enumerate(timepoints):
            if direction[i] == dir_value:
                for k in range(len(kernel)):
                    stim_times.append(time + k * kernel_step)
                stim_times.append(time + len(kernel) * kernel_step)
                stim_amps.extend(kernel)
                stim_amps.append(0)

        stim_dicts[dir_value] = {'stim_time': stim_times, 'stim_amps': stim_amps}

    return stim_dicts


import numpy as np

def FF_across_all_directions_and_neurons(spiketimes, timepoints, directions, direction_range, window_size=400, step_size=100):
    fano_factors_across_all_windows = []  # Speichert den Durchschnitt des Fano-Faktors für jedes Zeitfenster
    time_axis = None  # Initialize to None; will be set only once

    # Define offsets for each trial start and end
    trial_start_offset = -500
    trial_end_offset = 2500

    # Precompute unique neuron IDs outside the loop for efficiency
    unique_neurons = np.unique(spiketimes[:, 1])

    for window_start in np.arange(trial_start_offset, trial_end_offset - window_size + 1, step_size):
        fano_factors_for_all_directions = []  # Speichert Fano-Faktoren für alle Richtungen in diesem Fenster

        for dir_value in direction_range:
            # Retrieve start times of trials for the current direction
            relevant_trials = [timepoints[i] for i, d in enumerate(directions) if d == dir_value]

            # Für jedes Neuron, sammeln wir die Spike-Zählungen über alle Trials
            spike_counts_per_neuron = {neuron: [] for neuron in unique_neurons}

            for trial_start in relevant_trials:
                # Define actual window start and end for each trial
                trial_window_start = trial_start + window_start
                trial_window_end = trial_window_start + window_size

                for neuron in unique_neurons:
                    # Calculate spike count in the window for the current neuron and trial
                    spike_count = np.histogram(
                        spiketimes[(spiketimes[:, 1] == neuron) &
                                   (spiketimes[:, 0] >= trial_window_start) &
                                   (spiketimes[:, 0] < trial_window_end)],
                        bins=1, range=(trial_window_start, trial_window_end)
                    )[0][0]
                    spike_counts_per_neuron[neuron].append(spike_count)

            # Berechne den Fano-Faktor für jedes Neuron in diesem Fenster über alle Trials dieser Richtung
            fano_factors_for_window = []
            for neuron, counts in spike_counts_per_neuron.items():
                # Berechne nur den Fano-Faktor, wenn Daten für mehrere Trials vorhanden sind
                if len(counts) > 1:
                    mean_spike_count = np.mean(counts)
                    var_spike_count = np.var(counts)
                    fano_factor = var_spike_count / mean_spike_count if mean_spike_count > 0 else 0
                    fano_factors_for_window.append(fano_factor)
                else:
                    fano_factors_for_window.append(0)  # Keine Spikes oder nur ein Trial

            # Durchschnitt der Fano-Faktoren über alle Neuronen in diesem Fenster und dieser Richtung
            if fano_factors_for_window:
                fano_factors_for_all_directions.append(np.mean(fano_factors_for_window))

        # Durchschnitt der Fano-Faktoren über alle Richtungen für dieses Fenster
        if fano_factors_for_all_directions:
            fano_factors_across_all_windows.append(np.mean(fano_factors_for_all_directions))

        # Set time_axis on the first pass only, using midpoints of windows
        if time_axis is None:
            time_axis = np.arange(trial_start_offset, trial_end_offset - window_size + 1, step_size) + (window_size / 2)

    return fano_factors_across_all_windows, time_axis




def plot_fano_factors(avg_fano_factor_curve, time_axis):
    plt.figure()
    plt.plot(time_axis, avg_fano_factor_curve, label='Average Fano Factor across all directions')
    plt.xlabel("Time (ms)")
    plt.ylabel("Average Fano Factor")
    plt.title("Average Fano Factor over Time")
    plt.legend()
    plt.savefig(f"FF_opti_24.png")


def plot_stimulus_kernel(stim_kernel, kernel_step):
    """Plots the given stimulus kernel as a bar chart with a time axis."""
    time_points = np.arange(0, len(stim_kernel) * kernel_step, kernel_step)  # Zeitpunkte in ms für jeden Stimulus

    plt.figure(figsize=(10, 6))
    plt.bar(time_points, stim_kernel, width=kernel_step, color='skyblue', align='edge')
    plt.xlabel("Time (ms)")
    plt.ylabel("Stimulus Amplitude")
    plt.title("Stimulus Kernel Amplitudes Over Time")
    plt.show()




# Example usage
# avg_fano_factors_over_time, time_axis = FF_per_direction(spiketimes, timepoints, directions, direction_range)
# plot_fano_factors(avg_fano_factors_over_time, time_axis)


# def align_and_trim(time_line1, ff_1, time_line2, ff_2):
#     """
#     Aligns two timelines by their zero point and trims the firing rate arrays to have matching time windows.
#
#     Inputs:
#         time_line1: Time points for the first set of firing rates
#         firing_rate1: Simulated FFs corresponding to time_line1
#         time_line2: Time points for the experimental set of FFs
#         firing_rate2: Experimental FFs corresponding to time_line2
#
#     Returns:
#         trimmed_time_1: Aligned and trimmed time points
#         trimmed_ff_1: FF for the first set, trimmed to match the common time window
#         trimmed_ff_2: FF for the second set, trimmed to match the common time window
#     """
#     idx1_zero = (np.abs(time_line1 - 0)).argmin()
#     idx2_zero = (np.abs(time_line2 - 0)).argmin()
#
#     max_left = min(idx1_zero, idx2_zero)
#     max_right = min(len(time_line1) - idx1_zero, len(time_line2) - idx2_zero)
#
#     trimmed_time_1 = time_line1[idx1_zero - max_left: idx1_zero + max_right]
#     trimmed_ff_1 = ff_1[idx1_zero - max_left: idx1_zero + max_right]
#     trimmed_ff_2 = ff_2[idx2_zero - max_left: idx2_zero + max_right]
#
#     return trimmed_time_1, trimmed_ff_1, trimmed_ff_2
#
# # Section 3: Penalty Calculation
# def calculate_penalty(avg_fano_factors_per_direction, experimental_avg_firing_rate1):
#     """
#     Calculates the penalty as the squared deviation between the simulated and experimental average firing rates.
#
#     Inputs:
#         average_firing_rate: Simulated average firing rate
#         experimental_avg_firing_rate1: Experimental average firing rate
#
#     Returns:
#         penalty: Squared deviation between the two firing rates
#     """
#     penalty = np.sum((average_firing_rate - experimental_avg_firing_rate1) ** 2)
#     return penalty
#
#
#
# def compare_fano_factors(sim_fano_factors, exp_fano_factors):
#     common_directions = set(sim_fano_factors.keys()).intersection(exp_fano_factors.keys())
#     sim_fano_list = [sim_fano_factors[direction] for direction in common_directions]
#     exp_fano_list = [exp_fano_factors[direction] for direction in common_directions]
#
#     min_len = min(len(sim_fano_list), len(exp_fano_list))
#     sim_fano_list = sim_fano_list[:min_len]
#     exp_fano_list = exp_fano_list[:min_len]
#
#     penalty = np.sum((np.array(sim_fano_list) - np.array(exp_fano_list)) ** 2)
#     return penalty

def simulate_model(experimental_trials, direction_range, stim_kernel, kernel_step, plot=False, num_stimuli=12, best_penalty=None):
    timepoints, direction = generate_trials(experimental_trials, direction_range)
    stim_dicts = stim_amplitudes(timepoints, direction, stim_kernel, kernel_step)
    stim_dict['experiment_stim'] = stim_dicts
    sim_dict['simtime'] = max(timepoints) + 3000

    ei_network = network.ClusteredNetwork(sim_dict, net_dict, stim_dict)
    result = ei_network.get_simulation()

    spiketimes = result["spiketimes"].T
    window_size = 400  # Size of the window for Fano Factor calculation

    # Berechnung des durchschnittlichen Fano-Faktors über alle Neuronen und Richtungen
    avg_fano_factor_curve, time_axis = FF_across_all_directions_and_neurons(spiketimes, timepoints, direction, direction_range)

    # Plotten
    plot_fano_factors(avg_fano_factor_curve, time_axis)

    return None



    # Get experimental Fano Factors
   # time, exp_fano_factors = get_exp_data_ff(1)  # Ensure this returns a dictionary of Fano factors

    # Compare Fano Factors
    #penalty = compare_fano_factors(sim_fano_factors_per_direction, exp_fano_factors)
    #print(f"Penalty for stimulus {stim_kernel} is: {penalty}")

    # Plot results if needed
    # if plot:
    #     ax = raster_plot(
    #         result["spiketimes"],
    #         tlim=(0, sim_dict["simtime"]),
    #         colorgroups=[
    #             ("k", 0, net_dict["N_E"]),
    #             ("darkred", net_dict["N_E"], net_dict["N_E"] + net_dict["N_I"]),
    #         ],
    #     )
    #     plt.figure()
    #     for direction, fano_value in sim_fano_factors_per_direction.items():
    #         plt.bar(direction, fano_value, color='blue', alpha=0.6,
    #                 label='Simulated Fano Factor' if direction == list(sim_fano_factors_per_direction.keys())[0] else "")
    #     for direction, fano_value in exp_fano_factors.items():
    #         plt.bar(direction, fano_value, color='red', alpha=0.6,
    #                 label='Experimental Fano Factor' if direction == list(exp_fano_factors.keys())[0] else "")
    #
    #     plt.xlabel('Direction')
    #     plt.ylabel('Fano Factor')
    #
    #     if best_penalty is None:
    #         best_penalty = penalty
    #     plt.title(f"Optimization: GP, Tested Stimuli: {num_stimuli}, Best value: {best_penalty:.4f}")
    #
    #     plt.legend()
    #     plt.savefig(f"Optimized_GP_Fano_Factor_{num_stimuli}_value_{best_penalty:.4f}.png")
    #
    # return None


# Run the simulation
if __name__ == "__main__":
    stim_kernel = np.array([0.21799755524146394, 0.7465940659178374, 0.5445673349710587, 0.3233604287043149,
                            0.33339708690904063, 0.3076231945321583, 0.3261545541369318, 0.2763110424131106,
                            0.3722823914045572, 0.40977055757756575, 0.36313810569091065, 0.38629283567642636,
                            0.3540581714914624, 0.21577296619757869, 0.18168249273716844, 0.0,
                            0.0, 0.0, 0.32404642647104503, 0.5332587489610445, 0.9032086833422716, 1.0,
                            0.9999999999999999, 1.0
                            ]),  # Stimulus-Kernel
    kernel_step = 2000 // len(stim_kernel)  # kernel step 167 ms pro stimulus


    simulate_model(
        experimental_trials= 5,  # number of trials
        direction_range=[0, 1, 2],  # direction range
        stim_kernel= stim_kernel,
        kernel_step= kernel_step,  # kernel step 167 ms pro stimulus
        plot=True
    )


    plot_stimulus_kernel(stim_kernel)

