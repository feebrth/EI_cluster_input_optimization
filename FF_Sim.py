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
from Exp_data import get_exp_data_ff  # Funktion zum Laden der experimentellen Daten importieren




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


# def FF_across_all_directions_and_neurons_optimized(spiketimes, timepoints, directions, direction_range, window_size=400,
#                                                    step_size=100):
#     fano_factors_across_all_windows = []
#     time_axis = np.arange(-500 + window_size / 2, 2500 - window_size / 2 + 1, step_size)
#
#     trial_start_offset = -500
#     trial_end_offset = 2500
#     unique_neurons = np.unique(spiketimes[:, 1])
#
#     # Organisiere Zeitfenster-Indizes
#     windows = np.arange(trial_start_offset, trial_end_offset - window_size + 1, step_size)
#
#     # Precompute spike counts for each neuron and direction in each window
#     for window_start in windows:
#         fano_factors_for_all_directions = []
#
#         for dir_value in direction_range:
#             relevant_trials = [timepoints[i] for i, d in enumerate(directions) if d == dir_value]
#             spike_counts_per_neuron = {neuron: [] for neuron in unique_neurons}
#
#             # Berechne Spike-Zählungen für jedes Neuron und trial
#             for neuron in unique_neurons:
#                 neuron_spikes = spiketimes[spiketimes[:, 1] == neuron][:, 0]
#
#                 for trial_start in relevant_trials:
#                     trial_window_start = trial_start + window_start
#                     trial_window_end = trial_window_start + window_size
#
#                     spike_count = np.sum((neuron_spikes >= trial_window_start) & (neuron_spikes < trial_window_end))
#                     spike_counts_per_neuron[neuron].append(spike_count)
#
#             # Berechnung der Fano-Faktoren pro Fenster und Richtung
#             fano_factors_for_window = [
#                 np.var(counts) / np.mean(counts) if len(counts) > 1 and np.mean(counts) > 0 else 0
#                 for counts in spike_counts_per_neuron.values()
#             ]
#             fano_factors_for_all_directions.append(np.mean(fano_factors_for_window))
#
#         # Durchschnitt der Fano-Faktoren über alle Richtungen
#         fano_factors_across_all_windows.append(np.mean(fano_factors_for_all_directions))
#
#     return fano_factors_across_all_windows, time_axis


def FF_across_all_directions_and_neurons_optimized(spiketimes, timepoints, directions, direction_range, window_size=400,
                                                   step_size=100):
    fano_factors_across_all_windows = []
    time_axis = np.arange(-500 + window_size / 2, 2500 - window_size / 2 + 1, step_size)

    trial_start_offset = -500
    trial_end_offset = 2500
    unique_neurons = np.unique(spiketimes[:, 1])

    # Organisiere Zeitfenster-Indizes
    windows = np.arange(trial_start_offset, trial_end_offset - window_size + 1, step_size)

    # Precompute spike counts for each neuron and direction in each window
    for window_start in windows:
        fano_factors_for_all_directions = []

        for dir_value in direction_range:
            relevant_trials = [timepoints[i] for i, d in enumerate(directions) if d == dir_value]

            spike_counts_per_neuron = {neuron: [] for neuron in unique_neurons}

            # Berechne Spike-Zählungen für jedes Neuron und trial
            for neuron in unique_neurons:
                neuron_spikes = spiketimes[spiketimes[:, 1] == neuron][:, 0]

                for trial_start in relevant_trials:
                    trial_window_start = trial_start + window_start
                    trial_window_end = trial_window_start + window_size

                    spike_count = np.sum((neuron_spikes >= trial_window_start) & (neuron_spikes < trial_window_end))
                    spike_counts_per_neuron[neuron].append(spike_count)

            # Berechnung der Fano-Faktoren pro Fenster und Richtung
            fano_factors_for_window = [
                np.var(counts) / np.mean(counts) if len(counts) > 1 and np.mean(counts) > 0 else 0
                for counts in spike_counts_per_neuron.values()
            ]
            fano_factors_for_all_directions.append(np.mean(fano_factors_for_window))

        # Durchschnitt der Fano-Faktoren über alle Richtungen
        fano_factors_across_all_windows.append(np.mean(fano_factors_for_all_directions))

    return fano_factors_across_all_windows, time_axis


def plot_fano_factors(avg_fano_factor_curve, time_axis, exp_time, exp_ff):
    plt.figure()
    plt.plot(time_axis, avg_fano_factor_curve, label='Simulated Fano Factor')
    plt.plot(exp_time, exp_ff, label='Experimental Fano Factor', linestyle='--')
    plt.xlabel("Time (ms)")
    plt.ylabel("Fano Factor")
    plt.title("Fano Factor Comparison: Simulation vs Experiment")
    plt.legend()
    plt.savefig("FF_comparison.png")




def plot_stimulus_kernel(stim_kernel, kernel_step):
    """Plots the given stimulus kernel as a bar chart with a time axis."""
    time_points = np.arange(0, len(stim_kernel) * kernel_step, kernel_step)  # Zeitpunkte in ms für jeden Stimulus

    plt.figure(figsize=(10, 6))
    plt.bar(time_points, stim_kernel, width=kernel_step, color='skyblue', align='edge')
    plt.xlabel("Time (ms)")
    plt.ylabel("Stimulus Amplitude")
    plt.title("Stimulus Kernel Amplitudes Over Time")




def simulate_model(experimental_trials, direction_range, stim_kernel, kernel_step, plot=False):
    timepoints, direction = generate_trials(experimental_trials, direction_range)
    stim_dicts = stim_amplitudes(timepoints, direction, stim_kernel, kernel_step)
    stim_dict['experiment_stim'] = stim_dicts
    sim_dict['simtime'] = max(timepoints) + 3000

    ei_network = network.ClusteredNetwork(sim_dict, net_dict, stim_dict)
    result = ei_network.get_simulation()

    spiketimes = result["spiketimes"].T
    window_size = 400

    # Fano-Faktor berechnen
    avg_fano_factor_curve, time_axis = FF_across_all_directions_and_neurons_optimized(spiketimes, timepoints, direction, direction_range)

    # Experimentelle Daten laden und plotten
    exp_time, exp_ff = get_exp_data_ff(1)
    plot_fano_factors(avg_fano_factor_curve, time_axis, exp_time, exp_ff)

    return None



# Run the simulation
if __name__ == "__main__":

    stim_kernel = np.array([0.17312816364086267, 0.8259290941384977, 0.44182977675232843,  0.3558826809260373,
                            0.2856217738814025,  0.3040297739070603,  0.2366001890107655,  0.2472914682618193,
                            0.3150498635199546, 0.3790337694745662, 0.36722502142689295, 0.39463791746376475,
                            0.2633404217765265,  0.32753879091207094,  0.12843437450070974,  0.0,  0.0,  0.0,
                            0.3216556491748419,  0.49706313013206366,  1.0,  1.0,  1.0,  1.0])
    kernel_step = 2000 // len(stim_kernel)  # 167 ms pro Stimulus

    simulate_model(
        experimental_trials= 5,  # Anzahl der Trials
        direction_range=[0, 1, 2],
        stim_kernel=stim_kernel,
        kernel_step=kernel_step,
        plot=True
    )

    # Stimulus-Kernel plotten
    plot_stimulus_kernel(stim_kernel, kernel_step)


