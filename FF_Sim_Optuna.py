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
import optuna


# Zugriff auf die Optuna-Studie
storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
study_name = "GP_24"
study = optuna.load_study(study_name=study_name, storage=storage_url)

# Hole die besten Parameter aus der Datenbank
best_trial = study.best_trial
best_params = best_trial.params
num_stimuli = 24
stim_kernel = [best_params[f'stimulus{i+1}'] for i in range(num_stimuli)]
kernel_step = 2000 // num_stimuli  # Zeitabstand zwischen Stimuli in ms



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
    plt.savefig("Stim_amplitudes")




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
    avg_fano_factor_curve, time_axis = FF_across_all_directions_and_neurons(spiketimes, timepoints, direction, direction_range)

    # Experimentelle Daten laden und plotten
    exp_time, exp_ff = get_exp_data_ff(1)
    plot_fano_factors(avg_fano_factor_curve, time_axis, exp_time, exp_ff)

    return None

    # Get experimental Fano Factors
   # time, exp_fano_factors = get_exp_data_ff(1)  # Ensure this returns a dictionary of Fano factors



# Run the simulation

# Run the simulation
if __name__ == "__main__":
    simulate_model(
        experimental_trials= 60,  # Anzahl der Trials
        direction_range=[0, 1, 2],
        stim_kernel=stim_kernel,
        kernel_step=kernel_step,
        plot=True
    )

    # Stimulus-Kernel plotten
    plot_stimulus_kernel(stim_kernel, kernel_step)
