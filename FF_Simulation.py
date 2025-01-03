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


def FF_per_direction(spiketimes, timepoints, directions, direction_range, window_size=400, step_size=100):
    avg_fano_factors_per_direction = {dir_value: [] for dir_value in direction_range} #dictionary that creates an empty list for every direction
    time_axis = None  # Initialize to None; will be set only once

    # Define offsets for each trial start and end
    trial_start_offset = -500
    trial_end_offset = 2500

    # Precompute unique neuron IDs outside the loop to improve efficiency
    unique_neurons = np.unique(spiketimes[:, 1])

    for dir_value in direction_range:
        #getting spikes for current direction
        dir_spiketimes = spiketimes[np.isin(spiketimes[:, 1], unique_neurons)]

        # Retrieve start times of trials for the current direction
        relevant_trials = [timepoints[i] for i, d in enumerate(directions) if d == dir_value]
        trial_fano_factors = []  # Store Fano factors across all trials in this direction

        for trial_start in relevant_trials: #loop goes over all trials in current direction
            # Define start times for each window within this trial
            window_start_times = np.arange(trial_start + trial_start_offset,
                                           trial_start + trial_end_offset - window_size + 1,
                                           step_size)

            # Set time_axis on the first pass only
            if time_axis is None and len(window_start_times) > 0:
                time_axis = window_start_times + (window_size / 2)

            trial_fano_factors_per_window = []  # Store Fano factors for each window in this trial

            for window_start in window_start_times:
                window_end = window_start + window_size

                # Calculate spike counts for each neuron within the current window using np.histogram
                spike_counts = np.array([
                    np.histogram(
                        dir_spiketimes[(dir_spiketimes[:, 1] == neuron) &
                                       (dir_spiketimes[:, 0] >= window_start) &
                                       (dir_spiketimes[:, 0] < window_end)],
                        bins=1, range=(window_start, window_end)
                    )[0][0]
                    for neuron in unique_neurons
                ])

                # Calculate Fano factor only for neurons with spikes in this window
                neurons_with_spikes = spike_counts[spike_counts > 0]
                if neurons_with_spikes.size > 1:
                    mean_spike_count = neurons_with_spikes.mean()
                    var_spike_count = neurons_with_spikes.var()
                    fano_factor = var_spike_count / mean_spike_count
                else:
                    fano_factor = 0  # If no spikes are present, set Fano factor to zero

                trial_fano_factors_per_window.append(fano_factor)

            # Append Fano factors for this trial to the overall list for this direction
            trial_fano_factors.append(trial_fano_factors_per_window)
            print(len(neurons_with_spikes))  # Debug print to check the count of neurons with spikes

        # Calculate average Fano factors across trials if they exist; otherwise, set zeros
        avg_fano_factors_per_direction[dir_value] = (
            np.mean(trial_fano_factors, axis=0) if trial_fano_factors
            else np.zeros_like(time_axis)
        )

    # After looping over directions: Ensure time_axis was set at least once
    if time_axis is None:
        print("Warning: time_axis was not set due to lack of relevant trials or directions.")
        time_axis = np.array([])  # Safe fallback if time_axis was never set

    return avg_fano_factors_per_direction, time_axis




def plot_fano_factors(avg_fano_factors_per_direction, time_axis):
    plt.figure()
    for direction, fano_factors in avg_fano_factors_per_direction.items():
        # Überprüfen Sie, ob fano_factors die gleiche Länge wie time_axis hat
        print(f"Direction {direction}: Länge fano_factors = {len(fano_factors)}, Länge time_axis = {len(time_axis)}")
        if len(fano_factors) == len(time_axis):  # Sicherstellen, dass die Längen übereinstimmen
            plt.plot(time_axis, fano_factors, label=f'Direction {direction}')
        else:
            print(f"Skipping direction {direction} due to length mismatch.")
    plt.xlabel("Time (ms)")
    plt.ylabel("Average Fano Factor")
    plt.title("Average Fano Factor over Time per Direction")
    plt.legend()
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

def simulate_model(experimental_trials, direction_range, stim_kernel, kernel_step, plot=False, num_stimuli=4, best_penalty=None):
    timepoints, direction = generate_trials(experimental_trials, direction_range)
    stim_dicts = stim_amplitudes(timepoints, direction, stim_kernel, kernel_step)
    stim_dict['experiment_stim'] = stim_dicts
    sim_dict['simtime'] = max(timepoints) + 3000

    ei_network = network.ClusteredNetwork(sim_dict, net_dict, stim_dict)
    result = ei_network.get_simulation()

    spiketimes = result["spiketimes"].T
    window_size = 400  # Size of the window for Fano Factor calculation

    # Innerhalb simulate_model
    avg_fano_factors_per_direction, time_axis = FF_per_direction(spiketimes, timepoints, direction, direction_range)

    # Plotten
    plot_fano_factors(avg_fano_factors_per_direction, time_axis)

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
    simulate_model(
        experimental_trials=10,  # number of trials
        direction_range=[0, 1, 2],  # direction range
        stim_kernel=np.array([0.4, 0.3, 0.2, 0.2]),  # Stimulus-Kernel
        kernel_step=500,  # kernel step
        plot=True
    )
