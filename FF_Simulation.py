import numpy as np

import numpy as np


def FF_per_direction(spiketimes, timepoints, direction, window_size=400):
    """
    Calculates the Fano factor for each neuron and direction.

    Inputs:
        spiketimes: Array where column 0 contains spike times and column 1 contains neuron IDs
        timepoints: List of time points (ms) to define trials
        directions: List of direction values associated with each trial
        window_size: Size of the time window (in ms) over which the Fano factor is calculated


    Returns:
        avg_fano_factors_per_direction: Dictionary with directions as keys and the averaged Fano factor across neurons for each direction
    """
    fano_factors_per_direction = {direction: {} for direction in set(direction)}

    # Iterate over each direction
    for direction in set(direction):
        # Get timepoints for this direction
        direction_timepoints = [timepoints[i] for i in range(len(timepoints)) if direction[i] == direction]

        # Iterate over each time window for this direction
        for start_time in direction_timepoints:
            end_time = start_time + window_size

            # Filter spikes for this time window
            spikes_in_window = spiketimes[(spiketimes[:, 0] >= start_time) & (spiketimes[:, 0] < end_time)]

            if spikes_in_window.size == 0:
                print(f"No spikes in time window {start_time} to {end_time} for direction {direction}.")
                continue

            # Get unique neurons in this window
            unique_neurons = np.unique(spikes_in_window[:, 1].astype(int))

            # Calculate Fano factor for each neuron
            for neuron in unique_neurons:
                neuron_spikes = spikes_in_window[spikes_in_window[:, 1] == neuron]
                neuron_spike_count = len(neuron_spikes)

                if neuron not in fano_factors_per_direction[direction]:
                    fano_factors_per_direction[direction][neuron] = []

                fano_factors_per_direction[direction][neuron].append(neuron_spike_count)

    # Now calculate the Fano factor for each neuron and direction
    avg_fano_factors_per_direction = {}
    for direction, neuron_data in fano_factors_per_direction.items():
        fano_factors = []

        for neuron, spike_counts in neuron_data.items():
            mean_spike_count = np.mean(spike_counts)
            var_spike_count = np.var(spike_counts)

            if mean_spike_count > 0:
                fano_factor = var_spike_count / mean_spike_count
            else:
                fano_factor = 0

            fano_factors.append(fano_factor)

        # Average Fano factor over all neurons for this direction
        avg_fano_factors_per_direction[direction] = np.mean(fano_factors)

    return avg_fano_factors_per_direction


def compare_fano_factors(sim_fano_factors, exp_fano_factors):
    """
    Compares the Fano factors from the simulation with those from the experimental data.

    Inputs:
        sim_fano_factors: List of Fano factors from the simulation
        exp_fano_factors: List of Fano factors from the experimental data

    Returns:
        penalty: Sum of squared deviations between simulation and experiment
    """
    # Ensure both lists are the same length
    min_len = min(len(sim_fano_factors), len(exp_fano_factors))

    sim_fano_factors = sim_fano_factors[:min_len]
    exp_fano_factors = exp_fano_factors[:min_len]

    # Compute the deviation
    penalty = np.sum((np.array(sim_fano_factors) - np.array(exp_fano_factors)) ** 2)

    return penalty
