# -*- coding: utf-8 -*-
#
# run_simulation_EI.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""PyNEST EI-clustered network: Run Simulation
-----------------------------------------------

This is an example script for running the EI-clustered model
with two stimulations and generating a raster plot.
"""

# Importing necessary libraries
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import spiketools

# Importing custom modules
import network_EI as network
from helper_EI import raster_plot
from network_params_EI import net_dict
from sim_params_EI import sim_dict
from stimulus_params_EI import stim_dict
from Exp_data import get_exp_data



# Section 1: Data Generation Functions
def generate_trials(num_trials, direction_range, start=2000, step=3000, variation=200):
    """
    Generates a series of time points and associated directional values for a given number of trials.

    Inputs:
        num_trials: The total number of trials to generate
        direction_range: List of possible direction values
        start: The starting time for the first trial in ms
        step: The base time interval between trials in ms
        variation: The amount of variation added to the step to create random time intervals

    Returns:
        timepoints: List of time points when each trial occurs
        direction: List of randomly chosen direction values for each trial
    """
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
    """
    Generates stimulus timepoints and amplitudes for each direction based on provided timepoints, directions, and kernel.

    Inputs:
        timepoints: List of time points when trials occur
        direction: List of direction values for each trial
        kernel: List of amplitude values to be applied for each trial
        kernel_step: Time step in ms between each value in the kernel

    Returns:
        stim_dicts: A dictionary where each key is a unique direction, with corresponding stimulus times and amplitudes
    """
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


# Section 2: Firing Rate and Alignment Functions
def trial_firing_rates(spiketimes, timepoints, kernel, dt=1):
    """
    Generates trial-specific firing rates using spike times and a kernel for smoothing.

    Inputs:
        spiketimes: Array where column 0 contains spike times and column 1 contains neuron IDs
        timepoints: List of trial time points where each trial starts
        kernel: Array representing the kernel used for smoothing spike data
        dt: Time step in ms for generating the timeline

    Returns:
        firing_rates_array: Array of firing rates for each trial
        timeline: Array of time values corresponding to the firing rates
    """
    firing_rates = []
    for i, timepoint in enumerate(timepoints):
        trial_start = timepoint - 500 - len((kernel * dt) / 2)
        trial_end = timepoint + 2500 + len((kernel * dt) / 2)

        exc_trial_spikes = spiketimes[
            (spiketimes[:, 1] < 1200) & (spiketimes[:, 0] >= trial_start) & (spiketimes[:, 0] < trial_end)]

        if exc_trial_spikes.size == 0:
            print(f"No spikes for Trial {i} between {trial_start} and {trial_end}.")
            break
        else:
            rates, _ = spiketools.kernel_rate(exc_trial_spikes.T, kernel, tlim=(trial_start, trial_end))

        firing_rates.append(np.squeeze(rates))

    num_timepoints = len(firing_rates[0])
    timeline = np.arange(-500 - int(len(kernel) * dt / 2), -500 - int(len(kernel) * dt / 2) + num_timepoints * dt, dt)
    return np.array(firing_rates), timeline


def align_and_trim(time_line1, firing_rate1, time_line2, firing_rate2):
    """
    Aligns two timelines by their zero point and trims the firing rate arrays to have matching time windows.

    Inputs:
        time_line1: Time points for the first set of firing rates
        firing_rate1: Firing rates corresponding to time_line1
        time_line2: Time points for the second set of firing rates
        firing_rate2: Firing rates corresponding to time_line2

    Returns:
        trimmed_time_1: Aligned and trimmed time points
        trimmed_firing_rate1: Firing rates for the first set, trimmed to match the common time window
        trimmed_firing_rate2: Firing rates for the second set, trimmed to match the common time window
    """
    idx1_zero = (np.abs(time_line1 - 0)).argmin()
    idx2_zero = (np.abs(time_line2 - 0)).argmin()

    max_left = min(idx1_zero, idx2_zero)
    max_right = min(len(time_line1) - idx1_zero, len(time_line2) - idx2_zero)

    trimmed_time_1 = time_line1[idx1_zero - max_left: idx1_zero + max_right]
    trimmed_firing_rate1 = firing_rate1[idx1_zero - max_left: idx1_zero + max_right]
    trimmed_firing_rate2 = firing_rate2[idx2_zero - max_left: idx2_zero + max_right]

    return trimmed_time_1, trimmed_firing_rate1, trimmed_firing_rate2


# Section 3: Penalty Calculation
def calculate_penalty(average_firing_rate, experimental_avg_firing_rate1):
    """
    Calculates the penalty as the squared deviation between the simulated and experimental average firing rates.

    Inputs:
        average_firing_rate: Simulated average firing rate
        experimental_avg_firing_rate1: Experimental average firing rate

    Returns:
        penalty: Squared deviation between the two firing rates
    """
    penalty = np.sum((average_firing_rate - experimental_avg_firing_rate1) ** 2)
    return penalty


# Section 4: Main Simulation Function
def simulate_model(experimental_trials, direction_range, stim_kernel, kernel_step,
                   plot=False, num_stimuli = 12, best_penalty = None):
    """
    Runs the simulation model, calculates firing rates, compares to experimental data, and plots results if enabled.

    Inputs:
        experimental_trials: Number of trials to simulate
        direction_range: Range of direction values for trials
        stim_kernel: Array of stimulus amplitudes
        kernel_step: Time step between kernel points
        plot: Whether to plot the results

    Returns:
        penalty: The calculated penalty (difference between simulation and experiment)
    """
    # Generate timepoints and directions
    timepoints, direction = generate_trials(experimental_trials, direction_range)

    # Generate stimulus amplitudes
    stim_dicts = stim_amplitudes(timepoints, direction, stim_kernel, kernel_step)
    stim_dict['experiment_stim'] = stim_dicts
    sim_dict['simtime'] = max(timepoints) + 3000

    # Create and run the EI clustered network
    ei_network = network.ClusteredNetwork(sim_dict, net_dict, stim_dict)
    result = ei_network.get_simulation()

    # Calculate firing rates per trial
    spiketimes = result["spiketimes"].T
    sigma = 50  # Kernel width
    kernel = spiketools.gaussian_kernel(sigma)
    firing_rates_array, timeline = trial_firing_rates(spiketimes, timepoints, kernel)

    # Average firing rate over all trials
    average_firing_rate = np.mean(firing_rates_array, axis=0)

    # Get experimental data
    exp_time, experimental_avg_firing_rate1 = get_exp_data(1)

    # Align and trim firing rates
    aligned_time1, aligned_firing_rate1, aligned_firing_rate2 = align_and_trim(timeline, average_firing_rate, exp_time,
                                                                               experimental_avg_firing_rate1)

    # Calculate penalty
    penalty = calculate_penalty(aligned_firing_rate1, aligned_firing_rate2)
    print(f"Penalty for stimulus {stim_kernel} is: {penalty}")

    # Plot results if needed
    if plot:
        ax = raster_plot(
            result["spiketimes"],
            tlim=(0, sim_dict["simtime"]),
            colorgroups=[
                ("k", 0, net_dict["N_E"]),
                ("darkred", net_dict["N_E"], net_dict["N_E"] + net_dict["N_I"]),
            ],
        )
        plt.figure()
        plt.plot(aligned_time1, aligned_firing_rate1, color='blue', label='Simulated Firing Rate')
        plt.plot(aligned_time1, aligned_firing_rate2, color='red', label='Experimental Firing Rate')
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (spikes/s)')

        if best_penalty is None:
            best_penalty = penalty

        plt.title(f"Optimization: GP, Tested Stimuli: {num_stimuli}, Best value: {best_penalty:.4f}")

        plt.legend()
        plt.savefig(f"Optimized_GP_2200sim_{num_stimuli}_value_{best_penalty:.4f}.png")


    return penalty


if __name__ == "__main__":


    loss = simulate_model(
            experimental_trials=3,  # number of trials
            direction_range=[0,1,2],  # direction range
            stim_kernel= np.array([0.5, 0.4, 0.3, 0.2]),  # Stimulus-Kernel
            kernel_step= 500, # kernel step
            plot = True
        )

