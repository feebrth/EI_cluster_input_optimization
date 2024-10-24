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

import matplotlib.pyplot as plt
import network_EI as network


from helper_EI import raster_plot
from network_params_EI import net_dict
from sim_params_EI import sim_dict
from stimulus_params_EI import stim_dict
import sys
import random
import numpy as np
import spiketools
from Exp_data import get_exp_data




def generate_trials(num_trials, direction_range, start=2000, step= 3000, variation= 200):
    """ generates a series of time points and associated directional values over a given number of trials.
        The timepoints are spaced by a random interval that varies around a given step size. The directions are chosen
        randomly from a provided range of values.

        Input:
        num_trials: The total number of trials to generate
        direction_range: list of possible direction values
        start: the starting time for the first trial in ms
        step: the base time interval between trials in ms
        variation: the amount of variation in ms added to the step to create random time intervals

        Output:
        timepoints: list of time points when each trial occurs
        direction: list of randomly chosen direction values for each trial
    """
    timepoints = []
    direction = []

    time = start

    for _ in range(num_trials):
    #while time < end:
        timepoints.append(time)
        direction.append(random.choice(direction_range))

        random_step = step + random.randint(0, variation)
        time += random_step

    return timepoints, direction

#example
#num_trials = 5
#direction_range = [0,1,2] #nur direction 0,1 und 2
#timepoints, direction = generate_trials(num_trials, direction_range




def stim_amplitudes(timepoints: object, direction: object, kernel: object, kernel_step: object) -> object:
    """ Generates timepoints for the different stimulus amplitudes for each direction based on the provided (trial start) time points and
        directions from the generates_trials function and kernel.
        For each unique direction, it creates a series of stimulus times and corresponding amplitudes using a predefined
        kernel (a list of amplitude values) and a kernel step (the time difference between successive points in the kernel).

        Input:
        timepoints: list of time points when trials occur (from generate_trials function)
        direction: list of direction values for each trial (from generate_trials function)
        kernel: list of amplitude values to be applied for each trial
        kernel_step: the time step in ms between each value in the kernel

        Output:
        stim_dicts: a dictionary where each key is a unique direction value, and the value is another dictionary with:
            - 'stim_time': list of stimulus times for that direction
            - 'stim_amps': list of corresponding stimulus amplitudes based on the kernel values and 0 after the kernel ends
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

        stim_dicts[dir_value] = {
            'stim_time': stim_times,
            'stim_amps': stim_amps
        }
    return stim_dicts


# if __name__ == "__main__":
#     timepoints, direction = generate_trials(5,[1,4,5])
#     print(timepoints)
#     print(direction)
#     stim_dicts = stim_amplitudes(timepoints, direction, np.array([0.2,0.15,0.1, 0.05]), 500)


def trial_firing_rates(spiketimes, timepoints, kernel, dt=1):
    """ Generates trial-specific firing rates using spike times and a kernel for smoothing.
        For each trial, the function extracts spike times within a window around the trial timepoints
        and applies a kernel to calculate the firing rate.

        Input:
        spiketimes: numpy array of shape (n_spikes, 2), where:
                    - column 0 contains the spike times
                    - column 1 contains neuron IDs (for selecting specific neurons)
        timepoints: list of trial time points where each trial starts
        kernel: array of values representing the kernel used for smoothing spike data
        dt: time step in ms for generating the timeline (default is 1 ms)

        Output:
        firing_rates_array: numpy array of firing rates for each trial
        timeline: numpy array of time values corresponding to the firing rates, centered on the trial start time
        """

    firing_rates = []

    for i, timepoint in enumerate(timepoints):      #Trial lengths
        trial_start = timepoint - 500 - len(((kernel)*dt)/2)
        trial_end = timepoint + 2500 + len(((kernel)*dt)/2)

        exc_trial_spikes = spiketimes[(spiketimes[:, 1] < 1200) & (spiketimes[:, 0] >= trial_start) & (spiketimes[:, 0] < trial_end)]

        if exc_trial_spikes.size == 0:
            print(f"No spikes for Trial {i} between {trial_start} and {trial_end}.")
            break
        else:
            rates, _ = spiketools.kernel_rate(exc_trial_spikes.T, kernel, tlim=(trial_start, trial_end))

        rates = np.squeeze(rates)
        firing_rates.append(rates)

    num_timepoints = len(firing_rates[0]) #number of timepoints from length of firing rate
    timeline = np.arange(-500 - int(len((kernel) * dt) / 2), - 500 - int(len((kernel) * dt) / 2) + num_timepoints *dt, dt)

    firing_rates_array = np.array(firing_rates)

    return firing_rates_array, timeline #firing rate and timeline


def align_and_trim(time_line1, firing_rate1, time_line2, firing_rate2):
    """ Aligns two timelines by their zero point and trims the firing rate arrays to have matching time windows.

            Input:
            time_line1: numpy array of time points for the first set of firing rates
            firing_rate1: numpy array of firing rates corresponding to time_line1
            time_line2: numpy array of time points for the second set of firing rates
            firing_rate2: numpy array of firing rates corresponding to time_line2

            Output:
            trimmed_time_1: aligned and trimmed time points (common time window around 0)
            trimmed_firing_rate1: firing rates for the first set, trimmed to match the common time window
            trimmed_firing_rate2: firing rates for the second set, trimmed to match the common time window
        """
    idx1_zero = (np.abs(time_line1 - 0)).argmin()
    idx2_zero = (np.abs(time_line2 - 0)).argmin()

    max_left = min(idx1_zero, idx2_zero)  # number of elements left side of 0
    max_right = min(len(time_line1) - idx1_zero, len(time_line2) - idx2_zero)  # number of elements right side of 0

    trimmed_time_1 = time_line1[idx1_zero - max_left: idx1_zero + max_right]
    #trimmed_arr2 = time_line2[idx2_zero - max_left: idx2_zero + max_right]

    trimmed_firing_rate1 = firing_rate1[idx1_zero - max_left: idx1_zero + max_right]
    trimmed_firing_rate2 = firing_rate2[idx2_zero - max_left: idx2_zero + max_right]

    return trimmed_time_1, trimmed_firing_rate1, trimmed_firing_rate2



def simulate_model(experimental_trials, direction_range, stim_kernel, kernel_step, plot = False):
    #generate timepoints and direction
    timepoints, direction = generate_trials(experimental_trials, direction_range)

    # Return timepoints and direction
    print("Timepoints:", timepoints)
    print("Directions:", direction)

    # generate stimulus amplitudes
    stim_dicts = stim_amplitudes(timepoints, direction, stim_kernel, kernel_step)



    # for dir_value, stim_dict in stim_dicts.items():
    #     print(f"Dictionary for direction {dir_value}:")
    #     print("stim_time:", stim_dict['stim_time'])
    #     print("stim_amps:", stim_dict['stim_amps'])
    #     print()

    #sys.exit(0)

    #stim_dict = {'experiment_stim': stim_dicts, **stim_dict}
    stim_dict['experiment_stim'] = stim_dicts
    sim_dict['simtime'] = max(timepoints) + 3000



    # Creates object which creates the EI clustered network in NEST
    ei_network = network.ClusteredNetwork(sim_dict, net_dict, stim_dict)

    # Runs the simulation and returns the spiketimes
    # get simulation initializes the network in NEST
    # and runs the simulation
    # it returns a dict with the average rates,
    # the spiketimes and the used parameters
    result = ei_network.get_simulation() #dictionary

#Calculation fire rates

    spiketimes = result["spiketimes"]
    spiketimes = spiketimes.T
    spike_times_array = np.vstack((spiketimes[:,0], spiketimes[:,1]))


    sigma = 50 #kernel width
    kernel = spiketools.gaussian_kernel(sigma)

    rates, time = spiketools.kernel_rate(spike_times_array, kernel, tlim=(0, sim_dict["simtime"]))

    # plt.figure()
    # plt.plot(time, rates.T)  # Plot of firerates over time
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Firing Rate (Hz)')
    # plt.title('Estimated Firing Rate using Triangular Kernel Smoothing')
    # plt.savefig("firing_rate_triangular.png")
    # plt.show()

    #Calculation of firing rates per trial
    #returns array with fire rate for each trial and corresponding timeline

    firing_rates_array, timeline = trial_firing_rates(spiketimes, timepoints, kernel)

    for i, rates in enumerate(firing_rates_array):
        print(f"Trial {i} has {len(rates)} firing rate points")

    print("Firing rates array:")
    print(firing_rates_array)

    print("timeline:")
    print(timeline)

    #Average over all trials

    average_firing_rate = np.mean(firing_rates_array, axis=0)

    print("Average firing rate over all trials:")
    print(average_firing_rate)

    print("Timeline:")
    print(timeline)

    #plt.plot(time, average_firing_rate, label='Average firing rate')

    # Plot labels
    # plt.xlabel('Time (ms)')
    # plt.ylabel('Average firing rate (spikes/s)')
    # plt.title('Average firing rate for all trials')
    # plt.legend()
    #
    # plt.show()


    time, experimental_avg_firing_rate1 = get_exp_data(1)


    #Retrieval of experimental and simulated firing rate
    aligned_time1, aligned_firing_rate1, aligned_firing_rate2 = align_and_trim(timeline, average_firing_rate, time, experimental_avg_firing_rate1)

    def calculate_penalty(aligned_firing_rate1, aligned_firing_rate2):

        penalty = np.sum((aligned_firing_rate1 - aligned_firing_rate2)**2) # quadratische Abweichung (square deviation) zwischen beiden Avg. fire rates
        return penalty  # ein Wert pro Simulation


    penalty = calculate_penalty(aligned_firing_rate1, aligned_firing_rate2)

    if plot:

    # Plot of average experimental and simulated firing rates (same timeline)
    # print("Aligned and trimmed time 1:", aligned_time1)
    # print("Trimmed firing rate simulated:", aligned_firing_rate1)
    # print("Trimmed firing rate experimental 2:", aligned_firing_rate2)
        ax = raster_plot(
            result["spiketimes"],
            tlim=(0, sim_dict["simtime"]),
            colorgroups=[
                ("k", 0, net_dict["N_E"]),
                ("darkred", net_dict["N_E"], net_dict["N_E"] + net_dict["N_I"]),
            ],
        )
        # plt.savefig("clustered_ei_raster.png")
        print(f"Firing rate of excitatory neurons: {result['e_rate']:6.2f} spikes/s")
        print(f"Firing rate of inhibitory neurons: {result['i_rate']:6.2f} spikes/s")


        plt.figure()

        plt.plot(aligned_time1, aligned_firing_rate1, color='blue', label='Firing Rate simulated')
        plt.plot(aligned_time1, aligned_firing_rate2, color = "red", label= "Firing Rate experimental data")


        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (spikes/s)')

        plt.title(f'Stimulus: {stim_kernel}')
        plt.legend()
        plt.show()


   # return penalty


if __name__ == "__main__":


    loss = simulate_model(
            experimental_trials=3,  # number of trials
            direction_range=[0,1,2],  # direction range
            stim_kernel= np.array([0.5, 0.4, 0.3, 0.2]),  # Stimulus-Kernel
            kernel_step= 500, # kernel step
            plot = True
        )

    print(loss)


    # for trial_id, data in trial_firing_rates_result.items():
    #     plt.figure()  # Erstellt einen neuen Plot für jeden Trial
    #
    #     # Plot der Feuerrate für den aktuellen Trial
    #     rates = np.squeeze(data['rates'])  # Stellt sicher, dass 'rates' 1D ist
    #     plt.plot(data['time'], rates, label=f'Trial {trial_id}')
    #
    #     # Achsenbeschriftungen und Titel
    #     plt.title(f'Trial {trial_id}: Feuerrate')
    #     plt.xlabel('Zeit (ms)')
    #     plt.ylabel('Feuerrate (spikes/s)')
    #     plt.legend()
    #
    # plt.show()
