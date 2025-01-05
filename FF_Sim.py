

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import network_EI as network
from helper_EI import raster_plot
from network_params_EI import net_dict
from sim_params_EI import sim_dict
from stimulus_params_EI import stim_dict
from Exp_data import get_exp_data_ff
from Exp_data import get_exp_data
import spiketools

def generate_trials(num_trials, direction_range, start= 2000, step=3000, variation=200):
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
                stim_times.extend([time + k * kernel_step for k in range(len(kernel))])
                stim_times.append(time + len(kernel) * kernel_step)
                stim_amps.extend(kernel)
                stim_amps.append(0)

        stim_dicts[dir_value] = {'stim_time': stim_times, 'stim_amps': stim_amps}
    return stim_dicts


import numpy as np


def preprocess_spiketimes(spiketimes):
    # Gives back dictionary that groups spiketimes for every neuron
    neurons_dict = {}
    for neuron_id in np.unique(spiketimes[:, 1]):
        neuron_spikes = spiketimes[spiketimes[:, 1] == neuron_id][:, 0]
        neurons_dict[neuron_id] = neuron_spikes
    return neurons_dict


def organize_spikes_by_trial(spiketimes, timepoints,trial_start_offset, trial_end_offset):
    spike_times = []
    trial_indices = []
    neuron_ids = []

    for trial_idx, trial_start in enumerate(timepoints):
        trial_end = timepoints[trial_idx + 1] if trial_idx < len(timepoints) - 1 else trial_start + trial_end_offset
        # Filter Spikes, die in den aktuellen Trial fallen
        spikes_in_trial = spiketimes[(spiketimes[:, 0] >= trial_start + trial_start_offset) & (spiketimes[:, 0] < trial_end)]

        if len(spikes_in_trial) > 0:
            spike_times.extend(spikes_in_trial[:, 0] - trial_start)  # relative Spikezeiten
            trial_indices.extend([trial_idx] * len(spikes_in_trial))  # Trial-Index
            neuron_ids.extend(spikes_in_trial[:, 1])  # Neuronen-IDs übernehmen

    spiketimes_array = np.array([spike_times, trial_indices, neuron_ids])
    return spiketimes_array



def preprocess_spiketimes(spiketimes):
    # Organisiere Spikes pro Neuron und berechne Feuerraten
    neurons_dict = {}
    firing_rates = {}

    # Berechne die Feuerrate für jedes Neuron
    for neuron_id in np.unique(spiketimes[:, 1]):
        neuron_spikes = spiketimes[spiketimes[:, 1] == neuron_id][:, 0]
        total_time = spiketimes[:, 0].max() - spiketimes[:, 0].min()  # Gesamtzeit in ms
        firing_rate = len(neuron_spikes) / (total_time / 1000)  # Feuerrate in Hz

        if firing_rate >= 5:  # Nur Neuronen mit Feuerrate >= 5 Hz speichern
            neurons_dict[neuron_id] = neuron_spikes
            firing_rates[neuron_id] = firing_rate

    return neurons_dict, firing_rates


import numpy as np
from spiketools import kernel_fano


def calculate_ff(spiketimes, timepoints, directions, direction_range, kernel, window_size=400, step_size=1):
    trial_start_offset = -1200
    trial_end_offset = 2500

    # Organisiere Spiketimes mit Trial- und Neuronen-IDs
    spiketimes_array = organize_spikes_by_trial(spiketimes, timepoints, trial_start_offset, trial_end_offset)
    print(f"Spiketimes array shape after organizing by trial: {spiketimes_array.shape}")

    # Berechnung der globalen Feuerrate
    num_trials = len(timepoints)
    num_neurons = len(np.unique(spiketimes_array[2, :]))  # Anzahl der Neuronen

    # Setze alle Trial-IDs auf 0 für globale Feuerrate
    global_spiketimes = spiketimes_array.copy()
    global_spiketimes[1, :] = 0  # Trial-IDs auf 0 setzen

    try:
        rates, time_axis_rates = spiketools.kernel_rate(
            global_spiketimes[:2, :],
            kernel,
            tlim=[trial_start_offset, trial_end_offset],
            pool=True  # Mittelwert über alle Neuronen und globales Trial
        )
        normalized_firing_rates = np.squeeze(rates) / (num_trials * num_neurons)  # Normalisierung der Feuerraten
    except Exception as e:
        print(f"Error calculating firing rates: {e}")
        normalized_firing_rates = None
        time_axis_rates = None

    # Berechnung der Fano-Faktoren
    fano_factors_by_neuron_and_direction = {}
    all_time_points_ff = None  # Zeitachse für Fano-Faktoren

    for dir_value in direction_range:
        relevant_trials = np.where(np.array(directions) == dir_value)[0]

        trial_mapping = {trial: idx for idx, trial in enumerate(relevant_trials)}

        relevant_spiketimes = spiketimes_array[:, np.isin(spiketimes_array[1, :], relevant_trials)]
        if relevant_spiketimes.shape[1] == 0:
            print(f"No spikes for direction {dir_value}")
            continue

        relevant_spiketimes[1, :] = np.vectorize(trial_mapping.get)(relevant_spiketimes[1, :])

        for neuron_id in np.unique(relevant_spiketimes[2, :]):
            if neuron_id >= 1200:
                continue

            neuron_spikes = relevant_spiketimes[:, relevant_spiketimes[2, :] == neuron_id]
            if neuron_spikes.shape[1] == 0:
                continue

            neuron_spikes_2d = neuron_spikes[:2, :]
            if neuron_spikes_2d.shape[0] != 2:
                continue

            try:
                fano, time_axis_ff = spiketools.kernel_fano(
                    neuron_spikes_2d,
                    window=window_size,
                    dt=step_size,
                    tlim=[trial_start_offset, trial_end_offset]
                )
            except Exception as e:
                print(f"Error in kernel_fano for neuron {neuron_id} in direction {dir_value}: {e}")
                continue

            if np.any(~np.isnan(fano)):
                if neuron_id not in fano_factors_by_neuron_and_direction:
                    fano_factors_by_neuron_and_direction[neuron_id] = {}
                fano_factors_by_neuron_and_direction[neuron_id][dir_value] = fano

                if all_time_points_ff is None:
                    all_time_points_ff = time_axis_ff

    # Berechnung der kombinierten Fano-Faktoren
    combined_fano_factors = []
    if all_time_points_ff is not None:
        for time_idx in range(len(all_time_points_ff)):
            factors = [
                fano_factors_by_neuron_and_direction[neuron][direction][time_idx]
                for neuron in fano_factors_by_neuron_and_direction
                for direction in fano_factors_by_neuron_and_direction[neuron]
                if not np.isnan(fano_factors_by_neuron_and_direction[neuron][direction][time_idx])
            ]
            combined_fano_factors.append(np.mean(factors) if factors else np.nan)

    return combined_fano_factors, normalized_firing_rates, all_time_points_ff, time_axis_rates




def align_and_trim(time_line1, firing_rate1, time_line2, firing_rate2):

    idx1_zero = (np.abs(time_line1 - 0)).argmin()
    idx2_zero = (np.abs(time_line2 - 0)).argmin()

    max_left = min(idx1_zero, idx2_zero)
    max_right = min(len(time_line1) - idx1_zero, len(time_line2) - idx2_zero)

    trimmed_time_1 = time_line1[idx1_zero - max_left: idx1_zero + max_right]
    trimmed_firing_rate1 = firing_rate1[idx1_zero - max_left: idx1_zero + max_right]
    trimmed_firing_rate2 = firing_rate2[idx2_zero - max_left: idx2_zero + max_right]

    return trimmed_time_1, trimmed_firing_rate1, trimmed_firing_rate2



def calculate_penalty(
    simulated_ff, exp_ff, simulated_rates, exp_rates,
    time_axis_ff, exp_time_ff, time_axis_rates, exp_time_rates,
    use_delta=False
):
    # Baselines und Deltas vor dem Alignment berechnen, falls notwendig
    if use_delta:
        # Berechnung der Baselines und Deltas
        sim_delta_ff, exp_delta_ff, sim_delta_rates, exp_delta_rates = calculate_baseline_and_delta(
            time_axis_ff, simulated_ff, exp_time_ff, exp_ff,
            time_axis_rates, simulated_rates, exp_time_rates, exp_rates
        )
    else:
        # Wenn keine Deltas verwendet werden, direkt die Originaldaten verwenden
        sim_delta_ff, exp_delta_ff = simulated_ff, exp_ff
        sim_delta_rates, exp_delta_rates = simulated_rates, exp_rates

    # Daten anpassen (align and trim) nach Baseline-Berechnung
    aligned_time_ff, aligned_sim_ff, aligned_exp_ff = align_and_trim(
        time_axis_ff, sim_delta_ff, exp_time_ff, exp_delta_ff
    )
    aligned_time_rates, aligned_sim_rates, aligned_exp_rates = align_and_trim(
        time_axis_rates, sim_delta_rates, exp_time_rates, exp_delta_rates
    )

    # Berechne die Penalty (z. B. mittlere quadrierte Differenz)
    ff_penalty = np.mean((aligned_sim_ff - aligned_exp_ff) ** 2)
    rates_penalty = np.mean((aligned_sim_rates - aligned_exp_rates) ** 2)


    return ff_penalty, rates_penalty





def plot_raster_for_neuron_by_direction(spiketimes, neuron_id, timepoints, directions, direction_range):
    # Filter spike times for the specific neuron
    neuron_spikes = spiketimes[spiketimes[:, 1] == neuron_id][:, 0]

    # Create a list to store spike times for each trial and direction
    spikes_per_trial = []

    # Loop through each direction to plot trials separately
    for dir_idx, dir_value in enumerate(direction_range):
        # Find trials with the current direction
        relevant_trials = [time for i, time in enumerate(timepoints) if directions[i] == dir_value]

        # Loop over each relevant trial
        for trial_idx, trial_start in enumerate(relevant_trials):
            # Find spikes relative to the trial start time and shift to align all trials
            trial_spikes = neuron_spikes[
                               (neuron_spikes >= trial_start) & (neuron_spikes < trial_start + 2500)] - trial_start
            trial_spikes += trial_idx * 2500  # Shift each trial on the time axis by 2500 ms
            spikes_per_trial.append((trial_spikes, dir_idx, trial_idx))

    # Plotting the raster plot
    plt.figure(figsize=(15, 8))
    for trial_data in spikes_per_trial:
        trial_spikes, dir_idx, trial_idx = trial_data
        y_position = dir_idx * len(relevant_trials) + trial_idx  # Stack trials per direction
        plt.scatter(trial_spikes, [y_position] * len(trial_spikes), color='black', s=10)

    # Set the labels and title
    plt.xlabel("Time (ms)")
    plt.ylabel("Trial and Direction Index")
    plt.title(f"Raster Plot for Neuron ID {neuron_id} Across Trials and Directions")

    # Set the y-ticks to label the directions
    num_trials_per_direction = len(relevant_trials)
    plt.yticks(
        [i * num_trials_per_direction + num_trials_per_direction // 2 for i in range(len(direction_range))],
        [f'Direction {dir}' for dir in direction_range]
    )

    # Adjust the x-axis limit to cover all trials
    total_trials = max([len([t for i, t in enumerate(timepoints) if directions[i] == dir]) for dir in direction_range])
    plt.xlim(0, total_trials * 2500)





def plot_stimulus_kernel(stim_kernel, kernel_step):
    time_points = np.arange(0, len(stim_kernel) * kernel_step, kernel_step)
    plt.figure(figsize=(10, 6))
    plt.bar(time_points, stim_kernel, width=kernel_step, color='skyblue', align='edge')
    plt.xlabel("Time (ms)")
    plt.ylabel("Stimulus Amplitude")
    plt.title("Stimulus Kernel Amplitudes Over Time")

def calculate_baseline_and_delta(
    time_axis_ff, simulated_ff, exp_time_ff, exp_ff,
    time_axis_rates, simulated_rates, exp_time_rates, exp_rates,
    sim_ff_baseline_range=(-700, -300), exp_ff_baseline_range=(-250, 0),
    sim_rate_baseline_range=(-750, -300), exp_rate_baseline_range=(-300, 0)
):
    # Arrays sicherstellen
    simulated_ff = np.array(simulated_ff)
    exp_ff = np.array(exp_ff)
    simulated_rates = np.array(simulated_rates)
    exp_rates = np.array(exp_rates)

    # Baseline für Fano Faktoren
    sim_ff_baseline_idx = np.where((time_axis_ff >= sim_ff_baseline_range[0]) & (time_axis_ff <= sim_ff_baseline_range[1]))[0]
    exp_ff_baseline_idx = np.where((exp_time_ff >= exp_ff_baseline_range[0]) & (exp_time_ff <= exp_ff_baseline_range[1]))[0]


    sim_ff_baseline = np.mean(simulated_ff[sim_ff_baseline_idx])
    exp_ff_baseline = np.mean(exp_ff[exp_ff_baseline_idx])

    # Delta Fano Faktoren
    sim_delta_ff = simulated_ff - sim_ff_baseline
    exp_delta_ff = exp_ff - exp_ff_baseline

    # Baseline für Feuerraten
    sim_rate_baseline_idx = np.where((time_axis_rates >= sim_rate_baseline_range[0]) & (time_axis_rates <= sim_rate_baseline_range[1]))[0]
    exp_rate_baseline_idx = np.where((exp_time_rates >= exp_rate_baseline_range[0]) & (exp_time_rates <= exp_rate_baseline_range[1]))[0]

    sim_rate_baseline = np.mean(simulated_rates[sim_rate_baseline_idx])
    exp_rate_baseline = np.mean(exp_rates[exp_rate_baseline_idx])

    # Delta Feuerraten
    sim_delta_rates = simulated_rates - sim_rate_baseline
    exp_delta_rates = exp_rates - exp_rate_baseline

    return sim_delta_ff, exp_delta_ff, sim_delta_rates, exp_delta_rates





def simulate_model(
    experimental_trials, direction_range, stim_kernel, kernel_step, plot=False, use_delta=False
):
    # Generiere Trials und Richtungen
    timepoints, directions = generate_trials(experimental_trials, direction_range)

    # Stimuli erzeugen
    stim_dicts = stim_amplitudes(timepoints, directions, stim_kernel, kernel_step)
    stim_dict['experiment_stim'] = stim_dicts
    sim_dict['simtime'] = max(timepoints) + 3000

    # Netzwerk erstellen und Simulation ausführen
    ei_network = network.ClusteredNetwork(sim_dict, net_dict, stim_dict)
    result = ei_network.get_simulation()

    # Spiketimes extrahieren
    spiketimes = result["spiketimes"].T
    kernel = spiketools.gaussian_kernel(50)  # Beispiel-Kernel

    # Fano-Faktoren und Feuerraten berechnen
    fano_factors, firing_rates, time_axis_ff, time_axis_rates = calculate_ff(
        spiketimes=spiketimes,
        timepoints=timepoints,
        directions=directions,
        direction_range=direction_range,
        kernel=kernel,
        window_size=400,
        step_size=1
    )

    # Experimentelle Daten laden
    exp_time_ff, exp_ff = get_exp_data_ff(1)
    exp_time_rates, exp_rates = get_exp_data(1)

    # Berechnung der Penalty
    penalty_ff, penalty_rates = calculate_penalty(
        simulated_ff=fano_factors,
        exp_ff=exp_ff,
        simulated_rates=firing_rates,
        exp_rates=exp_rates,
        time_axis_ff=time_axis_ff,
        exp_time_ff=exp_time_ff,
        time_axis_rates=time_axis_rates,
        exp_time_rates=exp_time_rates,
        use_delta=use_delta
    )
    print(f"Loss FF: {penalty_ff}, Loss Rates: {penalty_rates}")



    # Plotten der Ergebnisse
    if plot:
        plot_simulated_and_experimental_data(
            simulated_ff=fano_factors,
            simulated_rates=firing_rates,
            time_axis_ff=time_axis_ff,
            time_axis_rates=time_axis_rates,
            exp_time_ff=exp_time_ff,
            exp_ff=exp_ff,
            exp_time_rates=exp_time_rates,
            exp_rates=exp_rates,
            plot_delta=use_delta
        )

    return fano_factors, firing_rates, time_axis_ff, time_axis_rates, exp_time_ff, exp_ff, exp_time_rates, exp_rates, penalty_ff, penalty_rates


def plot_fano_factors(simulated_ff, time_axis, exp_time, exp_ff):
    plt.figure(figsize=(10, 6))
    print(f"Simulated time axis for plot: {time_axis}")
    print(f"Experimental time axis for plot: {exp_time}")
    plt.plot(time_axis, simulated_ff, label='Simulated Fano Factor', color='blue')
    plt.plot(exp_time, exp_ff, label='Experimental Fano Factor', linestyle='--', color='red')
    plt.xlabel("Time (ms)")
    plt.ylabel("Fano Factor")
    plt.xlim(min(time_axis[0], exp_time[0]), max(time_axis[-1], exp_time[-1]))

    plt.title("Comparison of Simulated and Experimental Fano Factors")
    plt.legend()
    plt.grid(True)



def plot_simulated_and_experimental_data(
    simulated_ff, simulated_rates, time_axis_ff, time_axis_rates,
    exp_time_ff, exp_ff, exp_time_rates, exp_rates, stim_kernel=None, kernel_step=None,
    plot_delta=True
):
    plt.rcParams['lines.linewidth'] = 1.5

    sim_delta_ff, exp_delta_ff, sim_delta_rates, exp_delta_rates = calculate_baseline_and_delta(
        time_axis_ff, simulated_ff, exp_time_ff, exp_ff,
        time_axis_rates, simulated_rates, exp_time_rates, exp_rates
    )

    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2,  # Einheitliche Linienbreite
        'figure.titlesize': 18
    })

    # Figure erstellen
    fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex='col', gridspec_kw={'height_ratios': [0.5, 1, 1]})
    tick_fontsize = 12
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12, hspace=0.4, wspace=0.1)  # Weniger Abstand horizontal

    # Stimulus Kernel (erste Zeile, links und rechts)
    if stim_kernel is not None and kernel_step is not None:
        stim_time_points = np.arange(0, len(stim_kernel) * kernel_step, kernel_step)
        aligned_stim_curve = np.zeros_like(time_axis_rates)
        for i, stim in enumerate(stim_kernel):
            stim_start_idx = np.searchsorted(time_axis_rates, stim_time_points[i])
            stim_end_idx = np.searchsorted(time_axis_rates, stim_time_points[i] + kernel_step)
            aligned_stim_curve[stim_start_idx:stim_end_idx] = stim

        axs[0, 0].plot(time_axis_rates, aligned_stim_curve, label="Stimulus Kernel", color="black")
        axs[0, 0].set_ylabel('Amplitude', fontsize=16)

        axs[0, 0].grid(True)
        axs[0, 0].tick_params(axis='both', labelsize=tick_fontsize)

        axs[0, 1].plot(time_axis_rates, aligned_stim_curve, label="Stimulus Kernel", color="black")

        axs[0, 1].grid(True)
        axs[0, 1].tick_params(axis='both', labelsize=tick_fontsize)

    # Firing Rates (zweite Zeile, links)
    axs[1, 0].plot(time_axis_rates, simulated_rates, label='Simulated Firing Rates', color='green')
    axs[1, 0].plot(exp_time_rates, exp_rates, label='Experimental Firing Rates', linestyle='--', color='orange')
    axs[1, 0].set_ylabel('Firing Rate (spikes/s)', fontsize=16)

    axs[1, 0].grid(True)
    axs[1, 0].tick_params(axis='both', labelsize=tick_fontsize)

    # Fano Factors (zweite Zeile, rechts)
    axs[1, 1].plot(time_axis_ff, simulated_ff, label='Simulated Fano Factors', color='blue')
    axs[1, 1].plot(exp_time_ff, exp_ff, label='Experimental Fano Factors', linestyle='--', color='red')

    axs[1, 1].grid(True)
    axs[1, 1].tick_params(axis='both', labelsize=tick_fontsize)

    if plot_delta:
        # Delta Firing Rates (dritte Zeile, links)
        axs[2, 0].plot(time_axis_rates, sim_delta_rates, label='Delta Simulated Firing Rates', color='green')
        axs[2, 0].plot(exp_time_rates, exp_delta_rates, label='Delta Experimental Firing Rates', linestyle='--', color='orange')
        axs[2, 0].set_xlabel('Time (ms)', fontsize=16)
        axs[2, 0].set_ylabel('Delta Firing Rate (spikes/s)', fontsize=16)

        axs[2, 0].grid(True)
        axs[2, 0].tick_params(axis='both', labelsize=tick_fontsize)

        # Delta Fano Factors (dritte Zeile, rechts)
        axs[2, 1].plot(time_axis_ff, sim_delta_ff, label='Delta Simulated Fano Factors', color='blue')
        axs[2, 1].plot(exp_time_ff, exp_delta_ff, label='Delta Experimental Fano Factors', linestyle='--', color='red')
        axs[2, 1].set_xlabel('Time (ms)', fontsize=16)

        axs[2, 1].grid(True)
        axs[2, 1].tick_params(axis='both', labelsize=tick_fontsize)

    # Nummerierung der Subplots
    # Nummerierung der Subplots
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    row_offsets = [1.05, 1.05, 1.05]  # Vertikale Höhe für jede Zeile relativ zum Plot
    horizontal_offset = -0.15  # Horizontaler Abstand für die Nummerierung (mehr nach links)

    for i, ax in enumerate(axs.flat):
        row_index = i // 2  # Zeilenindex bestimmen
        ax.text(
            horizontal_offset, row_offsets[row_index], labels[i],
            transform=ax.transAxes, fontsize=16, fontweight='bold',
            va='center', ha='left'
        )

    # Legende zentriert unterhalb der Figure
    handles = [
        plt.Line2D([0], [0], color="blue", lw=2, label="Simulated Fano Factors"),
        plt.Line2D([0], [0], color="red", linestyle="--", lw=2, label="Experimental Fano Factors"),
        plt.Line2D([0], [0], color="green", lw=2, label="Simulated Firing Rates"),
        plt.Line2D([0], [0], color="orange", linestyle="--", lw=2, label="Experimental Firing Rates"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),  # Vertikale Position (-0.01 ist näher an der Figure)
        ncol=2,
        fontsize=14
    )

    # Layout anpassen
    plt.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.12, hspace=0.1, wspace=0.2)

    # Figure speichern
    plt.savefig("Optimal_FR_corr.FF4.png", dpi=300)
    plt.show()


# Hauptprogramm
if __name__ == "__main__":

    stim_kernel = np.array([0.17312816364086267, 0.8259290941384977, 0.44182977675232843,  0.3558826809260373,
                                 0.2856217738814025,  0.3040297739070603,  0.2366001890107655,  0.2472914682618193,
                                0.3150498635199546, 0.3790337694745662, 0.36722502142689295, 0.39463791746376475,
                                 0.2633404217765265,  0.32753879091207094,  0.12843437450070974,  0.0,  0.0,  0.0,
                                 0.3216556491748419,  0.49706313013206366,  1.0,  1.0,  1.0,  1.0])


    kernel_step = 2000 // len(stim_kernel)

        # Simuliere Modell und extrahiere Ergebnisse

    (sim_fano_factors, sim_firing_rates, time_axis_ff,
     time_axis_rates, exp_time_ff, exp_ff, exp_time_rates, exp_rates, penalty_ff, penalty_rates) = simulate_model(
        experimental_trials=5,
        direction_range=[0, 1, 2],
        stim_kernel=stim_kernel,
        kernel_step=kernel_step,
        use_delta=True,
        plot=False,
    )

    # Plotte die Ergebnisse mit zusätzlichem Delta der Feuerraten
    plot_simulated_and_experimental_data(
        simulated_ff=sim_fano_factors,
        simulated_rates=sim_firing_rates,
        time_axis_ff=time_axis_ff,
        time_axis_rates=time_axis_rates,
        exp_time_ff=exp_time_ff,
        exp_ff=exp_ff,
        exp_time_rates=exp_time_rates,
        exp_rates=exp_rates,
        stim_kernel=stim_kernel,
        kernel_step=kernel_step,
        plot_delta=True  # Delta der Feuerraten plotten
    )


#stim_kernel = 0.25 * np.array([0.17312816364086267, 0.8259290941384977, 0.44182977675232843,  0.3558826809260373,
#                             0.2856217738814025,  0.3040297739070603,  0.2366001890107655,  0.2472914682618193,
#                             0.3150498635199546, 0.3790337694745662, 0.36722502142689295, 0.39463791746376475,
#                             0.2633404217765265,  0.32753879091207094,  0.12843437450070974,  0.0,  0.0,  0.0,
#                             0.3216556491748419,  0.49706313013206366,  1.0,  1.0,  1.0,  1.0])

 # stim_kernel = 0.25 * np.array([0.48491825131034916, 0.32473703078742916, 0.3213764508701387, 0.3559706343482832,
    #                                0.1600546571600051, 0.10332896768233334,  0.033182727643445285, 0.018301728702906486,
    #                                0.473729003865031, 0.7842154568014152, 0.3061051722976537, 0.5379486363859469,
    #                                0.7357127303237456, 0.045103931882521264, 0.4286873051771577, 0.4209310585423314,
    #                                0.04671725381969315, 0.030049245579308215, 0.015167354140789166, 0.4882844499110816,
    #                                0.9415020162495253, 0.4519384264379514, 0.7956600093375796, 0.1970416608701149])
