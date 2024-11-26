import optuna
import matplotlib.pyplot as plt
import numpy as np
from FF_Sim import simulate_model, calculate_baseline_and_delta

plt.rc('font', size=14)  # Standardtextgröße
plt.rc('axes', titlesize=16)  # Titelgröße der Achsen
plt.rc('axes', labelsize=14)  # Beschriftung der Achsen
plt.rc('xtick', labelsize=16)  # X-Tick-Beschriftungen
plt.rc('ytick', labelsize=16)  # Y-Tick-Beschriftungen
plt.rc('legend', fontsize=16)  # Legendentextgröße
plt.rc('figure', titlesize=18)  # Figurentitelgröße


# Parameters for the Optuna study and simulation
trials = 5
direction_range = [0]
num_stimuli = 24
kernel_step = 2000 // num_stimuli  # 167 ms pro Stimulus

# Optuna Study laden
storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
study = optuna.load_study(study_name="24_FR_FF", storage=storage_url)

# Extrahiere die Best Trials
sorted_trials_ff = sorted(study.best_trials, key=lambda trial: trial.values[0])  # Nach "Loss FF" sortieren
sorted_trials_rates = sorted(study.best_trials, key=lambda trial: trial.values[1])  # Nach "Loss Rates" sortieren

# Auswahl der fünf repräsentativen Trials
selected_trials = [
    sorted_trials_ff[0],  # Minimaler Loss FF
    sorted_trials_ff[len(sorted_trials_ff) // 4],  # Ein Viertel in Richtung Minimaler Loss Rates
    sorted_trials_ff[len(sorted_trials_ff) // 2],  # Median der FF-basierten Loss-Werte
    sorted_trials_ff[3 * len(sorted_trials_ff) // 4],  # Drei Viertel in Richtung Minimaler Loss Rates
    sorted_trials_rates[0],  # Minimaler Loss Rates
]

# Pareto-Front erstellen
loss_ff = [trial.values[0] for trial in study.best_trials]
loss_rates = [trial.values[1] for trial in study.best_trials]

# Figure mit Subplots erstellen
fig = plt.figure(figsize=(20, 18))

grid = fig.add_gridspec(4, 5, height_ratios=[1.5, 1, 3, 3], hspace= 0.6)

# Pareto-Front-Plot
pareto_ax = fig.add_subplot(grid[0, :])
pareto_ax.scatter(loss_ff, loss_rates, color="blue", label="Trials", alpha=0.6)
pareto_ax.set_xlabel("Loss FF", fontsize=14)
pareto_ax.set_ylabel("Loss Rates", fontsize=14)
pareto_ax.set_title("Pareto-Front Plot", fontsize=16)
pareto_ax.grid(alpha=0.3)

# Markiere die ausgewählten Trials und nummeriere sie
labels = ['a', 'b', 'c', 'd', 'e']
selected_ff = [trial.values[0] for trial in selected_trials]
selected_rates = [trial.values[1] for trial in selected_trials]

dx, dy = 0.005, 0.005  # Anpassung für die Position der Labels
for i, (ff, rate) in enumerate(zip(selected_ff, selected_rates)):
    pareto_ax.scatter(ff, rate, color="red", s=60)
    pareto_ax.text(ff + dx, rate + dy, labels[i], fontsize=10, fontweight="bold", color="black")


pareto_ax.legend(fontsize=12)

# Simulation und Subplots
for col, (trial, label) in enumerate(zip(selected_trials, labels)):
    stimuli = [trial.params[f"stimulus{i + 1}"] for i in range(num_stimuli)]

    # Simulation ausführen
    (sim_fano_factors, sim_firing_rates, time_axis_ff, time_axis_rates,
     exp_time_ff, exp_ff, exp_time_rates, exp_rates, _, _) = simulate_model(
        experimental_trials=trials,
        direction_range=direction_range,
        stim_kernel=stimuli,
        kernel_step=kernel_step,
        plot=False,
        use_delta=True
    )

    # Delta-Werte berechnen
    sim_delta_ff, exp_delta_ff, sim_delta_rates, exp_delta_rates = calculate_baseline_and_delta(
        simulated_ff=sim_fano_factors,
        simulated_rates=sim_firing_rates,
        time_axis_ff=time_axis_ff,
        time_axis_rates=time_axis_rates,
        exp_time_ff=exp_time_ff,
        exp_ff=exp_ff,
        exp_time_rates=exp_time_rates,
        exp_rates=exp_rates,
    )

    # Stimulus-Amplituden plotten (erste Zeile)
    axs_stim = fig.add_subplot(grid[1, col])

    # Stimulus-Zeitpunkte anpassen
    stim_time_points = np.arange(0, len(stimuli) * kernel_step, kernel_step)
    aligned_stim_curve = np.zeros_like(time_axis_rates)
    for i, stim in enumerate(stimuli):
        stim_start_idx = np.searchsorted(time_axis_rates, stim_time_points[i])
        stim_end_idx = np.searchsorted(time_axis_rates, stim_time_points[i] + kernel_step)
        aligned_stim_curve[stim_start_idx:stim_end_idx] = stim

    # Plot Stimulus auf derselben Zeitachse
    axs_stim.plot(time_axis_rates, aligned_stim_curve, label="Stimulus Kernel", color="black")
    axs_stim.set_xlim(time_axis_ff[0], time_axis_ff[-1])  # Gleiche Zeitachse wie die anderen Plots
    axs_stim.set_ylim(0, 1.1)
    axs_stim.set_yticks([0, 0.5, 1.0])  # Y-Ticks anzeigen
    axs_stim.set_title(f"{label}", fontsize=10, fontweight="bold", loc="left")

    # Grid hinzufügen
    axs_stim.grid(alpha=0.3)

    # X- und Y-Achsenbeschriftungen
    if col == 0:
        axs_stim.set_ylabel("Stimulus Amplitude", fontsize=12)  # Y-Achse mit Label
    #axs_stim.set_xlabel("Time (ms)", fontsize=12)  # X-Achsenbeschriftung für Zeit
    axs_stim.tick_params(axis="x", labelsize=10)  # X-Achsenticks anzeigen
    axs_stim.tick_params(axis="y", labelsize=10)  # Y-Achsenticks anzeigen

    # Delta-Firing-Rates plotten (zweite Zeile)
    axs_fr_left = None  # Referenz für die erste Spalte
    for col, (trial, label) in enumerate(zip(selected_trials, labels)):
        axs_fr = fig.add_subplot(grid[2, col], sharey=axs_fr_left)  # Gleiche y-Achse wie die erste Spalte
        if axs_fr_left is None:
            axs_fr_left = axs_fr  # Speichere die Referenz zur linken Spalte

        axs_fr.plot(time_axis_rates, sim_delta_rates, label="Simulated Delta Rates", color="green")
        axs_fr.plot(exp_time_rates, exp_delta_rates, label="Experimental Delta Rates", linestyle="--", color="yellow")
        axs_fr.grid(alpha=0.3)
        if col == 0:
            axs_fr.set_ylabel("Delta Firing Rate", fontsize=12)
        else:
            axs_fr.yaxis.set_visible(False)  # Y-Ticks nur links

    # Delta-Fano-Factors plotten (dritte Zeile)
    axs_ff_left = None  # Referenz für die erste Spalte
    for col, (trial, label) in enumerate(zip(selected_trials, labels)):
        axs_ff = fig.add_subplot(grid[3, col], sharey=axs_ff_left)  # Gleiche y-Achse wie die erste Spalte
        if axs_ff_left is None:
            axs_ff_left = axs_ff  # Speichere die Referenz zur linken Spalte

        axs_ff.plot(time_axis_ff, sim_delta_ff, label="Simulated Delta Fano", color="blue")
        axs_ff.plot(exp_time_ff, exp_delta_ff, label="Experimental Delta Fano", linestyle="--", color="red")
        axs_ff.grid(alpha=0.3)
        if col == 0:
            axs_ff.set_ylabel("Delta Fano Factors", fontsize=12)
        else:
            axs_ff.yaxis.set_visible(False)  # Y-Ticks nur links
        axs_ff.set_xlabel("Time (ms)", fontsize=12)

# Gemeinsame Legende unten hinzufügen
handles_stim, labels_stim = axs_stim.get_legend_handles_labels()  # Stimulus-Legende
handles_fr, labels_fr = axs_fr.get_legend_handles_labels()  # Firing-Rate-Legende
handles_ff, labels_ff = axs_ff.get_legend_handles_labels()  # Fano-Factor-Legende

# Kombinierte Legenden-Handles und -Labels
combined_handles = handles_stim + handles_fr + handles_ff
combined_labels = labels_stim + labels_fr + labels_ff

fig.legend(combined_handles, combined_labels, loc="lower center", ncol=4, fontsize=12, frameon=False)

# Layout anpassen und speichern
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("Final_Figure.png")









