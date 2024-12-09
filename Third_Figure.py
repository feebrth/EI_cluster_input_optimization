import optuna
import matplotlib.pyplot as plt
import numpy as np
from FF_Sim import simulate_model, calculate_baseline_and_delta

# Globale Schriftgrößen-Einstellungen
plt.rc('font', size=14)
plt.rc('axes', titlesize=16)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=18)

# Parameters for the Optuna study and simulation
trials = 5
direction_range = [0]
num_stimuli = 8
kernel_step = 2000 // num_stimuli  # 167 ms pro Stimulus

# Optuna Study laden
storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
study = optuna.load_study(study_name="8_FR_FF", storage=storage_url)

# Extrahiere die Best Trials
sorted_trials_ff = sorted(study.best_trials, key=lambda trial: trial.values[0])  # Nach "Loss FF" sortieren
sorted_trials_rates = sorted(study.best_trials, key=lambda trial: trial.values[1])  # Nach "Loss Rates" sortieren

# Auswahl der fünf repräsentativen Trials
selected_trials = [
    sorted_trials_ff[0],
    sorted_trials_ff[len(sorted_trials_ff) // 4],
    sorted_trials_ff[len(sorted_trials_ff) // 2],
    sorted_trials_ff[3 * len(sorted_trials_ff) // 4],
    sorted_trials_rates[0],
]

# Pareto-Front erstellen
loss_ff = [trial.values[0] for trial in study.best_trials]
loss_rates = [trial.values[1] for trial in study.best_trials]

# Figure mit Subplots erstellen
fig = plt.figure(figsize=(22, 20))
grid = fig.add_gridspec(4, 5, height_ratios=[2.5, 1, 2.5, 2.5], hspace=0.3)

# Pareto-Front-Plot
pareto_ax = fig.add_subplot(grid[0, :])
pareto_ax.scatter(loss_ff, loss_rates, color="blue", label="Trials", alpha=0.6)
pareto_ax.set_xlabel("Loss FF")
pareto_ax.set_ylabel("Loss Rates")
pareto_ax.set_title("Pareto-Front")
pareto_ax.grid(alpha=0.3)

# Nummerierung hinzufügen
pareto_ax.text(-0.1, 1.05, "(a)", fontsize=14, fontweight="bold", transform=pareto_ax.transAxes)

# Markiere die ausgewählten Trials und nummeriere sie
labels = ['a', 'b', 'c', 'd', 'e']
selected_ff = [trial.values[0] for trial in selected_trials]
selected_rates = [trial.values[1] for trial in selected_trials]

for i, (ff, rate) in enumerate(zip(selected_ff, selected_rates)):
    pareto_ax.scatter(ff, rate, color="red", s=60)
    pareto_ax.text(ff + 0.005, rate + 0.005, labels[i], fontsize=12, fontweight="bold", color="black")

pareto_ax.legend(fontsize=12)

# Simulation und Subplots
axs_fr = None
axs_ff = None
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

    # Stimulus-Amplituden plotten
    axs_stim = fig.add_subplot(grid[1, col])
    stim_time_points = np.arange(0, len(stimuli) * kernel_step, kernel_step)
    aligned_stim_curve = np.zeros_like(time_axis_rates)
    for i, stim in enumerate(stimuli):
        stim_start_idx = np.searchsorted(time_axis_rates, stim_time_points[i])
        stim_end_idx = np.searchsorted(time_axis_rates, stim_time_points[i] + kernel_step)
        aligned_stim_curve[stim_start_idx:stim_end_idx] = stim

    axs_stim.plot(time_axis_rates, aligned_stim_curve, label="Stimulus Amplitude", color="black")
    axs_stim.set_xlim([-500, 2100])
    axs_stim.set_ylim(0, 1.1)
    axs_stim.set_yticks([0.5, 1.0] if col == 0 else [])
    axs_stim.set_title(f"({label})", fontsize=14, loc="left")
    axs_stim.grid(alpha=0.3)

    if col == 0:
        axs_stim.set_ylabel("Stim. Amp. [pA]")

    # Delta-Firing-Rates plotten
    axs_fr = fig.add_subplot(grid[2, col], sharey=axs_fr if axs_fr else None)
    axs_fr.plot(time_axis_rates, sim_delta_rates, label=r'Simulated $\Delta$ FR', color="green")
    axs_fr.plot(exp_time_rates, exp_delta_rates, label=r'Experimental $\Delta$ FR', linestyle="--", color="orange")
    axs_fr.set_xlim([-500, 2100])
    axs_fr.grid(alpha=0.3)

    if col == 0:
        axs_fr.set_ylabel(r'$\Delta$ FR [spikes/s]')

    # Delta-Fano-Factors plotten
    axs_ff = fig.add_subplot(grid[3, col], sharey=axs_ff if axs_ff else None)
    axs_ff.plot(time_axis_ff, sim_delta_ff, label=r'Simulated $\Delta$ FF', color="blue")
    axs_ff.plot(exp_time_ff, exp_delta_ff, label=r'Experimental $\Delta$ FF', linestyle="--", color="#CC6600")
    axs_ff.set_xlim([-500, 2100])
    axs_ff.grid(alpha=0.3)

    if col == 0:
        axs_ff.set_ylabel(r'$\Delta$ FF')
    axs_ff.set_xlabel("Time (ms)")

# Gemeinsame Legende
handles, labels = axs_ff.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("Improved_Figure.png")
