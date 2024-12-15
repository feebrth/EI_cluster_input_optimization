import optuna
import matplotlib.pyplot as plt
import numpy as np
from FF_Sim import simulate_model, calculate_baseline_and_delta

# Globale Schriftgrößen-Einstellungen
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,  # Linienbreite
    'figure.titlesize': 18
})


# Parameters for the Optuna study and simulation
trials = 5
direction_range = [0,1,2]
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
fig = plt.figure(figsize=(20, 20))
grid = fig.add_gridspec(4, 5, height_ratios=[2, 0.8, 2.5, 2.5], hspace=0.4, wspace=0.4)

# Pareto-Front-Plot
pareto_ax = fig.add_subplot(grid[0, :])
pareto_ax.scatter(loss_ff, loss_rates, color="blue", label="Trials", alpha=0.6)
pareto_ax.set_xlabel("Loss FF", fontsize=16)
pareto_ax.set_ylabel("Loss Rates", fontsize=16)
pareto_ax.set_title("Pareto-Front", fontsize=16)
pareto_ax.grid(alpha=0.3)

# Markiere die ausgewählten Trials und nummeriere sie
labels = ['a', 'b', 'c', 'd', 'e']
selected_ff = [trial.values[0] for trial in selected_trials]
selected_rates = [trial.values[1] for trial in selected_trials]

dx, dy = 0.005, 0.005  # Anpassung für die Position der Labels
for i, (ff, rate) in enumerate(zip(selected_ff, selected_rates)):
    pareto_ax.scatter(ff, rate, color="red", s=60)
    pareto_ax.text(ff + dx, rate + dy, labels[i], fontsize=14, fontweight="bold", color="black")

pareto_ax.legend(fontsize=14)

# Initialisiere Achsen-Referenzen für gemeinsames Scaling
axs_fr = None
axs_ff = None

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
    stim_time_points = np.arange(0, len(stimuli) * kernel_step, kernel_step)
    aligned_stim_curve = np.zeros_like(time_axis_rates)
    for i, stim in enumerate(stimuli):
        stim_start_idx = np.searchsorted(time_axis_rates, stim_time_points[i])
        stim_end_idx = np.searchsorted(time_axis_rates, stim_time_points[i] + kernel_step)
        aligned_stim_curve[stim_start_idx:stim_end_idx] = stim

    axs_stim.plot(time_axis_rates, aligned_stim_curve, label="Stim. Amp.", color="black")
    axs_stim.set_xlim([-500, 2100])
    axs_stim.set_ylim(0, 1.1)
    axs_stim.set_title(f"{label}", fontsize=14, fontweight="bold", loc="left")
    axs_stim.grid(alpha=0.3)

    if col == 0:
        axs_stim.set_ylabel("Stim. Amp. [pA]", fontsize=16)
    else:
        axs_stim.yaxis.set_tick_params(labelleft=False)

    # Delta-Firing-Rates plotten (zweite Zeile)
    axs_fr = fig.add_subplot(grid[2, col], sharey=axs_fr if axs_fr else None)  # Gemeinsame y-Achse
    axs_fr.plot(time_axis_rates, sim_delta_rates, label=r'Sim. $\Delta$ FR [spike/s]', color="green")
    axs_fr.plot(exp_time_rates, exp_delta_rates, label=r'Exp. $\Delta$ FR [spike/s]', linestyle="--", color="red")
    axs_fr.set_xlim([-500, 2100])
    axs_fr.grid(alpha=0.3)

    if col == 0:
        axs_fr.set_ylabel(r'$\Delta$ FR [spikes/s]', fontsize=16)
    else:
        axs_fr.yaxis.set_tick_params(labelleft=False)

    # Delta-Fano-Factors plotten (dritte Zeile)
    axs_ff = fig.add_subplot(grid[3, col], sharey=axs_ff if axs_ff else None)  # Gemeinsame y-Achse
    axs_ff.plot(time_axis_ff, sim_delta_ff, label=r'Sim. $\Delta$ FF', color="blue")
    axs_ff.plot(exp_time_ff, exp_delta_ff, label=r'Exp. $\Delta$ FF', linestyle="--", color="orange")
    axs_ff.set_xlim([-500, 2100])
    axs_ff.grid(alpha=0.3)

    if col == 0:
        axs_ff.set_ylabel(r'$\Delta$ FF', fontsize=16)
    else:
        axs_ff.yaxis.set_tick_params(labelleft=False)
    axs_ff.set_xlabel("Time (ms)", fontsize=16)

# Legende mit Gruppierung
# Gruppenbasierte Legende für Simulated und Experimental
handles = [
    plt.Line2D([0], [0], color="blue", lw=2, label=r"Sim. $\Delta$ FF"),
    plt.Line2D([0], [0], color="orange", lw=2, label=r"Exp. $\Delta$ FF"),
    plt.Line2D([0], [0], color="green", lw=2, label=r"Sim. $\Delta$ FR"),
    plt.Line2D([0], [0], color="red", lw=2, label=r"Exp. $\Delta$ FR"),
    plt.Line2D([0], [0], color="black", lw=2, label="Stim. Amp."),
    plt.Line2D([0], [0], color="blue", lw=0, marker="o", label="Trials"),
]

# Legende mit Gruppierung hinzufügen
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12, frameon=False)


# Nummerierung der Subplot-Reihen hinzufügen
# Einheitliche x-Position für alle Reihen-Nummerierungen
x_position = 0.02  # Einheitliche x-Position für alle Reihen

# Nummerierung der Subplot-Reihen mit fig.text() für globale Ausrichtung
fig.text(x_position, 0.90, "(a)", fontsize=16, fontweight="bold", ha="left", va="center", rotation=0)
fig.text(x_position, 0.68, "(b)", fontsize=16, fontweight="bold", ha="left", va="center", rotation=0)
fig.text(x_position, 0.56, "(c)", fontsize=16, fontweight="bold", ha="left", va="center", rotation=0)
fig.text(x_position, 0.31, "(d)", fontsize=16, fontweight="bold", ha="left", va="center", rotation=0)



plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("Third_Figure_test.png")


