import optuna
import matplotlib.pyplot as plt
import numpy as np
from FF_Sim import simulate_model, plot_simulated_and_experimental_data

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
fig, axs = plt.subplots(3, 5, figsize=(20, 15), gridspec_kw={"height_ratios": [1, 1, 1]})

# Pareto-Front-Plot
pareto_ax = fig.add_subplot(3, 1, 1)  # Gesamte Breite für Pareto-Front
pareto_ax.scatter(loss_ff, loss_rates, color="blue", label="Trials", alpha=0.6)
pareto_ax.set_xlabel("Loss FF", fontsize=14)
pareto_ax.set_ylabel("Loss Rates", fontsize=14)
pareto_ax.set_title("Pareto-Front Plot", fontsize=16)
pareto_ax.grid(alpha=0.3)

# Markiere die ausgewählten Trials in der Pareto-Front
selected_ff = [trial.values[0] for trial in selected_trials]
selected_rates = [trial.values[1] for trial in selected_trials]
pareto_ax.scatter(selected_ff, selected_rates, color="red", label="Selected Trials", s=100)
pareto_ax.legend(fontsize=12)

# Simulation und Subplots für Delta-Firing-Rates und Delta-Fano-Factors
for col, trial in enumerate(selected_trials):
    # Stimuli aus den Trial-Parametern extrahieren
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

    # Plot Delta-Firing-Rates
    axs[1, col].plot(time_axis_rates, sim_firing_rates, label="Simulated Delta Rates", color="blue")
    axs[1, col].plot(exp_time_rates, exp_rates, label="Experimental Delta Rates", linestyle="--", color="orange")
    axs[1, col].set_title(f"Delta Firing Rates (Trial {col+1})", fontsize=12)
    axs[1, col].legend(fontsize=10)
    axs[1, col].grid(alpha=0.3)

    # Plot Delta-Fano-Factors
    axs[2, col].plot(time_axis_ff, sim_fano_factors, label="Simulated Delta Fano", color="blue")
    axs[2, col].plot(exp_time_ff, exp_ff, label="Experimental Delta Fano", linestyle="--", color="red")
    axs[2, col].set_title(f"Delta Fano Factors (Trial {col+1})", fontsize=12)
    axs[2, col].legend(fontsize=10)
    axs[2, col].grid(alpha=0.3)

# Layout anpassen
plt.tight_layout()
plt.savefig("Third_Figure.png")
