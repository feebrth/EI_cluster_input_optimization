import optuna
import matplotlib.pyplot as plt
import numpy as np
from FF_Sim import simulate_model, calculate_baseline_and_delta

# Parameters
trials = 5
direction_range = [0]
num_stimuli = 24
kernel_step = 2000 // num_stimuli

# Load Optuna Study
storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
study = optuna.load_study(study_name="24_FR_FF", storage=storage_url)

# Select Best Trials
sorted_trials_ff = sorted(study.best_trials, key=lambda trial: trial.values[0])
sorted_trials_rates = sorted(study.best_trials, key=lambda trial: trial.values[1])
selected_trials = [
    sorted_trials_ff[0],
    sorted_trials_ff[len(sorted_trials_ff) // 4],
    sorted_trials_ff[len(sorted_trials_ff) // 2],
    sorted_trials_ff[3 * len(sorted_trials_ff) // 4],
    sorted_trials_rates[0],
]

# Pareto Front Data
loss_ff = [trial.values[0] for trial in study.best_trials]
loss_rates = [trial.values[1] for trial in study.best_trials]

# Create Figure
fig, axs = plt.subplots(4, 5, figsize=(20, 18), gridspec_kw={'height_ratios': [1, 0.5, 1, 1]})

# Pareto-Front Plot
pareto_ax = axs[0, :]
pareto_ax = pareto_ax[0]  # Single axis spanning all columns
pareto_ax.scatter(loss_ff, loss_rates, color="blue", label="Trials", alpha=0.6)
pareto_ax.set_xlabel("Loss FF", fontsize=14)
pareto_ax.set_ylabel("Loss Rates", fontsize=14)
pareto_ax.set_title("Pareto-Front Plot", fontsize=16)
pareto_ax.grid(alpha=0.3)

# Highlight selected trials
labels = ['a', 'b', 'c', 'd', 'e']
selected_ff = [trial.values[0] for trial in selected_trials]
selected_rates = [trial.values[1] for trial in selected_trials]
for i, (ff, rate) in enumerate(zip(selected_ff, selected_rates)):
    pareto_ax.scatter(ff, rate, color="red", s=100)
    pareto_ax.text(ff - 0.01, rate + 0.1, labels[i], fontsize=12, fontweight="bold", color="black")

pareto_ax.legend(fontsize=12)

# Process and plot each trial
for col, (trial, label) in enumerate(zip(selected_trials, labels)):
    stimuli = [trial.params[f"stimulus{i + 1}"] for i in range(num_stimuli)]

    # Simulate
    sim_fano_factors, sim_firing_rates, time_axis_ff, time_axis_rates, exp_time_ff, exp_ff, exp_time_rates, exp_rates, _, _ = simulate_model(
        experimental_trials=trials, direction_range=direction_range, stim_kernel=stimuli, kernel_step=kernel_step,
        plot=False, use_delta=True
    )

    # Calculate Deltas
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

    # Stimulus Plot
    stim_ax = axs[1, col]
    stim_time_points = np.arange(0, len(stimuli) * kernel_step, kernel_step)
    stim_ax.bar(stim_time_points, stimuli, width=kernel_step, color="black", align="edge")
    stim_ax.set_ylim(0, 1.1)
    stim_ax.set_xticks([])
    stim_ax.set_yticks([])
    stim_ax.set_title(f"({label})", fontsize=12, loc="left")
    stim_ax.grid(alpha=0.3)

    # Delta Firing Rates
    fr_ax = axs[2, col]
    fr_ax.plot(time_axis_rates, sim_delta_rates, label="Simulated Delta Rates", color="green")
    fr_ax.plot(exp_time_rates, exp_delta_rates, label="Experimental Delta Rates", linestyle="--", color="orange")
    fr_ax.grid(alpha=0.3)
    if col == 0:
        fr_ax.set_ylabel("Delta Firing Rate", fontsize=12)
    else:
        fr_ax.set_yticks([])

    # Delta Fano Factors
    ff_ax = axs[3, col]
    ff_ax.plot(time_axis_ff, sim_delta_ff, label="Simulated Delta Fano", color="blue")
    ff_ax.plot(exp_time_ff, exp_delta_ff, label="Experimental Delta Fano", linestyle="--", color="red")
    ff_ax.grid(alpha=0.3)
    ff_ax.set_xlabel("Time (ms)", fontsize=12)
    if col == 0:
        ff_ax.set_ylabel("Delta Fano Factors", fontsize=12)
    else:
        ff_ax.set_yticks([])

# Unified Legend
handles, labels = ff_ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=12, frameon=False)

# Adjust Layout
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("Final_Figure_Fixed.png")





