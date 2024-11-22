import optuna
from FF_Sim import simulate_model, plot_simulated_and_experimental_data
import matplotlib.pyplot as plt

# Parameters for the Optuna study and simulation
trials = 60
direction_range = [0, 1, 2]
num_stimuli = 24
kernel_step = 2000 // num_stimuli  # 167 ms pro Stimulus

# Optuna Study laden
storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
study = optuna.load_study(study_name="24_FR_FF", storage=storage_url)

# Pareto-Front visualisieren
fig = optuna.visualization.plot_pareto_front(study, target_names=["Loss FF", "Loss Rates"])
#plt.title("Pareto Front")
#plt.savefig("Pareto_Front.png")
fig.write_image("Pareto_Front.png")

# Optional: Alle Pareto-Lösungen anzeigen
print(f"Number of Pareto solutions: {len(study.best_trials)}")
for i, trial in enumerate(study.best_trials):
    print(f"Pareto solution {i}:")
    print("  Loss FF:", trial.values[0])
    print("  Loss Rates:", trial.values[1])
    print("  Stimulus Parameters:", trial.params)

# Wähle einen spezifischen Trial aus der Pareto-Front (z. B. den ersten)
selected_trial = study.best_trials[0]  # Alternativ: Lass den Benutzer einen Trial auswählen
best_params = selected_trial.params
print(f"Selected Trial Penalties: {selected_trial.values}")

# Retrieve the optimal stimulus values
stimuli = [best_params[f'stimulus{i + 1}'] for i in range(num_stimuli)]
kernel_step = 2000 // num_stimuli

# Run the simulation with the optimal stimuli
sim_fano_factors, sim_firing_rates, time_axis_ff, time_axis_rates, exp_time_ff, exp_ff, exp_time_rates, exp_rates, penalty_ff, penalty_rates = simulate_model(
    experimental_trials=trials,
    direction_range=direction_range,
    stim_kernel=stimuli,
    kernel_step=kernel_step,
    plot=False,
    use_delta=True,
)

# Plot the results with the optimal stimuli
plot_simulated_and_experimental_data(
    simulated_ff=sim_fano_factors,
    simulated_rates=sim_firing_rates,
    time_axis_ff=time_axis_ff,
    time_axis_rates=time_axis_rates,
    exp_time_ff=exp_time_ff,
    exp_ff=exp_ff,
    exp_time_rates=exp_time_rates,
    exp_rates=exp_rates,
    stim_kernel=stimuli,
    kernel_step=kernel_step,
    plot_delta=True  # Zeigt auch die Delta-Werte
)
plt.savefig("Optimal_Simulation.png")