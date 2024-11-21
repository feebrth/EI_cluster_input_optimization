from Simulation_final import simulate_model
import optuna
import matplotlib.pyplot as plt


# Parameters for the Optuna study and simulation
trials = 60
direction_range = [0, 1, 2]
num_stimuli = 24
kernel_step = 2000 // num_stimuli  # 167 ms pro Stimulus

storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
sampler = optuna.samplers.TPESampler()
study = optuna.create_study(study_name= "GP_24_FR_FF", storage= storage_url, load_if_exists = True,
                            directions = ['minimize', 'minimize'], sampler = sampler)

stimuli = [trial.params[f'stimulus{j + 1}'] for j in range(num_stimuli)]

# Run the simulation with the optimal stimuli
penalty_ff, penalty_rates, sim_ff, sim_rates, time_axis_ff, time_axis_rates, _, _, _ = simulate_model(
    experimental_trials=trials,
    direction_range=direction_range,
    stim_kernel=stimuli,
    kernel_step=kernel_step,
    use_delta=True,
    plot=False
)










optuna.visualization.plot_pareto_front(study, target_names=["Penalty FF", "Penalty Rates"])
plt.show()
