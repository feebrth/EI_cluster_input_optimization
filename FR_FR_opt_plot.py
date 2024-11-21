import optuna
from FF_Sim import simulate_model, plot_simulated_and_experimental_data
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

best_trial = study.best_trials[0]
best_params = best_trial.params

# Retrieve the optimal stimulus values
stimuli = [best_params[f'stimulus{i + 1}'] for i in range(num_stimuli)]


# Run the simulation with the optimal stimuli
penalty_ff, penalty_rates, fano_factors, firing_rates, time_axis_ff, time_axis_rates, exp_time_ff, exp_ff, exp_time_rates, exp_rates = simulate_model(
    experimental_trials=trials,
    direction_range=direction_range,
    stim_kernel=stimuli,
    kernel_step=kernel_step,
    plot=False,
    use_delta=True,
)

plot_simulated_and_experimental_data(
    simulated_ff=fano_factors,
    simulated_rates=firing_rates,
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









optuna.visualization.plot_pareto_front(study, target_names=["Penalty FF", "Penalty Rates"])
plt.show()
