import optuna
from FF_Sim import simulate_model, plot_simulated_and_experimental_data
import matplotlib.pyplot as plt

# Parameters for the Optuna study and simulation
trials = 5
direction_range = [0]
num_stimuli = 8
kernel_step = 2000 // num_stimuli  # 167 ms pro Stimulus

# Optuna Study laden
storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
    #"mysql://optuna:password@127.0.0.1:3306/optuna_db"
study = optuna.load_study(study_name="8_FR_FF", storage=storage_url)

# Pareto-Front visualisieren
fig = optuna.visualization.plot_pareto_front(study, target_names=["Loss FF", "Loss Rates"])

fig.write_image("8_Pareto_Front.png")


# Best Trials nach Fano-Faktor sortieren
sorted_trials = sorted(study.best_trials, key=lambda trial: trial.values[0])  # Nach "Loss FF" sortieren

# Alle Best Trials simulieren und plotten
for idx, trial in enumerate(sorted_trials[:5]):
    best_params = trial.params
    print(f"Simulating Trial {idx}:")
    print("  Loss FF:", trial.values[0])
    print("  Loss Rates:", trial.values[1])
    print("  Stimulus Parameters:", best_params)

    # Retrieve the optimal stimulus values
    stimuli = [best_params[f'stimulus{i + 1}'] for i in range(num_stimuli)]

    # Run the simulation with the optimal stimuli
    (sim_fano_factors, sim_firing_rates, time_axis_ff, time_axis_rates,
     exp_time_ff, exp_ff, exp_time_rates, exp_rates, penalty_ff, penalty_rates) = simulate_model(
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

    # Speichere den Plot mit der zugehörigen Trial-Nummer
    plt.savefig(f"{num_stimuli}Stimuli_Trial_{idx}_LossFF_{trial.values[0]:.3f}_test.png")
    plt.close()  # Schließe die aktuelle Figure, um Speicher freizugeben