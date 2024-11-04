from Simulation_final import simulate_model
import optuna
import matplotlib.pyplot as plt

# Parameters for the Optuna study and simulation
trials = 60
direction_range = [0, 1, 2]
num_stimuli = 24
kernel_step = 2000 // num_stimuli  # 167 ms pro Stimulus

storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
study = optuna.create_study(study_name="GP_24", storage=storage_url, load_if_exists=True,
                            direction='minimize',
                            sampler=optuna.samplers.GPSampler())  # erstellt Studie und verbindet mit SQL-Datenbank

# Display the best trial information so far
print(f"Best Trial so far: {study.best_trial.params}")
print(f"Best Penalty so far: {study.best_value}")

# Extract the best trial details
best_trial = study.best_trial
best_params = best_trial.params
best_penalty = best_trial.value

# Retrieve the optimal stimulus values
stimuli2 = [best_params[f'stimulus{i+1}'] for i in range(num_stimuli)]

# Run the simulation
simulate_model(trials, direction_range, stimuli2, kernel_step, plot=True, num_stimuli=num_stimuli, best_penalty=best_penalty)

# Plot the optimal stimuli values as a bar chart with a time axis
def plot_best_stimuli(stimuli, kernel_step):
    time_points = [i * kernel_step for i in range(len(stimuli))]  # Zeitpunkte für jeden Stimulus

    plt.figure(figsize=(10, 6))
    plt.bar(time_points, stimuli, width=kernel_step, color='skyblue', align='edge')
    plt.xlabel("Time (ms)")
    plt.ylabel("Optimal Stimulus Value")
    plt.title("Optimal Stimulus Values Over Time from Optuna Study")
    plt.savefig("Stimuli.png")
    plt.show()

# Call the plotting function with the best stimuli and kernel_step
plot_best_stimuli(stimuli2, kernel_step)


#optuna.samplers.CmaEsSampler()
