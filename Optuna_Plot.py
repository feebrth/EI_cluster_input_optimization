from Simulation_final import simulate_model
import optuna

trials = 60
direction_range = [0,1,2]
num_stimuli = 12
kernel_step = (2000 // num_stimuli)


storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
study = optuna.create_study(study_name="GP_12", storage=storage_url, load_if_exists=True,
                            direction='minimize',
                            sampler=optuna.samplers.GPSampler())  # erstellt studie und verbindet mit sql datenbank, erstellt objekt mit dem ich mit optuna studie interagieren kann

print(f"Best Trial so far: {study.best_trial.params}")
print(f"Best Penalty so far: {study.best_value}")

best_trial = study.best_trial
best_params = best_trial.params
best_penalty = best_trial.value

stimuli2 = [best_params[f'stimulus{i+1}'] for i in range(num_stimuli)]

simulate_model(trials, direction_range, stimuli2, kernel_step, plot=True, num_stimuli = num_stimuli, best_penalty = best_penalty)




#optuna.samplers.CmaEsSampler()
