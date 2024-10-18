from Simulation import simulate_model
import optuna

trials = 5
direction_range = [0,1,2]
num_stimuli = 4
kernel_step = (2000 // num_stimuli)


storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
study = optuna.create_study(study_name="RandomSearch3", storage=storage_url, load_if_exists=True,
                            direction='minimize',
                            sampler=optuna.samplers.RandomSampler())  # erstellt studie und verbindet mit sql datenbank, erstellt objekt mit dem ich mit optuna studie interagieren kann

print(f"Best Trial so far: {study.best_trial.params}")
print(f"Best Penaltyso far: {study.best_value}")

best_params = study.best_trial.params
stimuli2 = [best_params['stimulus1'], best_params['stimulus2'], best_params['stimulus3'], best_params['stimulus4']]

simulate_model(trials, direction_range, stimuli2, kernel_step, plot=True)

