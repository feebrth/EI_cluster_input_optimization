import os
import optuna
from FF_Sim import simulate_model

num_stimuli = 8
def objective(trial):

    # Dynamische Definition der Stimuli über eine Schleife

    stimuli = [trial.suggest_uniform(f'stimulus{i + 1}', 0, 1) for i in range(num_stimuli)]

    # Berechne die Schrittgröße für das Stimulus-Kernel
    kernel_step = round(2000 / len(stimuli))

    # Simuliere das Modell mit den vorgeschlagenen Stimuli
    _, _, _, _, _, _, _, _, penalty_ff, penalty_rates = simulate_model(
        experimental_trials=60,  # Anzahl der Trials
        direction_range=[0, 1, 2],  # Richtungsbereich
        stim_kernel=stimuli,  # Stimuli-Kernel
        kernel_step=kernel_step,  # Schrittgröße
        use_delta=True,  # Optional: Delta-basierte Penalty-Berechnung
        plot=False  # Kein Plot während der Optimierung
    )

    return penalty_ff, penalty_rates



if __name__ == '__main__':
    # read env variable DEBUG
    debug = int(os.environ.get('DEBUG', 0))

    print(f"Process ID: {os.getpid()}")
    Simulation_per_worker = 50


    storage_url = "mysql+pymysql://optuna:optuna@192.168.1.10:3307/optuna_Fee"
    sampler = optuna.samplers.TPESampler()
    if debug==1:
        print("DEBUG MODE")
        study = optuna.create_study(study_name= f"{num_stimuli}_FR_FF", load_if_exists = True,
                                directions = ['minimize', 'minimize'], sampler = sampler)
    else:
        study = optuna.create_study(study_name= f"{num_stimuli}_FR_FF", storage= storage_url, load_if_exists = True,
                                directions = ['minimize', 'minimize'], sampler = sampler)



    study.optimize(objective, n_trials = Simulation_per_worker, n_jobs = 1)

    print("Number of Pareto solutions:", len(study.best_trials))
    for i, trial in enumerate(study.best_trials):
        print(f"Pareto solution {i}:")
        print("  Penalty FF:", trial.values[0])
        print("  Penalty Rates:", trial.values[1])