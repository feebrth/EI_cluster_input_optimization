import random
import os
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import RandomSampler
from optuna.samplers import GPSampler
from optuna_dashboard import run_server

from Simulation_final import simulate_model

def objective(trial):

    #definition of random-search range for stimuli
    stim1 = trial.suggest_uniform('stimulus1', 0,1)
    stim2 = trial.suggest_uniform('stimulus2', 0,1)
    stim3 = trial.suggest_uniform('stimulus3', 0,1)
    stim4 = trial.suggest_uniform('stimulus4', 0,1)
    stim5 = trial.suggest_uniform('stimulus5', 0,1)
    stim6 = trial.suggest_uniform('stimulus6', 0,1)
    stim7 = trial.suggest_uniform('stimulus7', 0,1)
    stim8 = trial.suggest_uniform('stimulus8', 0,1)
    stim9 = trial.suggest_uniform('stimulus9', 0, 1)
    stim10 = trial.suggest_uniform('stimulus10', 0, 1)
    stim11 = trial.suggest_uniform('stimulus11', 0, 1)
    stim12 = trial.suggest_uniform('stimulus12', 0, 1)

    stimuli = [stim1, stim2, stim3, stim4,stim5,stim6,stim7,stim8,stim9,stim10,stim11,stim12]
    num_stimuli = len(stimuli)

    penalty = simulate_model(experimental_trials= 60, direction_range = [0, 1, 2], stim_kernel = stimuli, kernel_step= round(2000/(len(stimuli))), num_stimuli = len(stimuli))
    #hier Anzahl experimental trials ändern

    return penalty #optuna minimizes this value


if __name__ == '__main__':


    print(f"Process ID: {os.getpid()}")



    Simulation_per_worker = 50

    # iterations = 5
    #results, stimuli, best_stimulus = random_search_parallel(iterations = iterations, trials = trials, direction_range = direction_range, kernel_step = kernel_step)

    #simulate_model(trials, direction_range, best_stimulus, kernel_step, plot=True)
    storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"

    study = optuna.create_study(study_name= "GP_12", storage= storage_url, load_if_exists = True,
                                direction = 'minimize', sampler = optuna.samplers.GPSampler())
    # erstellt studie und verbindet mit sql datenbank, erstellt objekt mit dem ich mit optuna studie interagieren kann

    study.optimize(objective, n_trials = Simulation_per_worker, n_jobs = 1)

    print(f"Best Trial: {study.best_trial.params}")
    print(f"Best Penalty: {study.best_value}")

#optuna.samplers.CmaEsSampler()
#optuna.samplers.GPSampler()