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
num_stimuli = 8

def objective(trial):

    #definition of range for stimuli
    stimuli = [trial.suggest_uniform(f'stimulus{i + 1}', 0, 1) for i in range(num_stimuli)]

    penalty = simulate_model(experimental_trials= 60, direction_range = [0, 1, 2], stim_kernel = stimuli, kernel_step= 2000/(len(stimuli)))


    return penalty #optuna minimizes this value


if __name__ == '__main__':

    print(f"Process ID: {os.getpid()}")



    Simulation_per_worker = 50


    storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
    study = optuna.create_study(study_name= "GP_8", storage= storage_url, load_if_exists = True,
                                direction = 'minimize', sampler = optuna.samplers.GPSampler())
    # erstellt studie und verbindet mit sql datenbank, erstellt objekt mit dem ich mit optuna studie interagieren kann

    study.optimize(objective, n_trials = Simulation_per_worker, n_jobs = 1)

    print(f"Best Trial: {study.best_trial.params}")
    print(f"Best Penalty: {study.best_value}")

#optuna.samplers.CmaEsSampler()