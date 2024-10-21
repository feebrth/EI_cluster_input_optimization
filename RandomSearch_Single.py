import random
import os
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import optuna
import optuna
from optuna.samplers import GPSampler
from optuna_dashboard import run_server

from Simulation_final import simulate_model


# def model(a):
#     for ii, ai in enumerate(a):
#         print("a"+str(ii) + ": " + str(ai))
#     Loss = a.T@a
#     return Loss






#model(np.array([0.2,0.15,0.1, 0.05]))
#
# def random_search(iterations = 10, num_stimuli = 2, lower_bound = 0, upper_bound = 1): #generates 20 iterations from random stimuli, in every iteration four stimuli
#     results = []
#     losses = []
#
#     for _ in range(iterations):
#         stimuli = np.random.rand(num_stimuli) * (upper_bound - lower_bound) + lower_bound
#         loss = simulate_model(stim_kernel = stimuli, trials= 5, direction_range= [0,1,2], kernel_step= 1000) # randomly generated stimuli from each iteration are passed to the model function. This function iterates through the stimuli and prints them with an index
#
#         results.append(stimuli)
#         losses.append(loss)
#
#     min_index = np.argmin(losses)
#
#     return results[min_index], losses[min_index]
#
#
# #best_stimuli, best_loss = random_search()
#
#     print(f"Best stimuli: {best_stimuli}")
#     print(f"Best loss: {best_loss}")
#
#
# def random_search_parallel(iterations=8, num_stimuli=4, lower_bound=0, upper_bound=1, trials=5, direction_range= [0,1,2], kernel_step =500):
#     stimuli_list = []
#
#     # Generate a list of random stimuli for each iteration
#     for _ in range(iterations):
#         stimuli = np.random.rand(num_stimuli) * (upper_bound - lower_bound) + lower_bound
#         stimuli_list.append(stimuli)
#
#
#     # Using multiprocessing.Pool for parallel execution
#     with multiprocessing.Pool(processes=8) as pool:
#
#         results = pool.starmap(simulate_model, [(trials, direction_range, stimuli, kernel_step) for stimuli in stimuli_list])
#
#     min_penalty = min(results)
#     min_index = results.index(min_penalty)
#     best_stimulus = stimuli_list[min_index]
#
#
#
#     return results, stimuli_list, best_stimulus


def objective(trial):

    #definition of random-search range for stimuli
    stim1 = trial.suggest_uniform('stimulus1', 0,1)
    stim2 = trial.suggest_uniform('stimulus2', 0,1)
    stim3 = trial.suggest_uniform('stimulus3', 0,1)
    stim4 = trial.suggest_uniform('stimulus4', 0,1)

    stimuli = [stim1, stim2, stim3, stim4]

    penalty = simulate_model(experimental_trials= 60, direction_range = [0, 1, 2], stim_kernel = stimuli, kernel_step= 2000/(len(stimuli)))
    #hier Anzahl experimental trials ändern

    return penalty #optuna minimizes this value


     # Return results
     #for idx, (result, stim) in enumerate(zip(results, stimuli)):
     #    print(f"Simulation {idx+1}: Penalty = {result} , Stimulus: {stim}")


# Code for plotting the results with stimuli and penalty values
# def plot_results(stimuli,penalties):
#
#     stimuli = np.array(stimuli)
#     penalties = np.array(results)
#
#
#     plt.scatter(stimuli[:, 0], stimuli[:, 1], c=penalties, cmap='viridis', s=20)
#
#     plt.colorbar(label="Penalty")
#     plt.xlim(0,1)
#     plt.ylim(0,1)
#
#
#     plt.xlabel('Stimulus 1')
#     plt.ylabel('Stimulus 2')
#     plt.title('Stimuli vs Penalty')
#
#
#     plt.show()


if __name__ == '__main__':

    print(f"Process ID: {os.getpid()}")



    Simulation_per_worker = 20

    # iterations = 5
    #results, stimuli, best_stimulus = random_search_parallel(iterations = iterations, trials = trials, direction_range = direction_range, kernel_step = kernel_step)

    #simulate_model(trials, direction_range, best_stimulus, kernel_step, plot=True)
    storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
    study = optuna.create_study(study_name= "GP_4_Stimuli", storage= storage_url, load_if_exists = True,
                                direction = 'minimize', sampler = optuna.samplers.GPSampler())
    # erstellt studie und verbindet mit sql datenbank, erstellt objekt mit dem ich mit optuna studie interagieren kann

    study.optimize(objective, n_trials = Simulation_per_worker, n_jobs = 1)

    print(f"Best Trial: {study.best_trial.params}")
    print(f"Best Penalty: {study.best_value}")



