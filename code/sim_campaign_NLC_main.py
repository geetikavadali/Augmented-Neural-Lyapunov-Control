#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 20:16:44 2022

@author: Davide Grande

The main file to run the statistics for an NLC file: 
    this scripts executes recursive callbacks to the NLC training file and 
    saves the statistics at the end. 
    - An incremental seed is passed to the NLC file (seed_);

    
    CAVEAT: the file with the statistics is generated within the folder of the 
            last run.
            
    This scrip is used to generate the results for the ANLC paper and 
    showing that our system does not require weights initialisation.
    

"""

import numpy as np 
import timeit 
from dreal import *
import utilities.Functions as Functions
import torch 
import torch.nn.functional as F
import numpy as np
import timeit 
import matplotlib.pyplot as plt
import os
import sys
import copy
import time


'''
Seed and number of the runs to be saved
'''
init_seed = 1               # initial seed
control_initialised = True  # initialised control ANN with pre-computed LQR law
torch.set_default_dtype(torch.float32)  # setting default tensor type

config = Config()
config.use_polytope_in_forall = True
config.precision = 1e-6
epsilon = 0.01
ball_lb = 0.1
ball_ub = 6
loss_function_with_tuning = True   # use loss function with tuning terms


max_loops = 1
max_iters = 1000

campaign_run = 3  # number of the initial run
tot_runs = 50        # total number of tests to be run

'''
Initialising directorories (this prevents overwriting old results)
'''
## Generating result directories
# top directory
try:
    folder_results = "sim_campaign/results"
    current_dir = os.getcwd()
    final_dir = current_dir + "/" + folder_results
    os.mkdir(final_dir)

except OSError:
    print(f"\nResult directory: \n{final_dir} already existing!\n")
else:
    print(f"\nResult directory SUCCESSFULLY created as: \n{final_dir}\n")

# specific directory
try:
    folder_results_campaign = "campaign_" + str(campaign_run)
    current_dir = os.getcwd()
    final_dir_campaign = folder_results + "/" + folder_results_campaign
    os.mkdir(final_dir_campaign)

except OSError:
    print(f"\nResult directory: \n{final_dir_campaign} already existing!\n")
    print("\n\nERROR: Simulation campaign not started as you were overwriting old results.")
    print("\nCheck the 'campaign_run' parameter first!")
    sys.exit()
else:
    print(f"\nResult directory SUCCESSFULLY created as: \n{final_dir_campaign}\n")



'''
Main loop
'''
seed_ = init_seed - 1
time_elapse_hist = np.zeros(tot_runs)
iterations_hist = np.zeros(tot_runs)
conv_hist = np.zeros(tot_runs)  # convergence history
to_learner_hist = np.zeros(tot_runs)  # learner TO history
to_fals_hist = np.zeros(tot_runs)  # falsifier TO history
to_fals_check = np.zeros(tot_runs)  # falsifier TO history check
start_stat = timeit.default_timer()  # initialise timer
tot_falsifier_to = 0  # total number of falsifier time out
tot_learner_to = 0  # total number of learner time out
count_conv = 0  # total number of coverged tests

for i_loop in range(tot_runs):

    seed_ += 1

    exec(open("sim_campaign_NLC_train.py").read())
    
    # Saving run info
    time_elapse_hist[i_loop] = (stop - start)
    iterations_hist[i_loop] = (tot_iters)
    conv_hist[i_loop] = valid
    to_learner_hist[i_loop] = to_learner
    to_fals_hist[i_loop] = to_fals
    to_fals_check[i_loop] = fals_to_check
    if valid:
        count_conv += 1

'''
Final statistics
'''
stop_stat = timeit.default_timer()

print(f"Average running time = {time_elapse_hist.mean()} ['']")
print(f"Average iteration per run = {iterations_hist.mean()}")

# Postpro clock
minutes_elapsed_postp = (stop_stat - start_stat) / 60


# Saving statistic in the folder of the last test
result_stat = [f"Run of the NLC campaign number {campaign_run}.\n"+
               f"The control weights were intialised = {control_initialised}\n" + 
               f"\n{tot_runs} tests were run\n"+
               f"The seeds were cycled from {init_seed} to {seed_}.\n" +
               "\nThe overall time for the statistic generation was: " +
               f"{minutes_elapsed_postp} [']" +
               f"\n\nRun time (per test) (mu+-3sigma) = " +
               f"{time_elapse_hist.mean()}+-{3*time_elapse_hist.std()} ['']" +
               f"\nNumber of iterations (per test) (mu+-3sigma) = " +
               f"{iterations_hist.mean()}+-{3*iterations_hist.std()}" +
               f"\n\nElapsed time history = {time_elapse_hist} ['']" + 
               f"\nIteration history = {iterations_hist}" + 
               f"\nConvergence history = {conv_hist}" + 
               f"\nTO learner history = {to_learner_hist}" + 
               f"\nTO falsifier history = {to_fals_hist}" + 
               f"\nTO falsifier check [s] = {to_fals_check}" + 
               f"\n\nConverged tests = {count_conv}/{tot_runs}" +
               f"\n\nTotal Learner not-converged = {tot_learner_to}" +
               f"\nTotal Falsifier not-converged = {tot_falsifier_to}"
               ]

np.savetxt(final_dir_campaign + "/statistics.txt", result_stat, fmt="%s")



