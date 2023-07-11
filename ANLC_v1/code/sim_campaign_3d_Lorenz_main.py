#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:44:12 2022

@author: Davide Grande

The main file to run a simulation campaign for the training of a Lorenz system.

    This scripts executes recursive callbacks to the ANLC training file and 
    saves the statistics at the end. 
    - An incremental seed is passed to the ANLC file (seed_);
            
    This script is used to generate the results for the ANLC paper.
    

"""

#
# Preprocessing
#
clear_ws = False     # clear the variable and the console

if clear_ws:
    from IPython import get_ipython
    try:
        get_ipython().magic('clear')
        get_ipython().magic('reset -f')
    except clear_ws.Failed:
        pass


import time
import timeit
import os
import sys
from datetime import datetime
import random

import dreal
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import torch.onnx
import copy
import math
from IPython import get_ipython
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete, dlti, dstep, lti
import itertools

import closed_loop_testing.cl_3d_Lorenz as cl
import functions.compute_Lie_der as compute_Lie_der
import systems_dynamics.dynamics_3d_Lorenz as dynamic_sys
from utilities.ann_definition import Net
import utilities.sym_dyn_3d_Lorenz as sym_dyn
import utilities.Functions as Functions
import utilities.from_dreal_to_np as from_dreal_to_np
import utilities.saving_log as saving_log
import configuration.config_3d_Lorenz as config_file


'''
Seed and number of the runs to be saved
'''
init_seed = 3  # initial seed
torch.set_default_dtype(torch.float32)  # setting default tensor type
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Campaign parameters
campaign_run = 1475  # number of the run.
                     # The results will be saved in sim_campaign/results/campaign_'campaign_run'
tot_runs = 10        # total number of run of the campaigns (each with different seed)
max_loop_number = 1  # number of loops per run (>1 means that the weights will be re-initialised).
                     # default = 1
max_iters = 1000     # number of maximum learning iterations per run
system_name = "control_3d_Lorenz_campaign"
execute_postpro = True  #execute postprocessing

# SMT parameters
config = dreal.Config()
config.use_polytope_in_forall = True
config.precision = 1e-6  # delta-precision
epsilon = 0.0            # parameters to relax the Lie derivative conditions, to be kept at 0. 
gamma_underbar = 0.5     # domain lower boundary
gamma_overbar = 2.0      # domain upper boundary


# Loss function
alpha_1 = 1.0   # weight V
alpha_2 = 1.0   # weight V_dot
alpha_3 = 0.0   # weight V0
alpha_4 = 0.1   # weight tuning term V
alpha_roa = gamma_overbar  # Lyapunov function steepness
alpha_5 = 1.0   # general scaling factor


'''
Initialising directories (this prevents overwriting old results)
'''
# Generating result (top) directory
try:
    folder_results = "sim_campaign/results"
    current_dir = os.getcwd()
    final_dir = current_dir + "/" + folder_results
    os.mkdir(final_dir)

except OSError:
    print(f"\nResult directory: \n{final_dir} already existing!\n")
else:
    print(f"\nResult directory SUCCESSFULLY created as: \n{final_dir}\n")

# Generating campaign directory
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

    seed_ += 1  # incrementing seed over each run

    # callback to the training file
    exec(open("sim_campaign_3d_Lorenz_train.py").read())

    # Saving convergence information for final statistics
    time_elapse_hist[i_loop] = (stop - start)
    iterations_hist[i_loop] = (tot_iters)
    conv_hist[i_loop] = found_lyap_f
    to_learner_hist[i_loop] = to_learner
    to_fals_hist[i_loop] = to_fals
    if found_lyap_f:
        count_conv += 1
    if to_fals:
        to_fals_check[i_loop] = fals_to_check


'''
Final statistics
'''
stop_stat = timeit.default_timer()
print(f"Average running time = {time_elapse_hist.mean()} ['']")
print(f"Average iteration per run = {iterations_hist.mean()}")

# Postpro clock
minutes_elapsed_postp = (stop_stat - start_stat) / 60

# Saving statistic
result_stat = [f"Run of the controlled {system_name} system campaign number {campaign_run}.\n"+
               f"The control weights were intialised = {parameters['control_initialised']}\n" + 
               f"\n{tot_runs} tests were run\n"+
               f"The seeds were cycled from {init_seed} to {seed_}.\n" +
               "\nThe overall time for the statistic generation was: " +
               f"{minutes_elapsed_postp} [']" +
               f"\n\nRun time (per test) (mu+-3sigma) = " +
               f"{time_elapse_hist.mean()}+-{3*time_elapse_hist.std()} ['']" +
               f"\nNumber of iterations (per test) (mu+-3sigma) = " +
               f"{iterations_hist.mean()}+-{3*iterations_hist.std()}" +
               f"\nConvergence history = {conv_hist}" + 
               f"\nTO learner history = {to_learner_hist}" + 
               f"\nTO falsifier history = {to_fals_hist}" + 
               f"\nTO falsifier check [s] = {to_fals_check}" + 
               f"\n\nConverged tests = {count_conv}/{tot_runs}" +
               f"\n\nElapsed time history = {time_elapse_hist} ['']" + 
               f"\nIteration history = {iterations_hist}" + 
               f"\n\nTotal Learner not-converged = {tot_learner_to}" +
               f"\nTotal Falsifier not-converged = {tot_falsifier_to}"
               ]

np.savetxt(final_dir_campaign + "/statistics.txt", result_stat, fmt="%s")

