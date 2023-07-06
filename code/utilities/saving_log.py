#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:53:31 2023

@author: Davide Grande

A function to save the log file of a training run.

"""

import numpy as np
from datetime import datetime
import timeit
import time

def gen_log(system_name, found_lyap_f, seed_, max_loop_number, max_iters, 
            parameters, x, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, 
            alpha_roa, epsilon, gamma_underbar, gamma_overbar, config, model,
            to_fals, to_learner, seconds_elapsed, minutes_elapsed, hours_elapsed,
            out_iters, i_epoch, start, init_date, end_date, falsifier_elapsed,
            final_dir_run, x_star):

    print("Saving logs ...\n")
        
    dt_string_init = init_date.strftime("%d/%m/%Y %H:%M:%S")  # date init training
    dt_string_end = end_date.strftime("%d/%m/%Y %H:%M:%S")  # date end training
    
    # Postpro clock
    stop_postp = timeit.default_timer()
    minutes_elapsed_postp = (stop_postp - start) / 60
    
    result_report = [f"Run of the {system_name} system.\n" + 
                     f"Convergence reached = {found_lyap_f}\n\n" +
                     "TRAINING PARAMS: \n" + 
                     f"Seed = {seed_}\n" +
                     f"max_loop_number = {max_loop_number}\n" +
                     f"max_iters = {max_iters}\n" +
                     f"Initial dataset dimension = {parameters['N']}\n" +
                     f"Final dataset dimension = {len(x)}\n" + 
                     f"Using a sliding window = {parameters['sliding_window']}\n" +
                     f"Maximum dataset dimension (if using sliding wind) = {parameters['N_max']}\n" +
                     f"Equilibrium (x_star) = {x_star}\n" +
                     "\n\n" +
                     "LYAPUNOV ANN: \n" + 
                     f"layer 1 dim. = {parameters['lyap_hid1']}\n" +
                     f"layer 1 act. f. = {parameters['Lyap_act_fun1']}\n" +
                     f"layer 1 has bias = {parameters['Lyap_bias1']}\n" +
                     f"layer 2 dim. = {parameters['lyap_hid2']}\n" +
                     f"layer 2 has act. f. = {parameters['Lyap_act_fun2']}\n" +
                     f"layer 2 has bias = {parameters['Lyap_bias2']}\n" +
                     f"layer 3 dim. = {parameters['n_output']}\n" +
                     f"layer 3 has act. f. = {parameters['Lyap_act_fun3']}\n" +
                     f"layer 3 has bias = {parameters['Lyap_bias3']} \n" +
                     f"beta_sfpl = {parameters['beta_sfpl']}\n" +
                     f"Clipping Lyapunov weights = {parameters['clipping_V']}\n" +
                     "\n\n" +
                     "CONTROL ANN: \n" +
                     f"Use linear control = {parameters['use_lin_ctr']}\n" + 
                     "If nonlinear control law is used, then:\n" + 
                     f"dim. layer 1 = {parameters['contr_hid1']}\n" +
                     f"use bias layer 1 = {parameters['contr_bias1']}\n" +
                     f"layer 1 act. f. = {parameters['contr_act_fun1'] }\n" +
                     f"dim layer 2 = {parameters['contr_hid2']}\n" +
                     f"use bias layer 2 = {parameters['contr_bias2']}\n" +
                     f"layer 2 act. f. = {parameters['contr_act_fun2']}\n" +
                     "\n\n" +
                     "LEARNER: \n" + 
                     f"Learning rate Lyap. = {parameters['learning_rate']}\n" +
                     f"Learning rate control = {parameters['learning_rate_c']}\n" + 
                     f"use l.r. scheduler = {parameters['use_scheduler']}\n" +
                     f"scheduler period = {parameters['sched_T']}\n" +
                     "\nLYAPUNOV RISK:\n" +
                     f"alpha_1 (Weight V) = {alpha_1}\n" +
                     f"alpha_2 (Weight V_dot) = {alpha_2}\n" +
                     f"alpha_3 (Weight V0) = {alpha_3}\n" +
                     f"alpha_4 (Weight V tuning) = {alpha_4}\n" +
                     f"alpha_5(overall weight) = {alpha_5}\n" +
                     f"alpha_roa (ROA tuning) = {alpha_roa}\n" +
                     "\n\n" +
                     "TRAINING: \n" +
                     f"ANN control initialised (LQR) = {parameters['control_initialised']}\n" +
                     f"ANN control initial weights = {parameters['init_control'] }\n" +
                     f"ANN control final weights = {model.control.weight.data.detach()}\n" +
                     "\n\n" +
                     "FALSIFIER (SMT): \n" +
                     f"Epsilon = {epsilon}\n" +
                     f"Falsifier domain = {gamma_underbar} --- {gamma_overbar}\n" +
                     f"config.precision = {config.precision}\n"
                     f"zeta_SMT (SMT CE point cloud) = {parameters['zeta_SMT']}" +
                     "\n\n" + 
                     "DISCRETE FALSIFIER (DF): \n" +  
                     f"Use DF = {parameters['use_disc_falsifier']}\n" + 
                     f"zeta_D (DF CEs added at each callback) = {parameters['zeta_D']} \n" +
                     "\n\n" +
                     "RESULTS: \n" +
                     f"Falsifier Time Out = {to_fals}\n" +
                     f"Learner Time Out = {to_learner}\n" +
                     "Time elapsed:\n" +
                     f"seconds = {seconds_elapsed}\n" +
                     f"minutes = {minutes_elapsed}\n" +
                     f"hours = {hours_elapsed}\n" +
                     f"Falsifier time [']: {falsifier_elapsed}\n" +
                     f"Falsifier time [%]: {falsifier_elapsed/minutes_elapsed*100}\n" +
                     f"Posprocessing time ['] = {minutes_elapsed_postp}\n" +
                     f"Training iterations completed = {(out_iters)}\n" +
                     f"Training epochs (last iteration) = {i_epoch}\n\n" + 
                     f"Training start = {dt_string_init}\n" +
                     f"Training end = {dt_string_end}"          
                     ]

    np.savetxt(final_dir_run + "/logs.txt", result_report, fmt="%s")