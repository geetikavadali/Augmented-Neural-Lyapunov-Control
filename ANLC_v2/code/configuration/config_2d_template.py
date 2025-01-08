#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:39:58 2023

@authors: Davide Grande
          Andrea Peruffo

A function collecting the parameters for the synthesis of the CLF.

"""

import numpy as np
import torch


def set_params():

    campaign_params = {
        
        'init_seed': 1,        # initial campaign seed
        'campaign_run': 1500,  # number of the run.
                               # The results will be saved in /results/campaign_'campaign_run'
        'tot_runs': 5,        # total number of runs of the campaigns (each one with a different seed)
        'max_loop_number': 1,  # number of loops per run (>1 means that the weights will be re-initialised).
                                # default value = 1.
        'max_iters': 1000,     # number of maximum learning iterations per run
        'system_name': "test_2d_pendulum",  # name of the systems to be controlled
        'x_star': torch.tensor([0.0, 0.0]),  # target equilibrium point
    }        

    # Parameters for learner
    learner_params = {
        'N': 500,  # initial dataset size
        'N_max': 1000,  # maximum dataset size (if using a sliding window)
        'sliding_window': True,  # use sliding window
        'learning_rate': 0.1,  # learning rate Lyapunov branch
        'learning_rate_c': 0.1,  # learning rate control branch
        'use_scheduler': True,
        # use LR scheduler to allow dynamic learning rate reducing based on some validation measurements
        'sched_T': 300,  # cosine annealing scheduler period
        'print_interval': 200,  # interval of loss function printouts
    }

    # Parameters for Lyapunov ANN
    lyap_params = {
        'n_input': 2, # input dimension (n = n-dimensional system)
        'beta_sfpl': 2,  # the higher, the steeper the Softplus, the better approx. sfpl(0) ~= 0
        'clipping_V': True,  # clip weight of Lyapunov ANN
        'size_layers': [10, 10, 1],  # CAVEAT: the last entry needs to be = 1 (this ANN outputs a scalar)!
        'lyap_activations': ['pow2', 'linear', 'linear'],
        'lyap_bias': [False, False, False],
    }

    # Parameters for control ANN
    control_params = {
        'use_lin_ctr': True,  # use linear control law  -- defined as 'phi' in the publication
        'lin_contr_bias': False,  # use bias on linear control layer
        'control_initialised': False,  # initialised control ANN with pre-computed LQR law
        'init_control': torch.tensor([[-23.58639732, -5.31421063]]),  # initial control solution
        'size_ctrl_layers': [50, 1],  # CAVEAT: the last entry is the number of control actions!
        'ctrl_bias': [True, False],
        'ctrl_activations': ['tanh', 'linear'],
        'use_saturation': False,        # use saturations in the control law.
        'ctrl_sat': [18.3],             # actuator saturation values: 
                                        # this vector needs to be as long as 'size_ctrl_layers[-1]' (same size as the control vector).
    }

    falsifier_params = {
        # a) SMT parameters
        'gamma_underbar': 0.1,  # domain lower boundary
        'gamma_overbar': 6.0,   # domain upper boundary
        'zeta_SMT': 200,  # how many points are added to the dataset after a CE box
                          # is found
        'epsilon': 0.0,   # parameters to further relax the SMT check on the Lie derivative conditions.
                          # default value = 0 (inspect utilities/Functions/CheckLyapunov for further info).
                          
        # b) Discrete Falsifier parameters
        'grid_points': 50,  # sampling size grid
        'zeta_D': 50,  # how many points are added at each DF callback
    }


    loss_function = {
        # Loss function tuning
        'alpha_1': 1.0,  # weight V
        'alpha_2': 1.0,  # weight V_dot
        'alpha_3': 1.0,  # weight V0
        'alpha_4': 0.0,  # weight tuning term V
        'alpha_roa': 0.1*falsifier_params['gamma_overbar'],  # Lyapunov function steepness
        'alpha_5': 1.0,  # general scaling factor    
    }

    # Parameters specific to the dynamic system
    # CAVEAT: if you want to save the parameters, make sure to also report them
    #  in the following function 'get_dyn_sys_params()'
    dyn_sys_params = {
        'G': 9.81,  # gravity constant
        'L': 0.5,  # length of the pole
        'm': 0.15,  # ball mass
        'b': 0.1,  # friction
    }

    # Postprocessing parameters
    postproc_params = {
        'execute_postprocessing': True,  # True: triggers the generation of the plots described below
        'verbose_info': True,  # print info with high verbosity
        'dpi_': 300,  # DPI number for plots
        'plot_V': True,
        'plot_Vdot': True,
        'plot_u': True,
        'plot_4D_': True,  # plot 4D Lyapunov f, Lie derivative and control function
        'n_points_4D': 500,
        'n_points_3D': 100,
        'compare_first_last_iters': True,  # saving V, Vdot at the first iter to compare with the final res
        'plot_ctr_weights': True,
        'plot_V_weights': True,
        'plot_dataset': True,
    }

    # Closed-loop system testing parameters
    closed_loop_params = {
        'test_closed_loop_dynamics': True,
        'end_time': 50.0,  # [s] time span of closed-loop tests
        'Dt': 0.01,  # [s] sampling time for Forward Euler integration
    }

    # joining all the parameters in a single dictionary
    parameters = {**campaign_params,
                  **learner_params,
                  **lyap_params,
                  **control_params,
                  **falsifier_params,
                  **loss_function,
                  **dyn_sys_params,
                  **postproc_params,
                  **closed_loop_params}


    return parameters, dyn_sys_params

