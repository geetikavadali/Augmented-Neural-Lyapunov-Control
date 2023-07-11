#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:39:58 2023

@author: Davide Grande

A function collecting the parameters of the training.

"""

import torch


def get_params():

    # Parameters for learner
    learner_params = {
        'N': 50,                        # initial dataset size
        'N_max': 1000,                  # maximum dataset size (if using a sliding window)
        'sliding_window': True,         # use sliding window
        'control_initialised': False,   # initialised control ANN with pre-computed LQR law
        'learning_rate': 0.01,          # learning rate Lyapunov branch
        'learning_rate_c': 0.1,         # learning rate control branch
        'use_scheduler': True,          # use LR scheduler to allow dynamic learning rate reducing based on some validation measurements
        'sched_T': 400                  # cosine annealing scheduler period
    }
    
    
    
    # Parameters for Lyapunov ANN
    lyap_params = {
        'n_input': 3,                   # input dimension (do not change it!)
        'lyap_hid1': 8,                # hidden dimension Lyapunov layer 1
        'lyap_hid2': 8,                # hidden dimension Lyapunov layer 2
        'n_output': 1,                  # output dimension Lyapunov (do not change it!)
        'Lyap_act_fun1': 'pow2',        # activation function options:  'tanh', 'pow2', 'sfpl', 'linear' 
        'Lyap_act_fun2': 'linear',
        'Lyap_act_fun3': 'linear',
        'Lyap_bias1': False,            # use bias on 1st layer
        'Lyap_bias2': False,            # use bias on 2nd layer
        'Lyap_bias3': False,            # use bias on 3rd layer  - Leave this to False
        'beta_sfpl': 2,                 # the higher, the steeper the Softplus, the better approx. sfpl(0) ~= 0
        'clipping_V': True              # clip weight of Lyapunov ANN
    }
    
    # Parameters for control ANN
    control_params = {
        'use_lin_ctr': False,           # use linear control law  -- defined as 'phi' in the publication
        'lin_contr_bias': False,        # use bias on linear control layer
        'contr_hid1': 8,                # hidden dimension control layer 1 (if nonlinear ctr is used)
        'contr_hid2': 8,                # hidden dimension Lyapunov layer 2 (if nonlinear ctr is used)
        'contr_out': 3,                 # output dimension control ANN
        'contr_bias1': True,           # use bias on 1st layer (if nonlinear ctr is used)
        'contr_bias2': True,           # use bias on 2nd layer (if nonlinear ctr is used)
        'contr_bias3': False,           # use bias on 3rd layer (if nonlinear ctr is used)
        'contr_act_fun1': 'tanh',       # activation function options:  'tanh', 'sfpl', 'linear'
        'contr_act_fun2': 'tanh',       # activation function options:  'tanh', 'sfpl', 'linear'
        'contr_act_fun3': 'linear',     # activation function options:  'tanh', 'sfpl', 'linear'
        'init_control': -torch.tensor([[-7., 10., -0.        ],
                                       [28.,  1., -0.        ],
                                       [-0., -0., -1.66666667]])  # initial solution
    }


    falsifier_params = {
        'zeta_SMT': 200,                # how many points are added to the dataset after a CE box 
                                        # is found
        'use_disc_falsifier': True,     # use both SMT and Discrete Falsifier
        'grid_points': 50,              # sampling size grid
        'zeta_D': 50,                   # how many points are added at each DF callback
    }



    # Parameters for the dynamic system
    dyn_sys_params = {
        'sigma': 10,
        'b': 8/3,
        'r': 28,
    }
    
    # Postprocessing parameters
    postproc_params = {
        'debug_info': True, # print debug info
        'dpi_': 300,        # DPI number for plots
        'plot_V': True,
        'plot_Vdot': True,
        'plot_u': True,
        'plot_4D_': False,  # plot 4D Lyapunov f, Lie derivative and control function
        'n_points_4D': 500,
        'n_points_3D': 100,
        'compare_first_last_iters': True,  # saving V, Vdot at the first iter to compare with the final res
        'save_pdf': False,
        'plot_ctr_weights': False,
        'plot_V_weights': False,
        'steps_level_set': 10,
        'steps_roa': 10,
    }
    
    # Closed-loop system testing parameters
    closed_loop_params = {
        'test_closed_loop_dynamics': True,  
        'end_time': 10.0,       # [s] time span of closed-loop tests
        'Dt': 0.01,             # [s] sampling time for Forward Euler integration
        'trajectory_closed_loop': True,   # test closed-loop trajectories
        'traj_no': 30,          # number of closed-loop trajectories to test
        'end_time_traj': 0.6,   # [s] time span of the trajectories
    }

    # joining all the parameters in a single dictionary
    parameters = {**learner_params, 
                  **lyap_params, 
                  **control_params,
                  **falsifier_params, 
                  **dyn_sys_params, 
                  **postproc_params, 
                  **closed_loop_params}

    return parameters

