#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:49:27 2022

@authors: Andrea Peruffo
          Davide Grande
          

This script contains the CEGIS loop, entailing the Learner and the callbacks to 
the Falsifier function.

"""

import torch
import numpy as np
import timeit
import copy
import os
import time
from datetime import datetime
import torch.nn.functional as F
from utilities.falsifier import augm_falsifier
from utilities.utils_processing import postprocessing, init_history_arrays, save_cost_funct, save_lr_values
from utilities.nn import optimizer_setup
from utilities.sanity_checks import initial_checks, check_ANN_model
from utilities.standalone_controller import extract_controller_from_model
from utilities.lyapunov_control import analyze_trained_model
import logging

##############################################################################

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = timeit.default_timer()
        result = func(*args, **kwargs)
        t2 = timeit.default_timer()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


# execute postproc --
init_historical_data = True

    
def cegis(parameters, seed_,
          Net,
          vars_, f_torch, f_symb,
          config,
          final_dir_campaign, i_loop, saving_log,
          cl, dynamic_sys):
    
    '''
    Variables definition
    '''
    torch.manual_seed(seed_)
    init_date = datetime.now()


    # generate dataset
    x = torch.Tensor(parameters['N'], parameters['n_input']).uniform_(-parameters['gamma_overbar'], parameters['gamma_overbar'])
    x_dataset_init = x.detach().clone()  # initial dataset

    # init timing info
    optim_time = 0.
    clamp_time = 0.
    slide_time = 0.
    save_time = 0.
    forward_time = 0.
    actual_forward_time = 0.
    loss_time = 0.
    derivative_time = 0.
    step_time = 0.
    backward_time = 0.
    grad_time = 0.
    t_falsifier = 0.
    out_iters = 0
    found_lyap_f = False
    to_fals = False    # time out falsifier
    to_learner = False # time out learner

    # timer
    start = timeit.default_timer()

    if init_historical_data:
        history_arrays = init_history_arrays(parameters, parameters['max_iters']*parameters['max_loop_number'])

    # Sanity check on choice of the configuration parameters
    warn = initial_checks(parameters)

    '''
    Main
    '''
    while out_iters < parameters['max_loop_number'] and not found_lyap_f and not to_fals:

        # generated random (linear) control vector if not initialised
        if not parameters['control_initialised']:
            parameters['init_control'] = torch.rand([parameters['size_ctrl_layers'][-1], parameters['n_input']]) - 0.5

        # Instantiating ANN architecture
        model = Net(parameters, seed_)

        # Check correct instantiation of the ANN
        warn_ann = check_ANN_model(model, vars_)
        if (warn + warn_ann != 0):
            print("\n\nWARNING: the training will be started, but something is off with your parameters. Please check the WARNING message above for further information.\n\n")
            time.sleep(3)

        i_epoch = 0  # epochs counter

        # setting up optimiser
        optimizer = optimizer_setup(parameters, model)

        # setting up LR scheduluer
        if parameters['use_scheduler']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=parameters['sched_T'], eta_min=0)

        # training cycle
        while i_epoch < parameters['max_iters'] and not found_lyap_f and not to_fals:

            # sliding window
            start_ = timeit.default_timer()
            if parameters['sliding_window'] and x.shape[0] > parameters['N_max']:

                if parameters['verbose_info']:
                    print("'sliding wind': Sliding x.")
                    print(f"'sliding wind': x_old size = {x.shape[0]}")

                # removing initial points from the current dataset
                x_noinit = x[parameters['N']: x.shape[0], :]

                # sliding oldest CEs points out
                x_sw = x_noinit[(x.shape[0] - parameters['N_max']):, :]

                # restoring initial points
                x = torch.cat((x_dataset_init, x_sw), 0)

                if parameters['verbose_info']:
                    print(f"'sliding wind': x_new size = {x.shape[0]}\n")
            stop_ = timeit.default_timer()
            slide_time = slide_time + stop_ - start_

            # Forward ANN pass
            start_ = timeit.default_timer()
            sta_ = timeit.default_timer()

            V_candidate, Lie_V, Circle_Tuning, u = model.forward(x, f_torch, parameters)
            # evaluating model at the equilibrium point
            x_0 = torch.zeros([1, parameters['n_input']])  #x_star.view(1, -1)
            V0, Lie_V0, Circle_Tuning_0, u0 = model.forward(x_0, f_torch, parameters)

            sto_ = timeit.default_timer()
            actual_forward_time = actual_forward_time + sto_ - sta_

            # Loss function related to the first 3 Lyapunov conditions
            sta_ = timeit.default_timer()
            Lyap_risk_SLR = parameters['alpha_1'] * torch.sum(F.relu(-V_candidate)) + \
                            parameters['alpha_2'] * torch.sum(F.relu(Lie_V)) + \
                            parameters['alpha_3'] * torch.pow(V0, 2)

            # Extended Lyap loss
            Lyap_risk_ELR = parameters['alpha_5'] * (
                    Lyap_risk_SLR + parameters['alpha_4'] * ((Circle_Tuning - parameters['alpha_roa'] * V_candidate).pow(2)).mean())

            if i_epoch % parameters['print_interval'] == 0:
                message_update = f"(#{out_iters + 1}/{parameters['max_loop_number']}, #{i_epoch}/{parameters['max_iters']})\
                 L. Risk = {Lyap_risk_ELR.item()}"
                print(message_update)

            if init_historical_data:
                # Saving computed cost function values
                copied_m = copy.deepcopy(model)

                history_arrays = save_cost_funct(history_arrays, 
                                                 parameters,
                                                 out_iters, parameters['max_iters'], i_epoch,
                                                 Lyap_risk_SLR, Lyap_risk_ELR,
                                                 V_candidate, Lie_V, V0, Circle_Tuning, copied_m)


            sto_ = timeit.default_timer()
            loss_time = loss_time + sto_ - sta_

            stop_ = timeit.default_timer()
            forward_time = forward_time + stop_ - start_

            # SGD step
            start_opt = timeit.default_timer()

            # grad evaluation
            start_ = timeit.default_timer()
            stop_ = timeit.default_timer()
            grad_time = grad_time + stop_ - start_

            # backward step
            start_ = timeit.default_timer()
            backward_time = backward_time + start_ - stop_

            optimizer.zero_grad()
            Lyap_risk_ELR.backward()
            optimizer.step()
            if parameters['use_scheduler']:
                scheduler.step()

            stop_ = timeit.default_timer()
            step_time = step_time + stop_ - start_
            stop_optim = timeit.default_timer()
            optim_time = optim_time + stop_optim - start_opt

            # clipping weight
            start_ = timeit.default_timer()
            if parameters['clipping_V']:
                for j in range(len(model.layers)):
                    model.layers[j].weight.data = model.layers[j].weight.data.clamp_min(np.finfo(float).eps)

            stop_ = timeit.default_timer()
            clamp_time = clamp_time + stop_ - start_

            start_ = timeit.default_timer()
            if init_historical_data:
                # saving learning rate
                history_arrays = save_lr_values(history_arrays, optimizer, 
                                                out_iters, parameters['max_iters'], i_epoch)

            stop_ = timeit.default_timer()
            save_time = save_time + stop_ - start_

            # Augmented falsifier
            if Lyap_risk_SLR.item() == 0.:
                message_update = f"(#{out_iters + 1}/{parameters['max_loop_number']}, #{i_epoch}/{parameters['max_iters']})\
                                 SLR = {Lyap_risk_SLR.item()}"
                print(message_update)

                print('\n=========== Augmented Falsifier ==========')
                start_ = timeit.default_timer()
                x, u_learn, V_learn, lie_derivative_of_V, f_out_sym, found_lyap_f, to_fals =\
                    augm_falsifier(parameters, vars_, model, f_symb,
                                   parameters['gamma_underbar'], parameters['gamma_overbar'], config, parameters['epsilon'], x)
                stop_ = timeit.default_timer()
                t_falsifier = t_falsifier + stop_ - start_

            i_epoch += 1

        out_iters += 1

        if out_iters == parameters['max_loop_number'] and not found_lyap_f:
            print('============================================================')
            print('Training unsuccessful: Control Lyapunov Function NOT FOUND within the current training attempt.')
        print('============================================================')

    if not found_lyap_f and not to_fals:
        # if the test was completed and the Falsifier was not the cause of TO
        to_learner = True

    # Saving times for statistics generation
    stop = timeit.default_timer()
    seconds_elapsed = stop - start
    minutes_elapsed = seconds_elapsed / 60.
    hours_elapsed = minutes_elapsed / 60.
    falsifier_elapsed = t_falsifier / 60.
    end_date = datetime.now()

    print('\n')
    print("Total time [s]: ", seconds_elapsed)
    print("Total time [']: ", minutes_elapsed)
    print("Total time [h]: ", hours_elapsed)
    print("Falsifier time [']: ", falsifier_elapsed)
    print(f"Falsifier time [%]: {falsifier_elapsed / minutes_elapsed * 100}")

    print("Save time [s]: ", save_time)
    print("Optim time [s]: ", optim_time)
    print('of which')
    print("    step time [s]: ", step_time)
    print("    backward time [s]: ", backward_time)
    print("    grad time [s]: ", grad_time)

    print("Clamp time [s]: ", clamp_time)
    print("Forward time [s]: ", forward_time)
    print('of which')
    print("    Actual Forward time [s]: ", actual_forward_time)
    print("    derivative time [s]: ", derivative_time)
    print("    loss time [s]: ", loss_time)

    print("Sliding window time [s]: ", slide_time)
    print("\n")


    '''
    Postprocessing
    ''' 
    if not found_lyap_f and not to_fals:
        x, u_learn, V_learn, lie_derivative_of_V, f_out_sym, found_lyap_f, to_fals =\
                augm_falsifier(parameters, vars_, model, f_symb,
                               parameters['gamma_underbar'], parameters['gamma_overbar'], 
                               config, parameters['epsilon'], x)

    if parameters['execute_postprocessing']:
        final_dir_run = postprocessing(final_dir_campaign, i_loop, parameters, x,
                                       model, cl, dynamic_sys,
                                       history_arrays,
                                       parameters['gamma_overbar'], parameters['gamma_underbar'],
                                       u_learn, V_learn, lie_derivative_of_V,
                                       x_dataset_init)

    else:
        # 0) Generating result folder
        try:
            final_dir_run = final_dir_campaign + "/" + str(i_loop)
            os.mkdir(final_dir_run)
        except OSError:
            logging.error("Creation of the result directory %s failed" % final_dir_run)
        else:
            print("Result directory successfully created as: \n %s \n" % final_dir_run)


    end_date = datetime.now()

    # Automatic log report
    saving_log.gen_log(found_lyap_f, 
                       parameters, x, config, model, seed_,
                       to_fals, to_learner, seconds_elapsed, minutes_elapsed, hours_elapsed,
                       out_iters, i_epoch, start, init_date, end_date, falsifier_elapsed,
                       final_dir_run)
                        

    exit_info = {'stop': stop,
                 'start': start,
                 'tot_iters': i_epoch,
                 'found_lyap_f': found_lyap_f,
                 'to_learner': to_learner,
                 'to_fals': to_fals,
                 'history_arrays': history_arrays,
                 'model': model, 
                 'f_out_sym': f_out_sym,
                 'u_learn': u_learn,
                 'V_learn': V_learn,
                 'lie_derivative_of_V': lie_derivative_of_V,
                 'model': model}
                    
    if found_lyap_f:
              state_dim = parameters['n_input'] # state dimension
              control_dim = parameters['size_ctrl_layers'][-1] # control output dimension

              # create and add standalone controller to exit_info
              standalone_controller = extract_controller_from_model(model, state_dim, control_dim)
              exit_info['standalone_controller'] = standalone_controller
              exit_info['controller_weights'] = []
              exit_info['controller_biases'] = []
              for i, layer in enumerate(model.ctrl_layers):
                      exit_info['controller_weights'].append(standalone_controller.ctrl_layers[i].weight.data)
                      exit_info['controller_biases'].append(standalone_controller.ctrl_layers[i].bias.data)
              controller_state = {
                        'weights': [layer.weight.data for layer in standalone_controller.ctrl_layers],
                        'biases': [layer.bias.data for layer in standalone_controller.ctrl_layers]
              }
              torch.save(controller_state, os.path.join(final_dir_run, "controller_weights.pt"))
    
    print("exit info about controller: ", exit_info['standalone_controller'])

    print(analyze_trained_model(model))

    return parameters, exit_info

