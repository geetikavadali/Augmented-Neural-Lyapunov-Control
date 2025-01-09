#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:44:12 2022
@authors: Andrea Peruffo
          Davide Grande
          
"""
import dreal
import timeit
import logging
import utilities.Functions as Functions
import time
from utilities.translator import translator


def augm_falsifier(parameters, vars_, model, f_symb,
                   gamma_underbar,
                   gamma_overbar,
                   config,
                   epsilon,
                   x
                   ):

    found_lyap_f = False
    to_fals = False
    # Computing the system symbolic dynamics
    u_learn, V_learn, f_out_sym = translator(parameters, vars_, model, f_symb)

    print("\nDiscrete Falsifier computing CEs ...")
    lie_derivative_of_V = Functions.LieDerivative(vars_, f_out_sym,
                                                  V_learn,
                                                  gamma_underbar,
                                                  gamma_overbar,
                                                  config,
                                                  epsilon)

    if parameters['n_input'] == 2: 
        x, disc_viol_found = \
            Functions.AddLieViolationsOrder2_v4(x,
                                                gamma_overbar,
                                                parameters['grid_points'],
                                                parameters['zeta_D'],
                                                parameters['verbose_info'],
                                                V_learn,
                                                lie_derivative_of_V)

    elif parameters['n_input'] == 3: 
        x, disc_viol_found = \
            Functions.AddLieViolationsOrder3_v4(x,
                                                gamma_overbar,
                                                parameters['grid_points'],
                                                parameters['zeta_D'],
                                                parameters['verbose_info'],
                                                V_learn,
                                                lie_derivative_of_V)
            
    elif parameters['n_input'] == 4: 
        x, disc_viol_found = \
            Functions.AddLieViolationsOrder4_v4(x,
                                                gamma_overbar,
                                                parameters['grid_points'],
                                                parameters['zeta_D'],
                                                parameters['verbose_info'],
                                                V_learn,
                                                lie_derivative_of_V)

    else:
        dimension_sys = parameters['n_input']
        logging.error(f'Not implemented Falsifier for system of order {dimension_sys}!')
        raise ValueError('Functionality to be implemented in: utilities/falsifier.py')


    # If no CE is found, invoke the SMT Falsifier
    if disc_viol_found == 0:

        print("\nSMT Falsifier computing CE ...")
        try:
            start_ = timeit.default_timer()

            CE_SMT, \
            lie_derivative_of_V = Functions.CheckLyapunov(vars_,
                                                          f_out_sym,
                                                          V_learn,
                                                          gamma_underbar,
                                                          gamma_overbar,
                                                          config,
                                                          epsilon)


        except TimeoutError:
            logging.error("SMT Falsifier Timed Out")
            to_fals = True
            stop_ = timeit.default_timer()
            fals_to_check = stop_ - start_
            #time.sleep(5)
            
        if not to_fals:
            if CE_SMT:
                # if a counterexample is found
                print("SMT found a CE: ")
                print(CE_SMT)
    
                # Adding midpoint of the CE_SMT to the history
                # todo: history of CEs
    
                x = Functions.AddCounterexamples(x, CE_SMT, parameters['zeta_SMT'])
                if parameters['verbose_info']:
                    logging.debug(f"'SMT Falsifier': Added {parameters['zeta_SMT']} points in the vicinity of the CE.\n")
    
            else:
                # no CE_SMT is returned hence V_learn is a valid Lyapunov
                # function
                print("\nTraining SUCCESSFUL (no CE found)!")
                found_lyap_f = True
                print("\nControl Lyapunov Function synthesised as:")
                print(V_learn.Expand())

    else:
        print(f"Skipping SMT callback.\n")

    stop_ = timeit.default_timer()

    print('================================')

    return x, u_learn, V_learn, lie_derivative_of_V, f_out_sym, found_lyap_f, to_fals
