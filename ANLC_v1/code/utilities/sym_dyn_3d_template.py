#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:42:24 2022

Computing the symbolic dynamics of a generic 3-dimensional system.

@author: Davide Grande

"""

import numpy as np
import dreal

def symbolic_sys(parameters,
                 vars_,
                 model, x_star):
    
    # dReal variables
    x1 = vars_[0]
    x2 = vars_[1]
    x3 = vars_[2]
    
    # Model dynamics parameters
    sigma = parameters['sigma']
    b = parameters['b']
    r = parameters['r']

    # Lyapunov ANN parameters (weight and bias)
    w1 = model.layer1.weight.data.numpy()
    w2 = model.layer2.weight.data.numpy()
    w3 = model.layer3.weight.data.numpy()
    if parameters['Lyap_bias1']:
        b1 = model.layer1.bias.data.numpy()
    if parameters['Lyap_bias2']:
        b2 = model.layer2.bias.data.numpy()
    if parameters['Lyap_bias3']:
        b3 = model.layer3.bias.data.numpy()

    # Parameters control ANN (weight and bias)
    if parameters['use_lin_ctr']:
        q = model.control.weight.data.numpy()
        if parameters['lin_contr_bias']:
            bi = model.control.bias.data.numpy()
    else:
        wc1 = model.control1.weight.data.numpy()
        wc2 = model.control2.weight.data.numpy()
        wc3 = model.control3.weight.data.numpy()
        if parameters['contr_bias1']:
            bc1 = model.control1.bias.data.numpy()
        if parameters['contr_bias2']:
            bc2 = model.control2.bias.data.numpy()
        if parameters['contr_bias3']:
            bc3 = model.control3.bias.data.numpy()


    #
    # Candidate control function
    #
    if parameters['use_lin_ctr']:
        
        if parameters['lin_contr_bias']:
            u_NN = np.dot(vars_, q.T) + bi
        else: 
            u_NN = np.dot(vars_, q.T)  
        
        u_NN0 = u_NN.item(0)
        u_NN1 = u_NN.item(1)
        u_NN2 = u_NN.item(2)

    else:
        # First control layer
        if parameters['contr_bias1']:
            u1 = np.dot(vars_, wc1.T) + bc1  # w/ bias layer 1
        else:
            u1 = np.dot(vars_, wc1.T)  # w/o bias layer 1
        ua1 = []
        for j in range(0,len(u1)):  # activation function
            if parameters['contr_act_fun1'] == 'tanh':
                ua1.append(dreal.tanh(u1[j]))  # use tanh act. function
            elif parameters['contr_act_fun1'] == 'sfpl':
                ua1.append(1/parameters['beta_sfpl']*dreal.log(1 + dreal.exp(parameters['beta_sfpl']*u1[j])))
            elif parameters['contr_act_fun1'] == 'linear':
                ua1.append(u1[j])

        # Second control layer
        if parameters['contr_bias2']:
            u2 = np.dot(ua1, wc2.T) + bc2  # w/ bias layer 2
        else:
            u2 = np.dot(ua1, wc2.T)  # w/o bias layer 2
        ua2 = []
        for j in range(0,len(u2)):  # activation function
            if parameters['contr_act_fun2'] == 'tanh':
                ua2.append(dreal.tanh(u2[j]))  # use tanh act. function    
            elif parameters['contr_act_fun2'] == 'sfpl':
                ua2.append(1/parameters['beta_sfpl']*dreal.log(1 + dreal.exp(parameters['beta_sfpl']*u2[j])))
            elif parameters['contr_act_fun2'] == 'linear':
                ua2.append(u2[j])

        if parameters['contr_bias3']:
            u3 = np.dot(ua2, wc3.T) + bc3
        else:    
            u3 = np.dot(ua2, wc3.T)
        ua3 = []
        for j in range(0,len(u3)):  # activation function
            if parameters['contr_act_fun3'] == 'tanh':
                ua3.append(dreal.tanh(u3[j]))  # use tanh act. function    
            elif parameters['contr_act_fun3'] == 'sfpl':
                ua3.append(1/parameters['beta_sfpl']*dreal.log(1 + dreal.exp(parameters['beta_sfpl']*u3[j])))
            elif parameters['contr_act_fun3'] == 'linear':
                ua3.append(u3[j])
            
        u_NN0 = u3[0]
        u_NN1 = u3[1]
        u_NN2 = u3[2]    
            
    # coordinate transformation
    x1_shift = (x1+x_star.numpy()[0])
    x2_shift = (x2+x_star.numpy()[1])
    x3_shift = (x3+x_star.numpy()[2])        

    # Computing symbolic dynamics
    f_out_sym = [-sigma*(x1_shift-x2_shift) + u_NN0, 
                 r*x1_shift - x2_shift - x1_shift*x3_shift + u_NN1, 
                 x1_shift*x2_shift - b*x3_shift + u_NN2]

    #
    # Candidate V
    #

    # First layer
    if parameters['Lyap_bias1']:
        v1 = np.dot(vars_, w1.T) + b1  # w/ bias layer 1
    else:
        v1 = np.dot(vars_, w1.T)  # w/o bias layer 1
    va1 = []
    for j in range(0,len(v1)):  # activation function
        if (parameters['Lyap_act_fun1'] == 'tanh'):
            # use tanh act. function
            va1.append(dreal.tanh(v1[j]))  
        elif (parameters['Lyap_act_fun1'] == 'pow2'): 
            # use quadratic polynomial act. f.
            va1.append(v1[j]**2)  
        elif (parameters['Lyap_act_fun1 ']== 'sfpl'): 
            # softplus activation function
            va1.append( 1/parameters['beta_sfpl']*dreal.log(1 + dreal.exp(parameters['beta_sfpl']*v1[j]) ) )
        else:
            # linear activation function
            va1.append(v1[j])  

    # Second layer
    if parameters['Lyap_bias2']:
        v2 = np.dot(va1, w2.T) + b2  # w/ bias layer 2
    else:
        v2 = np.dot(va1, w2.T)  # w/o bias layer 2
    va2 = []
    for j in range(0,len(v2)):  # activation function
        if (parameters['Lyap_act_fun2']  == 'tanh'):
             # use tanh act. function
            va2.append(dreal.tanh(v2[j]))
        elif (parameters['Lyap_act_fun2'] == 'pow2'): 
            # use quadratic polynomial act. f.
            va2.append(v2[j]**2)
        elif (parameters['Lyap_act_fun2'] == 'sfpl'): 
            # softplus activation function
            va2.append( 1/parameters['beta_sfpl']*dreal.log(1 + dreal.exp(parameters['beta_sfpl']*v2[j]) ) )
        else:
            # linear activation function
            va2.append(v2[j])

    # Output layer
    if parameters['Lyap_bias3']:
        v3 = np.dot(va2, w3.T) + b3  # w/ bias layer 3
    else:
        v3 = np.dot(va2, w3.T)  # w/o bias layer 3

    if (parameters['Lyap_act_fun3']== 'tanh'):
        V_learn = dreal.tanh(v3.item(0))
    elif (parameters['Lyap_act_fun3'] == 'pow2'): 
        V_learn = (v3.item(0))**2
    elif (parameters['Lyap_act_fun3'] == 'sfpl'): 
        # softplus activation function
        V_learn = 1/parameters['beta_sfpl']*dreal.log(1 + dreal.exp(parameters['beta_sfpl']*v3.item(0)) ) 
    else:
        # linear activation function
        V_learn = (v3.item(0))    

    
    return u_NN0, u_NN1, u_NN2, V_learn, f_out_sym