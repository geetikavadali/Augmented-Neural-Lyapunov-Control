#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:25:11 2022

@author: Davide Grande

Generating the symbolic dynamics of an inverted pendulum needed by dReal.


"""

import numpy as np
import dreal

def symbolic_sys(use_lin_ctr, 
                 contr_act_fun1, contr_act_fun2, contr_act_fun3, 
                 contr_bias1, contr_bias2, contr_bias3, lin_contr_bias,
                 Lyap_bias1, Lyap_bias2, Lyap_bias3,
                 Lyap_act_fun1, Lyap_act_fun2, Lyap_act_fun3, 
                 beta_sfpl,
                 x1, x2, vars_,
                 model,
                 dyn_sys_params):
    
    # Model dynamics parameters
    G = dyn_sys_params.G
    L = dyn_sys_params.L
    m = dyn_sys_params.m
    b = dyn_sys_params.b


    # Lyapunov ANN parameters (weight and bias)
    w1 = model.layer1.weight.data.numpy()
    w2 = model.layer2.weight.data.numpy()
    w3 = model.layer3.weight.data.numpy()
    if Lyap_bias1:
        b1 = model.layer1.bias.data.numpy()
    if Lyap_bias2:
        b2 = model.layer2.bias.data.numpy()
    if Lyap_bias3:
        b3 = model.layer3.bias.data.numpy()

    # Parameters control ANN (weight and bias)
    if use_lin_ctr:
        q = model.control.weight.data.numpy()
        if lin_contr_bias:
            bi = model.control.bias.data.numpy()
    else:
        wc1 = model.control1.weight.data.numpy()
        wc2 = model.control2.weight.data.numpy()
        wc3 = model.control3.weight.data.numpy()
        if contr_bias1:
            bc1 = model.control1.bias.data.numpy()
        if contr_bias2:
            bc2 = model.control2.bias.data.numpy()
        if contr_bias3:
            bc3 = model.control3.bias.data.numpy()


    #
    # Candidate control function
    #
    if use_lin_ctr:

        if lin_contr_bias:
            u_NN = np.dot(vars_, q.T) + bi
        else: 
            u_NN = np.dot(vars_, q.T)        

        u_NN0 = u_NN.item(0)


    else:
        # First control layer
        if contr_bias1:
            u1 = np.dot(vars_, wc1.T) + bc1  # w/ bias layer 1
        else:
            u1 = np.dot(vars_, wc1.T)  # w/o bias layer 1
        ua1 = []
        for j in range(0,len(u1)):  # activation function
            if contr_act_fun1 == 'tanh':
                ua1.append(dreal.tanh(u1[j]))  # use tanh act. function
            elif contr_act_fun1 == 'sfpl':
                ua1.append(1/beta_sfpl*dreal.log(1 + dreal.exp(beta_sfpl*u1[j])))
            elif contr_act_fun1 == 'linear':
                ua1.append(u1[j])

        # Second control layer
        if contr_bias2:
            u2 = np.dot(ua1, wc2.T) + bc2  # w/ bias layer 2
        else:
            u2 = np.dot(ua1, wc2.T)  # w/o bias layer 2
        ua2 = []
        for j in range(0,len(u2)):  # activation function
            if contr_act_fun2 == 'tanh':
                ua2.append(dreal.tanh(u2[j]))  # use tanh act. function    
            elif contr_act_fun2 == 'sfpl':
                ua2.append(1/beta_sfpl*dreal.log(1 + dreal.exp(beta_sfpl*u2[j])))
            elif contr_act_fun2 == 'linear':
                ua2.append(u2[j])
            #ua2.append(dreal.ReLU(u2[j]))

        if contr_bias3:
            u3 = np.dot(ua2, wc3.T) + bc3
        else:    
            u3 = np.dot(ua2, wc3.T)
        ua3 = []
        for j in range(0,len(u3)):  # activation function
            if contr_act_fun3 == 'tanh':
                ua3.append(dreal.tanh(u3[j]))  # use tanh act. function    
            elif contr_act_fun3 == 'sfpl':
                ua3.append(1/beta_sfpl*dreal.log(1 + dreal.exp(beta_sfpl*u3[j])))
            elif contr_act_fun3 == 'linear':
                ua3.append(u3[j])

        u_NN0 = u3.item(0)


    # Computing symbolic dynamics
    f_out_sym = [x2, 
                 (m*G*L*dreal.sin(x1)- b*x2 + u_NN0) / (m*L**2) ]
    
    f_closed_1 = (x2) 
    f_closed_2 = (m*G*L*dreal.sin(x1)- b*x2 + u_NN0) / (m*L**2)
    
    f_open_1 = (x2)
    f_open_2 = (m*G*L*dreal.sin(x1)- b*x2) / (m*L**2)

    #
    # Candidate V
    #
    # First layer
    if Lyap_bias1:
        v1 = np.dot(vars_, w1.T) + b1  # w/ bias layer 1
    else:
        v1 = np.dot(vars_, w1.T)  # w/o bias layer 1
    va1 = []
    for j in range(0,len(v1)):  # activation function
        if (Lyap_act_fun1 == 'tanh'):
            # use tanh act. function
            va1.append(dreal.tanh(v1[j]))  
        elif (Lyap_act_fun1 == 'pow2'): 
            # use quadratic polynomial act. f.
            va1.append(v1[j]**2)  
        elif (Lyap_act_fun1 == 'sfpl'): 
            # softplus activation function
            va1.append( dreal.log(1 + dreal.exp(v1[j]) ) )
        else:
            # linear activation function
            va1.append(v1[j])  

    # Second layer
    if Lyap_bias2:
        v2 = np.dot(va1, w2.T) + b2  # w/ bias layer 2
    else:
        v2 = np.dot(va1, w2.T)  # w/o bias layer 2
    va2 = []
    for j in range(0,len(v2)):  # activation function
        if (Lyap_act_fun2  == 'tanh'):
             # use tanh act. function
            va2.append(dreal.tanh(v2[j]))
        elif (Lyap_act_fun2 == 'pow2'): 
            # use quadratic polynomial act. f.
            va2.append(v2[j]**2)
        elif (Lyap_act_fun2 == 'sfpl'): 
            # softplus activation function
            va2.append( dreal.log(1 + dreal.exp(v2[j]) ) )
        else:
            # linear activation function
            va2.append(v2[j])

    # Output layer
    if Lyap_bias3:
        v3 = np.dot(va2, w3.T) + b3  # w/ bias layer 3
    else:
        v3 = np.dot(va2, w3.T)  # w/o bias layer 3

    if (Lyap_act_fun3 == 'tanh'):
        V_learn = dreal.tanh(v3.item(0))
    elif (Lyap_act_fun3 == 'pow2'): 
        V_learn = (v3.item(0))**2
    elif (Lyap_act_fun3 == 'sfpl'): 
        # softplus activation function
        V_learn = dreal.log(1 + dreal.exp(v3.item(0)) ) 
    else:
        # linear activation function
        V_learn = (v3.item(0))    


    return u_NN0, V_learn, f_out_sym, f_closed_1, f_closed_2, f_open_1, f_open_2
