#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:56:06 2022

@author: Davide Grande

Compute the Lie derivative of V, given an ANN with 2 hidden layers.

"""

import torch


#
# Compute the Lie derivative -- Translator
#
def compute_v2(model, f, x, 
               parameters):

    Lie_der = torch.zeros([x.shape[0]])

    w1 = model.layer1.weight
    w2 = model.layer2.weight
    w3 = model.layer3.weight

    for jN in range(x.shape[0]):
        # Input layer
        z0 = x[jN].view(parameters['n_input'], 1)

        # First hidden layer
        if parameters['Lyap_bias1']:
            b1 = model.layer1.bias
            z1 = torch.mm(w1, z0) + b1.view(H1, 1)
        else:
            z1 = torch.mm(w1, z0)

        if (parameters['Lyap_act_fun1'] == 'tanh'):
            z1_act = torch.tanh(z1)
            der_z1_x = (1-torch.tanh(z1)**2)
            diag_z1 = torch.diag(der_z1_x.view(parameters['lyap_hid1']))
            out_z1 = torch.mm(diag_z1, w1)
        elif (parameters['Lyap_act_fun1'] == 'pow2'):
            z1_act = (z1)**2
            der_z1_x = 2*(z1)
            diag_z1 = torch.diag(der_z1_x.view(parameters['lyap_hid1']))
            out_z1 = torch.mm(diag_z1, w1)
        elif (parameters['Lyap_act_fun1'] == 'sfpl'):
            z1_act = 1/beta_sfpl * torch.log(1 + torch.exp(beta_sfpl*z1))
            der_z1_x = torch.exp(beta_sfpl*z1) / (1+torch.exp(beta_sfpl*z1))
            diag_z1 = torch.diag(der_z1_x.view(parameters['lyap_hid1']))
            out_z1 = torch.mm(diag_z1, w1)
        elif (parameters['Lyap_act_fun1'] == 'sigm'):
            z1_act = torch.sigmoid(z1)
            der_z1_x = 1 / (1+torch.exp(-z1))
            diag_z1 = torch.diag(der_z1_x.view(parameters['lyap_hid1']))
            out_z1 = torch.mm(diag_z1, w1)
        else:
            out_z1 = w1

        # Second hidden layer 
        if parameters['Lyap_bias2']:
            b2 = model.layer2.bias
            z2 = torch.mm(w2, z1) + b2.view(parameters['lyap_hid1'], 1)
        else:
            z2 = torch.mm(w2, z1)                        

        if (parameters['Lyap_act_fun2'] == 'tanh'):
            z2_act = torch.tanh(z2)
            der_z2_x = (1-torch.tanh(z2)**2)
            diag_z2 = torch.diag(der_z2_x.view(parameters['lyap_hid2']))
            out_z2 = torch.torch.mm(diag_z2, w2)  
        elif (parameters['Lyap_act_fun2'] == 'pow2'):
            z2_act = (z2)**2
            der_z2_x = 2*(z2)
            diag_z2 = torch.diag(der_z2_x.view(parameters['lyap_hid2']))
            out_z2 = torch.torch.mm(diag_z2, w2)  
        elif (parameters['Lyap_act_fun2'] == 'sfpl'):
            z2_act = 1/beta_sfpl*torch.log(1 + torch.exp(beta_sfpl*z2))
            der_z2_x =  torch.exp(beta_sfpl*z2) / (1+torch.exp(beta_sfpl*z2))
            diag_z2 = torch.diag(der_z2_x.view(parameters['lyap_hid2']))
            out_z2 = torch.torch.mm(diag_z2, w2)  
        elif (parameters['Lyap_act_fun2'] == 'sigm'):
            z2_act = torch.sigmoid(z2)
            der_z2_x = 1 / (1+torch.exp(-z2))
            diag_z2 = torch.diag(der_z2_x.view(parameters['lyap_hid2']))
            out_z2 = torch.mm(diag_z2, w2)   
        else: 
            out_z2 = w2

        # Output layer
        if parameters['Lyap_bias3']:
            b3 = model.layer3.bias
            z3 = torch.mm(w3, z2) + b3.view(1, 1)
        else:
            z3 = torch.mm(w3, z2)

        if (parameters['Lyap_act_fun3'] == 'tanh'):
            z3_act = torch.tanh(z3)
            der_z3_x = (1-torch.tanh(z3)**2)
            diag_z3 = torch.diag(der_z3_x.view(1))  # as V is a scalar function, this diagonalisation does not change the results
            out_z3 = torch.torch.mm(diag_z3, w3)         
        elif (parameters['Lyap_act_fun3'] == 'pow2'):
            z3_act = (z3)**2
            der_z3_x = 2*(z3)
            diag_z3 = torch.diag(der_z3_x.view(1))  # as V is a scalar function, this diagonalisation does not change the results
            out_z3 = torch.torch.mm(diag_z3, w3)         
        elif (parameters['Lyap_act_fun3'] == 'sfpl'):
            z3_act = 1/beta_sfpl*torch.log(1 + torch.exp(beta_sfpl*z3))
            der_z3_x = torch.exp(beta_sfpl*z3) / (1+torch.exp(beta_sfpl*z3))
            diag_z3 = torch.diag(der_z3_x.view(1))  # as V is a scalar function, this diagonalisation does not change the results
            out_z3 = torch.torch.mm(diag_z3, w3)               
        elif (parameters['Lyap_act_fun3'] == 'sigm'):
            z3_act = torch.sigmoid(z3)
            der_z3_x = 1 / (1+torch.exp(-z3))
            diag_z3 = torch.diag(der_z3_x.view(1))
            out_z3 = torch.mm(diag_z3, w3) 
        else:
            out_z3 = w3

        # Final products
        pi2 = torch.mm(out_z3, out_z2)  # out layer x 2nd hidden
        nabla_V = torch.mm(pi2, out_z1)  # pi2 x 1st hidden

        # Multiplication with dynamics
        Lie_der[jN] = torch.mm(nabla_V, f[jN].view(parameters['n_input'], 1))
    
    return Lie_der



#
# Compute the Lie derivative -- Translator
# (old version used by the inverted pendulum script)
def compute(model, f, x, 
            Lyap_bias1, Lyap_bias2, Lyap_bias3, 
            Lyap_act_fun1, Lyap_act_fun2, Lyap_act_fun3,
            D_in, H1, H2, beta_sfpl):

    Lie_der = torch.zeros([x.shape[0]])

    w1 = model.layer1.weight
    w2 = model.layer2.weight
    w3 = model.layer3.weight

    for jN in range(x.shape[0]):
        # Input layer
        z0 = x[jN].view(D_in, 1)

        # First hidden layer
        if Lyap_bias1:
            b1 = model.layer1.bias
            z1 = torch.mm(w1, z0) + b1.view(H1, 1)
        else:
            z1 = torch.mm(w1, z0)

        if (Lyap_act_fun1 == 'tanh'):
            z1_act = torch.tanh(z1)
            der_z1_x = (1-torch.tanh(z1)**2)
            diag_z1 = torch.diag(der_z1_x.view(H1))
            out_z1 = torch.mm(diag_z1, w1)
        elif (Lyap_act_fun1 == 'pow2'):
            z1_act = (z1)**2
            der_z1_x = 2*(z1)
            diag_z1 = torch.diag(der_z1_x.view(H1))
            out_z1 = torch.mm(diag_z1, w1)
        elif (Lyap_act_fun1 == 'sfpl'):
            z1_act = 1/beta_sfpl * torch.log(1 + torch.exp(beta_sfpl*z1))
            der_z1_x = torch.exp(beta_sfpl*z1) / (1+torch.exp(beta_sfpl*z1))
            diag_z1 = torch.diag(der_z1_x.view(H1))
            out_z1 = torch.mm(diag_z1, w1)
        elif (Lyap_act_fun1 == 'sigm'):
            z1_act = torch.sigmoid(z1)
            der_z1_x = 1 / (1+torch.exp(-z1))
            diag_z1 = torch.diag(der_z1_x.view(H1))
            out_z1 = torch.mm(diag_z1, w1)
        else:
            out_z1 = w1

        # Second hidden layer 
        if Lyap_bias2:
            b2 = model.layer2.bias
            z2 = torch.mm(w2, z1) + b2.view(H2, 1)
        else:
            z2 = torch.mm(w2, z1)                        

        if (Lyap_act_fun2 == 'tanh'):
            z2_act = torch.tanh(z2)
            der_z2_x = (1-torch.tanh(z2)**2)
            diag_z2 = torch.diag(der_z2_x.view(H2))
            out_z2 = torch.torch.mm(diag_z2, w2)  
        elif (Lyap_act_fun2 == 'pow2'):
            z2_act = (z2)**2
            der_z2_x = 2*(z2)
            diag_z2 = torch.diag(der_z2_x.view(H2))
            out_z2 = torch.torch.mm(diag_z2, w2)  
        elif (Lyap_act_fun2 == 'sfpl'):
            z2_act = 1/beta_sfpl*torch.log(1 + torch.exp(beta_sfpl*z2))
            der_z2_x =  torch.exp(beta_sfpl*z2) / (1+torch.exp(beta_sfpl*z2))
            diag_z2 = torch.diag(der_z2_x.view(H2))
            out_z2 = torch.torch.mm(diag_z2, w2)  
        elif (Lyap_act_fun2 == 'sigm'):
            z2_act = torch.sigmoid(z2)
            der_z2_x = 1 / (1+torch.exp(-z2))
            diag_z2 = torch.diag(der_z2_x.view(H2))
            out_z2 = torch.mm(diag_z2, w2)   
        else: 
            out_z2 = w2

        # Output layer
        if Lyap_bias3:
            b3 = model.layer3.bias
            z3 = torch.mm(w3, z2) + b3.view(1, 1)
        else:
            z3 = torch.mm(w3, z2)

        if (Lyap_act_fun3 == 'tanh'):
            z3_act = torch.tanh(z3)
            der_z3_x = (1-torch.tanh(z3)**2)
            diag_z3 = torch.diag(der_z3_x.view(1))  # as V is a scalar function, this diagonalisation does not change the results
            out_z3 = torch.torch.mm(diag_z3, w3)         
        elif (Lyap_act_fun3 == 'pow2'):
            z3_act = (z3)**2
            der_z3_x = 2*(z3)
            diag_z3 = torch.diag(der_z3_x.view(1))  # as V is a scalar function, this diagonalisation does not change the results
            out_z3 = torch.torch.mm(diag_z3, w3)         
        elif (Lyap_act_fun3 == 'sfpl'):
            z3_act = 1/beta_sfpl*torch.log(1 + torch.exp(beta_sfpl*z3))
            der_z3_x = torch.exp(beta_sfpl*z3) / (1+torch.exp(beta_sfpl*z3))
            diag_z3 = torch.diag(der_z3_x.view(1))  # as V is a scalar function, this diagonalisation does not change the results
            out_z3 = torch.torch.mm(diag_z3, w3)               
        elif (Lyap_act_fun3 == 'sigm'):
            z3_act = torch.sigmoid(z3)
            der_z3_x = 1 / (1+torch.exp(-z3))
            diag_z3 = torch.diag(der_z3_x.view(1))
            out_z3 = torch.mm(diag_z3, w3) 
        else:
            out_z3 = w3

        # Final products
        pi2 = torch.mm(out_z3, out_z2)  # out layer x 2nd hidden
        nabla_V = torch.mm(pi2, out_z1)  # pi2 x 1st hidden

        # Multiplication with dynamics
        Lie_der[jN] = torch.mm(nabla_V, f[jN].view(D_in, 1))
    
    return Lie_der



