#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 01:33:00 2023

@author: Davide Grande


"""

from dreal import *
import torch 
import numpy as np
import random
import utilities.from_dreal_to_np as from_dreal_to_np
import copy
from wrapt_timeout_decorator import *


@timeout(180, use_signals=True)
def CheckLyapunov(vars_, f_out_sym, V, ball_lb, ball_ub, config, epsilon):    
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # Check the Lyapunov conditions within a domain around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub). 
    # If it return unsat, then there is no state violating the conditions. 

    ball= Expression(0)
    lie_derivative_of_V = Expression(0)

    for i in range(len(vars_)):
        ball += vars_[i]*vars_[i]
        lie_derivative_of_V += f_out_sym[i]*V.Differentiate(vars_[i])  
    ball_in_bound = logical_and(ball_lb*ball_lb <= ball, ball <= ball_ub*ball_ub)

    # Constraint: x ∈ Ball → (V(c, x) > 0 ∧ Lie derivative of V <= 0)     
    condition = logical_and(logical_imply(ball_in_bound, V >= 0),
                           logical_imply(ball_in_bound, lie_derivative_of_V <= epsilon))  # CAVEAT: origina version: lie_derivative_of_V <= epsilon 
    return CheckSatisfiability(logical_not(condition), config), lie_derivative_of_V



def LieDerivative(vars_, f_out_sym, V, ball_lb, ball_ub, config, epsilon):    
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # this function returns the Lie derivative

    lie_derivative_of_V = Expression(0)

    for i in range(len(vars_)):
        lie_derivative_of_V += f_out_sym[i]*V.Differentiate(vars_[i])  
    
    return lie_derivative_of_V


def LieDerivative_v2(vars_, f_out_sym, V):    
    # Given a dynamical system dx/dt = f(x,u) and candidate Lyapunov function V
    # this function returns the Lie derivative.
    # Differently from "LieDerivative", "LieDerivative_v2" takes less arguments.

    lie_derivative_of_V = Expression(0)

    for i in range(len(vars_)):
        lie_derivative_of_V += f_out_sym[i]*V.Differentiate(vars_[i])  
    
    return lie_derivative_of_V


def AddCounterexamples(x, CE, zeta): 
    # Adding CE back to sample set
    c = []
    nearby= []
    for i in range(CE.size()):
        c.append(CE[i].mid())
        lb = CE[i].lb()
        ub = CE[i].ub()
        nearby_ = np.linspace(lb, ub, zeta)  #np.linspace(lb,ub,N)
        nearby.append(nearby_)
    for i in range(zeta):
        n_pt = []
        for j in range(x.shape[1]):
            n_pt.append(nearby[j][i])             
        x = torch.cat((x, torch.tensor([n_pt])), 0)
    return x
  
def dtanh(s):
    # Derivative of activation
    return 1.0 - s**2

def dquadp(s):
    # Derivative of activation
    return 2*s

def Norm2(x):
    # 2-norm
    y = []
    for r in range(0,len(x)):
        v = 0 
        for j in range(x.shape[1]):
            v += x[r][j]**2
        f = [torch.sqrt(v)]
        y.append(f)
    y = torch.tensor(y)
    return y


def Tune(x):
    # Circle function values
    # Original function used by NLC script
    y = []
    for r in range(0,len(x)):
        v = 0 
        for j in range(x.shape[1]):
            v += x[r][j]**2
        f = [torch.sqrt(v)]
        y.append(f)
    y = torch.tensor(y)
    return y


def SaveWeightHist(model, parameters, w1_hist, q_hist, 
                   wc1_hist, wc2_hist, wc3_hist,
                   bc1_hist, bc2_hist, bc3_hist, 
                   out_iters, max_iters, i_epoch):
    # This function add the latest values of the weight and bias of the ANN to
    # the time evolution history
    
    # Parameters Lyapunov ANN
    w1 = model.layer1.weight.data.numpy()
    w2 = model.layer2.weight.data.numpy()
    w3 = model.layer3.weight.data.numpy()
    if parameters['Lyap_bias1']:
        b1 = model.layer1.bias.data.numpy()
    if parameters['Lyap_bias2']:
        b2 = model.layer2.bias.data.numpy()
    if parameters['Lyap_bias3']:
        b3 = model.layer3.bias.data.numpy()
    w1_hist[out_iters * max_iters + i_epoch] = w1

    # Parameters control ANN
    if parameters['use_lin_ctr']:
        q = model.control.weight.data.numpy()
        q_hist[out_iters * max_iters + i_epoch] = q
    else:
        wc1 = model.control1.weight.data.numpy()
        wc2 = model.control2.weight.data.numpy()
        wc3 = model.control3.weight.data.numpy()
        wc1_hist[out_iters * max_iters + i_epoch] = wc1
        wc2_hist[out_iters * max_iters + i_epoch] = wc2
        wc3_hist[out_iters * max_iters + i_epoch] = wc3
        
        if parameters['contr_bias1']:
            bc1 = model.control1.bias.data.numpy()
            bc1_hist[out_iters * max_iters + i_epoch] = bc1
        if parameters['contr_bias2']:
            bc2 = model.control2.bias.data.numpy()
            bc2_hist[out_iters * max_iters + i_epoch] = bc2
        if parameters['contr_bias3']:
            bc3 = model.control3.bias.data.numpy()
            bc3_hist[out_iters * max_iters + i_epoch] = bc3

    return w1_hist, q_hist, wc1_hist, wc2_hist, wc3_hist, bc1_hist, bc2_hist, bc3_hist




def AddLieViolationsOrder1_v2(x, 
                              ball_ub, 
                              grid_points,
                              config,
                              lie_derivative_of_V,
                              zeta_D,
                              debug_info, 
                              V_learn):
    
    # Add violation of the Lie derivative negativity for a 3rd order system.
    # Updated version where points are shuffled and the total number of 
    # violations is returned
    
    x1 = np.linspace(-ball_ub, ball_ub, grid_points)

    # 1) Lyapunov function 
    # evaluate expression Lie derivative
    V_str = V_learn.to_string() 
    V_sub = from_dreal_to_np.sub(V_str)  # substitute dreal functions
    V_eval = eval(V_sub)

    # finding violations V:
    V_l0 =  V_eval[V_eval<0]  # computing values V < 0 (V_l0 = "V less than 0")
    #print(f"LOG 'Discrete Falsifier': {V_l0.size} points V<0")

    # check where is the V condition falsified
    where_V_neg = torch.zeros(min(V_l0.size, zeta_D), 1)
    for iFals in range(where_V_neg.size(0)):
        rand_pos = random.randrange(where_V_neg.size(0))
        coord = np.where(V_eval == V_l0[rand_pos])
        where_V_neg[iFals] = torch.tensor([coord[0][0]])

    x_V_neg = torch.zeros(where_V_neg.size()[0], 1)
    for iP in range(where_V_neg.size()[0]):
        x_p = int(where_V_neg[iP,0].item())

        x_V_neg[iP] = torch.tensor([x1[x_p]])

    # 2) Lie derivative
    # evaluate expression Lie derivative
    V_lie_str = lie_derivative_of_V.to_string() 
    V_lie_sub = from_dreal_to_np.sub(V_lie_str)  # substitute dreal functions
    V_lie_eval = eval(V_lie_sub)
   
    # finding violations Lie der:
    V_Lie_g0 =  V_lie_eval[V_lie_eval>0]  # computing values Vdot> 0 (V_g0 = "V greater than 0")

    # check where is the Lie derivative condition falsified
    where_Vdot_pos = torch.zeros(min(V_Lie_g0.size, zeta_D), 1)
    for iFals in range(where_Vdot_pos.size(0)):
        rand_pos = random.randrange(where_Vdot_pos.size(0))
        coord = np.where(V_lie_eval == V_Lie_g0[rand_pos])
        where_Vdot_pos[iFals] = torch.tensor([coord[0][0]])

    x_Vdot_pos = torch.zeros(where_Vdot_pos.size()[0], 1)
    for iP in range(where_Vdot_pos.size()[0]):
        x_p = int(where_Vdot_pos[iP,0].item())
        x_Vdot_pos[iP] = torch.tensor([x1[x_p]])


    # 3) Finding overall number of points with violations
    x_violations = torch.cat((x_Vdot_pos, x_V_neg), 0)
    x_violations_shuffled = x_violations[torch.randperm(x_violations.size()[0])]

    # cutting points if too many are found
    if (len(x_violations) > zeta_D): 
        x_violations_shuffled = x_violations_shuffled[x_violations_shuffled.shape[0] - zeta_D : x_violations_shuffled.shape[0], :]

    # augmenting dataset
    x = torch.cat((x, x_violations_shuffled), 0)


    if debug_info:
        print(f"LOG 'Discrete Falsifier': Found {V_l0.size} points V<0")
        print(f"LOG 'Discrete Falsifier': Found {V_Lie_g0.size} points V_dot>0")
        print(f"LOG 'Discrete Falsifier': Added {len(x_violations_shuffled)} points")
    
    tot_violation = V_l0.size + V_Lie_g0.size
    
    return x, grid_points, tot_violation
   

def AddLieViolationsOrder2_v3(x, 
                              gamma_overbar, 
                              grid_points,
                              config,
                              lie_derivative_of_V,
                              zeta_D,
                              debug_info,
                              print_Vdot_values,
                              V_learn):
    
    # Add violation of the Lie derivative negativity for a 3rd order system.
    # Updated version where points are shuffled
    # With respect to v2, this function returns the CE_DF already shuffled

    upper_bound  = gamma_overbar
    lower_bound = - gamma_overbar

    X1 = np.linspace(lower_bound, upper_bound, grid_points)
    X2 = np.linspace(lower_bound, upper_bound, grid_points)
    x1, x2 = np.meshgrid(X1, X2)

    V_str = V_learn.to_string() 
    V_sub = from_dreal_to_np.sub(V_str)  # substitute dreal functions
    V_eval = eval(V_sub)
        
    # compute points V<0
    V_l0 = V_eval[V_eval<0]
    all_violations = torch.transpose(torch.tensor(np.where(V_eval<0)), 0, 1)  
    # what are the coordinates of the 'V < 0' points

    where_V_neg = all_violations# all_violations[0:zeta_D, :]
    x_V_neg = torch.zeros(where_V_neg.size()[0], 2)
    
    for iFals in range(where_V_neg.size(0)):
        c1 = where_V_neg[iFals][0]
        c2 = where_V_neg[iFals][1]
        x_V_neg[iFals] = torch.tensor([x1[c1, c2],
                                       x2[c1, c2]])


    # 2) Lie derivative
    # evaluate expression Lie derivative
    V_lie_str = lie_derivative_of_V.to_string() 
    V_lie_sub = from_dreal_to_np.sub(V_lie_str)  # substitute dreal functions
    V_lie_eval = eval(V_lie_sub)

    # finding violations (Lie der > 0):
    V_Lie_g0 =  V_lie_eval[V_lie_eval>0]  # computing values Vdot> 0
    all_violations = torch.transpose(torch.tensor(np.where(V_lie_eval>0)), 0, 1)  
    # what are the coordinates of the 'V_dot' > 0 points

    where_Vdot_pos = all_violations #all_violations[0:zeta_D, :]
    x_Vdot_pos = torch.zeros(where_Vdot_pos.size()[0], 2)

    for iFals in range(where_Vdot_pos.size(0)):
        c1 = where_Vdot_pos[iFals][0]
        c2 = where_Vdot_pos[iFals][1]
        
        x_Vdot_pos[iFals] =  torch.tensor([x1[c1, c2],
                                           x2[c1, c2]])
        
        if print_Vdot_values:
            valueVdot = eval_Vdot_2D(V_lie_sub, x1[c1, c2], x2[c1, c2])
        

    # 3) Finding overall number of points with violations
    x_violations = torch.cat((x_Vdot_pos, x_V_neg), 0)
    x_violations_shuffled = x_violations[torch.randperm(x_violations.size()[0])]


    # cutting points if too many are found
    if (len(x_violations) > zeta_D): 
        x_violations_shuffled = x_violations_shuffled[x_violations_shuffled.shape[0] - zeta_D : x_violations_shuffled.shape[0], :]


    # augmenting dataset
    x = torch.cat((x, x_violations_shuffled), 0)


    if debug_info:
        print(f"LOG 'Discrete Falsifier': Found {V_l0.size} points V<0")
        print(f"LOG 'Discrete Falsifier': Found {V_Lie_g0.size} points V_dot>0")
        print(f"LOG 'Discrete Falsifier': Added {len(x_violations_shuffled)} points")
    
    tot_violation = V_l0.size + V_Lie_g0.size
    
    return x, grid_points, tot_violation


def AddLieViolationsOrder3_v4(x, 
                              gamma_up, 
                              grid_points,
                              zeta_D,
                              debug_info, 
                              V_learn,
                              lie_derivative_of_V):
    
    # Add violation of the Lie derivative negativity for a 3rd order system.
    # Updated version where points are shuffled
    # With respect to v3, this function was purged from unused input parameters.
    
    lower_bound = - gamma_up
    upper_bound = gamma_up

    X1 = np.linspace(lower_bound, upper_bound, grid_points)
    X2 = np.linspace(lower_bound, upper_bound, grid_points)
    X3 = np.linspace(lower_bound, upper_bound, grid_points)
    x1, x2, x3 = np.meshgrid(X1, X2, X3)

    V_str = V_learn.to_string() 
    V_sub = from_dreal_to_np.sub(V_str)  # substitute dreal functions
    V_eval = eval(V_sub)
        
    ##
    V_l0 = V_eval[V_eval<0]
    all_violations = torch.transpose(torch.tensor(np.where(V_eval<0)), 0, 1)  
    # what are the coordinates of the 'V < 0' points

    where_V_neg = all_violations# all_violations[0:zeta_D, :]
    x_V_neg = torch.zeros(where_V_neg.size()[0], 3)
    
    for iFals in range(where_V_neg.size(0)):
        c1 = where_V_neg[iFals][0]
        c2 = where_V_neg[iFals][1]
        c3 = where_V_neg[iFals][2]
        x_V_neg[iFals] = torch.tensor([x1[c1, c2, c3],
                                       x2[c1, c2, c3],
                                       x3[c1, c2, c3]])    
    
    # 2) Lie derivative
    # evaluate expression Lie derivative
    V_lie_str = lie_derivative_of_V.to_string() 
    V_lie_sub = from_dreal_to_np.sub(V_lie_str)  # substitute dreal functions
    V_lie_eval = eval(V_lie_sub)

    # finding violations Lie der:
    V_Lie_g0 =  V_lie_eval[V_lie_eval>0]  # computing values Vdot> 0
    all_violations = torch.transpose(torch.tensor(np.where(V_lie_eval>0)), 0, 1)  
    # what are the coordinates of the 'V_dot' > 0 points

    where_Vdot_pos = all_violations #all_violations[0:zeta_D, :]
    x_Vdot_pos = torch.zeros(where_Vdot_pos.size()[0], 3)

    for iFals in range(where_Vdot_pos.size(0)):
        c1 = where_Vdot_pos[iFals][0]
        c2 = where_Vdot_pos[iFals][1]
        c3 = where_Vdot_pos[iFals][2]
        
        x_Vdot_pos[iFals] =  torch.tensor([x1[c1, c2, c3],
                                           x2[c1, c2, c3],
                                           x3[c1, c2, c3]])

    # 3) Finding overall number of points with violations
    x_violations = torch.cat((x_Vdot_pos, x_V_neg), 0)
    x_violations_shuffled = x_violations[torch.randperm(x_violations.size()[0])]


    # cutting points if too many are found
    if (len(x_violations) > zeta_D): 
        x_violations_shuffled = x_violations_shuffled[x_violations_shuffled.shape[0] - zeta_D : x_violations_shuffled.shape[0], :]

    # augmenting dataset
    x = torch.cat((x, x_violations_shuffled), 0)


    if debug_info:
        print(f"LOG 'Discrete Falsifier': Found {V_l0.size} points V<0")
        print(f"LOG 'Discrete Falsifier': Found {V_Lie_g0.size} points V_dot>0")
        print(f"LOG 'Discrete Falsifier': Added {len(x_violations_shuffled)} points")
    
    tot_violation = V_l0.size + V_Lie_g0.size
    
    return x, tot_violation


def AddLieViolationsOrder4_v2(x, 
                              gamma_up, 
                              grid_points,
                              config,
                              lie_derivative_of_V,
                              zeta_D,
                              debug_info, 
                              V_learn):

    # Add violation of the Lie derivative negativity for a 2nd order system
    
    x1 = np.linspace(-gamma_up, gamma_up, grid_points)
    x2 = np.linspace(-gamma_up, gamma_up, grid_points)
    x3 = np.linspace(-gamma_up, gamma_up, grid_points)
    x4 = np.linspace(-gamma_up, gamma_up, grid_points)
    x1, x2, x3, x4 = np.meshgrid(x1, x2, x3, x4)
    

    # 1) Lyapunov function 
    # evaluate expression Lie derivative
    V_str = V_learn.to_string() 
    V_sub = from_dreal_to_np.sub(V_str)  # substitute dreal functions
    V_eval = eval(V_sub)

    # finding violations V:
    V_l0 =  V_eval[V_eval<0]  # computing values V < 0 (V_l0 = "V less than 0")
    #print(f"LOG 'Discrete Falsifier': {V_l0.size} points V<0")

    # check where is the V condition falsified
    where_V_neg = torch.zeros(min(V_l0.size, zeta_D), 4)
    for iFals in range(where_V_neg.size(0)):
        rand_pos = random.randrange(where_V_neg.size(0))
        coord = np.where(V_eval == V_l0[rand_pos])
        where_V_neg[iFals] = torch.tensor([coord[0][0],
                                           coord[1][0],
                                           coord[2][0],
                                           coord[3][0]])

    x_V_neg = torch.zeros(where_V_neg.size()[0], 4)
    for iP in range(where_V_neg.size()[0]):
        p1 = int(where_V_neg[iP,0].item())
        p2 = int(where_V_neg[iP,1].item())
        p3 = int(where_V_neg[iP,2].item())
        p4 = int(where_V_neg[iP,3].item())
        x_V_neg[iP] = torch.tensor([x1[p1][p2][p3][p4], 
                                    x2[p1][p2][p3][p4],  
                                    x3[p1][p2][p3][p4], 
                                    x4[p1][p2][p3][p4]])
    
    
    # 2) Lie derivative
    # evaluate expression Lie derivative
    V_lie_str = lie_derivative_of_V.to_string() 
    V_lie_sub = from_dreal_to_np.sub(V_lie_str)  # substitute dreal functions
    V_lie_eval = eval(V_lie_sub)
   
    # finding violations Lie der:
    V_Lie_g0 =  V_lie_eval[V_lie_eval>0]  # computing values Vdot> 0 (V_g0 = "V greater than 0")
    #print(f"LOG 'Discrete Falsifier': {V_Lie_g0.size} points V_dot>0")

    # check where is the Lie derivative condition falsified
    where_Vdot_pos = torch.zeros(min(V_Lie_g0.size, zeta_D), 4)
    for iFals in range(where_Vdot_pos.size(0)):
        
        rand_pos = random.randrange(where_Vdot_pos.size(0))
        coord = np.where(V_lie_eval == V_Lie_g0[rand_pos])
        where_Vdot_pos[iFals] = torch.tensor([coord[0][0],
                                              coord[1][0],
                                              coord[2][0],
                                              coord[3][0]])

    x_Vdot_pos = torch.zeros(where_Vdot_pos.size()[0], 4)
    for iP in range(where_Vdot_pos.size()[0]):
        p1 = int(where_Vdot_pos[iP,0].item())
        p2 = int(where_Vdot_pos[iP,1].item())
        p3 = int(where_Vdot_pos[iP,2].item())
        p4 = int(where_Vdot_pos[iP,3].item())
        x_Vdot_pos[iP] =  torch.tensor([x1[p1][p2][p3][p4], 
                                        x2[p1][p2][p3][p4],  
                                        x3[p1][p2][p3][p4], 
                                        x4[p1][p2][p3][p4]])
        

    # 3) Finding overall number of points with violations
    x_violations = torch.cat((x_Vdot_pos, x_V_neg), 0)
    x_violations_shuffled = x_violations[torch.randperm(x_violations.size()[0])]

    # cutting points if too many are found
    if (len(x_violations) > zeta_D): 
        x_violations_shuffled = x_violations_shuffled[x_violations_shuffled.shape[0] - zeta_D : x_violations_shuffled.shape[0], :]

    # augmenting dataset
    x = torch.cat((x, x_violations_shuffled), 0)


    if debug_info:
        print(f"LOG 'Discrete Falsifier': Found {V_l0.size} points V<0")
        print(f"LOG 'Discrete Falsifier': Found {V_Lie_g0.size} points V_dot>0")
        print(f"LOG 'Discrete Falsifier': Added {len(x_violations_shuffled)} points")
    
    tot_violation = V_l0.size + V_Lie_g0.size

    
    return x, grid_points, tot_violation


def AddLieViolationsOrder6(x, 
                           gamma_up, 
                           grid_points,
                           adaptive_grid,
                           config,
                           lie_derivative_of_V,
                           zeta_D,
                           debug_info, 
                           V_learn):

    # Add violation of the Lie derivative negativity for a 2nd order system
    
    x1 = np.linspace(-gamma_up, gamma_up, grid_points)
    x2 = np.linspace(-gamma_up, gamma_up, grid_points)
    x3 = np.linspace(-gamma_up, gamma_up, grid_points)
    x4 = np.linspace(-gamma_up, gamma_up, grid_points)
    x5 = np.linspace(-gamma_up, gamma_up, grid_points)
    x6 = np.linspace(-gamma_up, gamma_up, grid_points)
    x1, x2, x3, x4, x5, x6 = np.meshgrid(x1, x2, x3, x4, x5, x6)  # TODO
    

    # 1) Lyapunov function 
    # evaluate expression Lie derivative
    V_str = V_learn.to_string() 
    V_sub = from_dreal_to_np.sub(V_str)  # substitute dreal functions
    V_eval = eval(V_sub)

    # finding violations V:
    V_l0 =  V_eval[V_eval<0]  # computing values V < 0 (V_l0 = "V less than 0")
    #print(f"LOG 'Discrete Falsifier': {V_l0.size} points V<0")

    # check where is the V condition falsified
    where_V_neg = torch.zeros(min(V_l0.size, 50), 6)
    for iFals in range(where_V_neg.size(0)):
        coord = np.where(V_eval == V_l0[iFals])
        where_V_neg[iFals] = torch.tensor([coord[0][0],
                                           coord[1][0],
                                           coord[2][0],
                                           coord[3][0],
                                           coord[4][0],
                                           coord[5][0]])

    x_V_neg = torch.zeros(where_V_neg.size()[0], 6)
    for iP in range(where_V_neg.size()[0]):
        p1 = int(where_V_neg[iP,0].item())
        p2 = int(where_V_neg[iP,1].item())
        p3 = int(where_V_neg[iP,2].item())
        p4 = int(where_V_neg[iP,3].item())
        p5 = int(where_V_neg[iP,4].item())
        p6 = int(where_V_neg[iP,5].item())
        x_V_neg[iP] = torch.tensor([x1[p1][p2][p3][p4][p5][p6], 
                                    x2[p1][p2][p3][p4][p5][p6],  
                                    x3[p1][p2][p3][p4][p5][p6], 
                                    x4[p1][p2][p3][p4][p5][p6], 
                                    x5[p1][p2][p3][p4][p5][p6], 
                                    x6[p1][p2][p3][p4][p5][p6]])
    
    
    # 2) Lie derivative
    # evaluate expression Lie derivative
    V_lie_str = lie_derivative_of_V.to_string() 
    V_lie_sub = from_dreal_to_np.sub(V_lie_str)  # substitute dreal functions
    V_lie_eval = eval(V_lie_sub)
   
    # finding violations Lie der:
    V_Lie_g0 =  V_lie_eval[V_lie_eval>0]  # computing values Vdot> 0 (V_g0 = "V greater than 0")
    #print(f"LOG 'Discrete Falsifier': {V_Lie_g0.size} points V_dot>0")

    # check where is the Lie derivative condition falsified
    where_Vdot_pos = torch.zeros(min(V_Lie_g0.size, 50), 6)
    for iFals in range(where_Vdot_pos.size(0)):
        coord = np.where(V_lie_eval == V_Lie_g0[iFals])
        where_Vdot_pos[iFals] = torch.tensor([coord[0][0],
                                              coord[1][0],
                                              coord[2][0],
                                              coord[3][0],
                                              coord[4][0],
                                              coord[5][0]])

    x_Vdot_pos = torch.zeros(where_Vdot_pos.size()[0], 6)
    for iP in range(where_Vdot_pos.size()[0]):
        p1 = int(where_Vdot_pos[iP,0].item())
        p2 = int(where_Vdot_pos[iP,1].item())
        p3 = int(where_Vdot_pos[iP,2].item())
        p4 = int(where_Vdot_pos[iP,3].item())
        p5 = int(where_Vdot_pos[iP,4].item())
        p6 = int(where_Vdot_pos[iP,5].item())
        x_Vdot_pos[iP] =  torch.tensor([x1[p1][p2][p3][p4][p5][p6], 
                                        x2[p1][p2][p3][p4][p5][p6],  
                                        x3[p1][p2][p3][p4][p5][p6], 
                                        x4[p1][p2][p3][p4][p5][p6], 
                                        x5[p1][p2][p3][p4][p5][p6], 
                                        x6[p1][p2][p3][p4][p5][p6]])
        

    # 3) Finding overall number of points with violations
    x_violations = torch.cat((x_Vdot_pos, x_V_neg), 0)
    x_violations_shuffled = x_violations[torch.randperm(x_violations.size()[0])]

    # cutting points if too many are found
    if (len(x_violations) > zeta_D): 
        x_violations_shuffled = x_violations_shuffled[x_violations_shuffled.shape[0] - zeta_D : x_violations_shuffled.shape[0], :]

    # augmenting dataset
    x = torch.cat((x, x_violations_shuffled), 0)


    if debug_info:
        print(f"LOG 'Discrete Falsifier': Found {V_l0.size} points V<0")
        print(f"LOG 'Discrete Falsifier': Found {V_Lie_g0.size} points V_dot>0")
        print(f"LOG 'Discrete Falsifier': Added {len(x_violations_shuffled)} points")
    
    tot_violation = V_l0.size + V_Lie_g0.size
    
    return x, grid_points, tot_violation
