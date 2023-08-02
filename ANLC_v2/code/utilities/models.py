#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:42:24 2022

Computing the symbolic dynamics of a generic 3-dimensional system.

@authors: Andrea Peruffo
          Davide Grande

"""

import numpy as np
import dreal
import torch


# test System Dynamics
class test_f():
    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]

    @staticmethod
    def f_torch(x, u, parameters):

        x_dot = [- 0.5 * x[:, 0],
                 - x[:, 1] + u[:, 0] ]

        x_dot = torch.transpose(torch.stack(x_dot), 0, 1)

        return x_dot

    @staticmethod
    def f_symb(x, u, parameters):
        x1, x2 = x
        u1 = u[0]
        return [- 0.5 * x1, - x2 + u1]


class YourModel2D():

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]

    @staticmethod
    def f_torch(x, u, parameters):

        x1, x2 = x[:,0], x[:,1]
        x1_shift = x1 + parameters['x_star'][0]
        x2_shift = x2 + parameters['x_star'][1]
        u_NN0, u_NN1, u_NN2 = u[:,0], u[:,1], u[:,2]

        sigma = parameters['sigma']
        r = parameters['r']

        x_dot = [-sigma * (x1_shift - x2_shift) + u_NN0, r * x1_shift - x2_shift - x1_shift + u_NN1 + u_NN2]

        x_dot = torch.transpose(torch.stack(x_dot), 0, 1)

        return x_dot


    @staticmethod
    def f_symb(x, u, parameters):

        x1, x2 = x[0], x[1]
        x1_shift = x1 + parameters['x_star'].numpy()[0]
        x2_shift = x2 + parameters['x_star'].numpy()[1]
        u_NN0, u_NN1, u_NN2 = u[0], u[1], u[2]

        sigma = parameters['sigma']
        r = parameters['r']

        return [-sigma * (x1_shift - x2_shift) + u_NN0, r*x1_shift - x2_shift - x1_shift + u_NN1 + u_NN2]


class Pendulum():

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]

    @staticmethod
    def f_torch(x, u, parameters):

        x1, x2 = x[:,0], x[:,1]
        x1_shift = x1 + parameters['x_star'][0]
        x2_shift = x2 + parameters['x_star'][1]
        u_NN0 = u[:,0]

        G = parameters['G']
        L = parameters['L']
        m = parameters['m']
        b = parameters['b']

        x_dot = [x2_shift,
                 (m*G*L*np.sin(x1_shift) - b*x2_shift + u_NN0) / (m*L**2)]

        x_dot = torch.transpose(torch.stack(x_dot), 0, 1)

        return x_dot


    @staticmethod
    def f_symb(x, u, parameters):

        x1, x2 = x[0], x[1]
        x1_shift = x1 + parameters['x_star'].numpy()[0]
        x2_shift = x2 + parameters['x_star'].numpy()[1]
        u_NN0 = u[0]

        G = parameters['G']
        L = parameters['L']
        m = parameters['m']
        b = parameters['b']
        
        x_dot = [x2_shift,
                 (m*G*L*dreal.sin(x1_shift) - b*x2_shift + u_NN0) / (m*L**2)]

        return x_dot


class AUV2D_3Thrusters():

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2")]

    @staticmethod
    def f_torch(x, u, parameters):

        x1, x2 = x[:,0], x[:,1]
        x1_shift = x1 + parameters['x_star'][0]
        x2_shift = x2 + parameters['x_star'][1]
        u_NN0 = u[:,0]
        u_NN1 = u[:,1]
        u_NN2 = u[:,2]       

        m = parameters['m']
        Jz = parameters['Jz']
        Xu = parameters['Xu']
        Xuu = parameters['Xuu']
        Nr = parameters['Nr']
        Nrr = parameters['Nrr']
        l1x = parameters['l1x']
        l1y = parameters['l1y']
        alpha1 = parameters['alpha1']
        l2x = parameters['l2x']
        l2y = parameters['l2y']
        alpha2 = parameters['alpha2']
        l3x = parameters['l3x']
        l3y = parameters['l3y']
        alpha3 = parameters['alpha3']
    
        h1 = parameters['h1']
        h2 = parameters['h2']
        h3 = parameters['h3']

        F1x = u_NN0*torch.sin(torch.tensor(alpha1))
        F1y = u_NN0*torch.cos(torch.tensor(alpha1))
        F2x = u_NN1*torch.sin(torch.tensor(alpha2))
        F2y = u_NN1*torch.cos(torch.tensor(alpha2))
        F3x = u_NN2*torch.sin(torch.tensor(alpha3))
        F3y = u_NN2*torch.cos(torch.tensor(alpha3))
    
        x_dot = [(-Xu*x1_shift-Xuu*x1_shift**2 + F1x*h1                + F2x*h2                + F3x*h3                )/m,
                 (-Nr*x2_shift-Nrr*x2_shift**2 + (-F1x*l1y+F1y*l1x)*h1 + (-F2x*l2y+F2y*l2x)*h2 + (-F3x*l3y+F3y*l3x)*h3 )/Jz]


        x_dot = torch.transpose(torch.stack(x_dot), 0, 1)

        return x_dot


    @staticmethod
    def f_symb(x, u, parameters):

        x1, x2 = x[0], x[1]
        x1_shift = x1 + parameters['x_star'].numpy()[0]
        x2_shift = x2 + parameters['x_star'].numpy()[1]
        u_NN0 = u[0]
        u_NN1 = u[1]
        u_NN2 = u[2]    

        m = parameters['m']
        Jz = parameters['Jz']
        Xu = parameters['Xu']
        Xuu = parameters['Xuu']
        Nr = parameters['Nr']
        Nrr = parameters['Nrr']
        l1x = parameters['l1x']
        l1y = parameters['l1y']
        alpha1 = parameters['alpha1']
        l2x = parameters['l2x']
        l2y = parameters['l2y']
        alpha2 = parameters['alpha2']
        l3x = parameters['l3x']
        l3y = parameters['l3y']
        alpha3 = parameters['alpha3']
    
        h1 = parameters['h1']
        h2 = parameters['h2']
        h3 = parameters['h3']

        F1x = u_NN0*dreal.sin(alpha1)
        F1y = u_NN0*dreal.cos(alpha1)
        F2x = u_NN1*dreal.sin(alpha2)
        F2y = u_NN1*dreal.cos(alpha2)
        F3x = u_NN2*dreal.sin(alpha3)
        F3y = u_NN2*dreal.cos(alpha3)

        x_dot = [(-Xu*x1_shift-Xuu*x1_shift**2 + F1x*h1                + F2x*h2                + F3x*h3                 )/m,
                 (-Nr*x2_shift-Nrr*x2_shift**2 + (-F1x*l1y+F1y*l1x)*h1 + (-F2x*l2y+F2y*l2x)*h2 + (-F3x*l3y+F3y*l3x)*h3  )/Jz]

        return x_dot



class ControlledLorenz():

    def __init__(self):
        self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2"), dreal.Variable("x3")]

    @staticmethod
    def f_torch(x, u, parameters):

        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        x1_shift = x1 + parameters['x_star'][0]
        x2_shift = x2 + parameters['x_star'][1]
        x3_shift = x3 + parameters['x_star'][2]
        u_NN0 = u[:,0]
        u_NN1 = u[:,1]
        u_NN2 = u[:,2]

        sigma = parameters['sigma']
        b = parameters['b']
        r = parameters['r']

        
        x_dot = [-sigma*(x1_shift - x2_shift) + u[:, 0],
                 r*x1_shift - x2_shift - x1_shift*x3_shift + u[:, 1],
                 x1_shift * x2_shift - b*x3_shift + u[:, 2]]

        x_dot = torch.transpose(torch.stack(x_dot), 0, 1)

        return x_dot


    @staticmethod
    def f_symb(x, u, parameters):

        x1, x2, x3 = x[0], x[1], x[2]
        x1_shift = x1 + parameters['x_star'].numpy()[0]
        x2_shift = x2 + parameters['x_star'].numpy()[1]
        x3_shift = x3 + parameters['x_star'].numpy()[2]
        u_NN0 = u[0]
        u_NN1 = u[1]
        u_NN2 = u[2]

        sigma = parameters['sigma']
        b = parameters['b']
        r = parameters['r']


        x_dot = [-sigma*(x1_shift-x2_shift) + u_NN0, 
                 r*x1_shift - x2_shift - x1_shift*x3_shift + u_NN1, 
                 x1_shift*x2_shift - b*x3_shift + u_NN2]

        return x_dot
