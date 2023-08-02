#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:14:45 2023

@authors: Davide Grande
          Andrea Peruffo

A script containing the discrete dynamics of an inverted pendulum.

"""
import numpy as np
import torch

#
# Pendulum dynamics
#
def dyn(x, u, Dt, parameters):
    
    G = parameters['G']
    L = parameters['L']
    m = parameters['m']
    b = parameters['b']
    Jz = m * L**2

    x1, x2 = x
    u1 = u

    x_dot = [x2 * Dt + x1,
             (m*G*L*np.sin(x1) - b*x2 + u1) / Jz * Dt + x2 ]
    
    return torch.Tensor([x_dot])
    