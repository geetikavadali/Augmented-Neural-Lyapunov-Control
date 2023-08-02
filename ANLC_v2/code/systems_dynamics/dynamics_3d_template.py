#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:11:49 2022

@authors: Davide Grande
          Andrea Peruffo

A script containing the discrete dynamics of a Controlled Lorenz system.
The dynamics is integrated with a Forward Euler.

"""
import numpy as np
import torch

#
# Controlled Lorenz system
#
def dyn(x, u, Dt, parameters):

    # Model dynamics parameters
    sigma = parameters['sigma']
    b = parameters['b']
    r = parameters['r']

    x1, x2, x3 = x
    u1, u2, u3 = u

    dydt = [(-sigma*(x1-x2) + u1) * Dt + x1, 
            (r*x1 - x2 - x1*x3 + u2) * Dt + x2,
            (x1*x2 - b*x3 + u3) * Dt + x3]

    return torch.Tensor([dydt])
