#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 19:25:12 2022

@author: Davide Grande 

A script containing the discrete dynamics of an inverted pendulum.

"""
import numpy as np
import torch

#
# Pendulum dynamics
#
def pend(x, u, Dt, dyn_sys_params):
    
    G = dyn_sys_params.G
    L = dyn_sys_params.L
    m = dyn_sys_params.m
    b = dyn_sys_params.b
    Jz = m * L**2


    theta, omega = x
    dydt = [omega * Dt + theta,
            (m * G * L * np.sin(theta) - b * omega + u ) / Jz * Dt + omega]
    
    return torch.Tensor([dydt])
    
