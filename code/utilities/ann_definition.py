#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:47:24 2023

@author: Davide Grande

A class to define an Feedforward ANN composed of 3 branches: 
1. a Lyapunov ANN;
2. a linear control ANN;
3. a nonlinear control ANN.

"""

import torch 

#
# Class defining the ANN architecture
#
class Net(torch.nn.Module):

    def __init__(self, parameters, seed_):
        super(Net, self).__init__()
        torch.manual_seed(seed_)
        
        n_input = parameters['n_input']
        lyap_hid1 = parameters['lyap_hid1']
        lyap_hid2 = parameters['lyap_hid2']
        n_output = parameters['n_output']
        Lyap_bias1 = parameters['Lyap_bias1']
        Lyap_bias2 = parameters['Lyap_bias2']
        Lyap_bias3 = parameters['Lyap_bias3']
        Lyap_act_fun1 = parameters['Lyap_act_fun1']
        Lyap_act_fun2 = parameters['Lyap_act_fun2']
        Lyap_act_fun3 = parameters['Lyap_act_fun3']
        contr_hid1 = parameters['contr_hid1']
        contr_hid2 = parameters['contr_hid2']
        contr_out = parameters['contr_out']
        contr_bias1 = parameters['contr_bias1']
        contr_bias2 = parameters['contr_bias2']
        contr_bias3 = parameters['contr_bias3']
        contr_act_fun1 = parameters['contr_act_fun1']
        contr_act_fun2 = parameters['contr_act_fun2']
        contr_act_fun3 = parameters['contr_act_fun3']
        use_lin_ctr = parameters['use_lin_ctr']
        lin_contr_bias = parameters['lin_contr_bias']
        init_control = parameters['init_control']
        
        
        self.layer1 = torch.nn.Linear(n_input, lyap_hid1, bias=Lyap_bias1)
        self.layer2 = torch.nn.Linear(lyap_hid1, lyap_hid2, bias=Lyap_bias2)
        self.layer3 = torch.nn.Linear(lyap_hid2, n_output, bias=Lyap_bias3)
        
        self.control1 = torch.nn.Linear(n_input, contr_hid1, bias=contr_bias1)
        self.control2 = torch.nn.Linear(contr_hid1, contr_hid2, bias=contr_bias2)
        self.control3 = torch.nn.Linear(contr_hid2, contr_out, bias=contr_bias3)
        
        self.control = torch.nn.Linear(n_input, contr_out, bias=lin_contr_bias)
        self.control.weight = torch.nn.Parameter(init_control)  
        
        print('============================================================')
        print("New Lyapunov Control architecture instantiated: \n")
        print("Lyapunov ANN: ")
        print(f"s1 = {Lyap_act_fun1}, s2 = {Lyap_act_fun2}, s3 = {Lyap_act_fun3}")
        print(f"b1 = {Lyap_bias1}, b2 = {Lyap_bias2}, b3 = {Lyap_bias3}")
        print(f"{n_input} x {lyap_hid1} x {lyap_hid2} x {n_output} \n")    
        if use_lin_ctr:
            print("Linear control ANN: ")
            print(f"Use linear control bias = {lin_contr_bias}")
            print(f"{n_input} x {contr_out}")
        else:
            print("Nonlinear control ANN: ")
            print(f"s1 = {contr_act_fun1}, s2 = {contr_act_fun2} s3 = {contr_act_fun3}")
            print(f"b1 = {contr_bias1}, b2 = {contr_bias2}, b3 = {contr_bias3}")
            print(f"{n_input} x {contr_hid1} x {contr_hid2} x {contr_out}")
        print('============================================================\n')


        # Setting attributes for forward method        
        self.Lyap_act_fun1 = parameters['Lyap_act_fun1']
        self.Lyap_act_fun2 = parameters['Lyap_act_fun2']
        self.Lyap_act_fun3 = parameters['Lyap_act_fun3']
        self.use_lin_ctr = parameters['use_lin_ctr']
        self.contr_act_fun1 = parameters['contr_act_fun1']
        self.contr_act_fun2 = parameters['contr_act_fun2']
        self.contr_act_fun3 = parameters['contr_act_fun3']
        self.beta_sfpl = parameters['beta_sfpl']
        

    def forward(self,x):
        act_fun_tanh = torch.nn.Tanh()
        act_fun_sfpl = torch.nn.Softplus(beta=self.beta_sfpl, threshold=50)
        
        # Lyapunov ANN
        if (self.Lyap_act_fun1 == 'tanh'):
            # using tanh activation function
            h_1 = act_fun_tanh(self.layer1(x))
        elif (self.Lyap_act_fun1 == 'pow2'): 
            # using quadratic polynomial activation function
            h_1 = self.layer1(x)**2
        elif (self.Lyap_act_fun1 == 'sfpl'): 
            # using softplus activation function
            h_1 = act_fun_sfpl(self.layer1(x))
        else: 
            # linear activation function
            h_1 = self.layer1(x)  

        if (self.Lyap_act_fun2 == 'tanh'):
            # using tanh activation function
            h_2 = act_fun_tanh(self.layer2(h_1))
        elif (self.Lyap_act_fun2 == 'pow2'): 
            # using quadratic polynomial activation function
            h_2 = self.layer2(h_1)**2
        elif (self.Lyap_act_fun2 == 'sfpl'): 
            # using softplus activation function
            h_2 = act_fun_sfpl(self.layer2(h_1))
        else:
            # linear activation function
            h_2 = self.layer2(h_1)

        # Output layer
        if (self.Lyap_act_fun3 == 'tanh'):
            # using tanh activation function
            V = act_fun_tanh(self.layer3(h_2))
        elif (self.Lyap_act_fun3 == 'pow2'): 
            # using quadratic polynomial activation function
            V = self.layer3(h_2)**2
        elif (self.Lyap_act_fun3 == 'sfpl'):
            # using softplus activation function
            V = act_fun_sfpl(self.layer3(h_2))
        else:
            # linear activation function
            V = self.layer3(h_2)  # no activation function on 2nd layer
        

        # Control ANN        
        if self.use_lin_ctr:
            u = self.control(x)
        else:
            # nonlinear control
            if self.contr_act_fun1 == 'tanh':
                u1 = act_fun_tanh(self.control1(x))
            elif self.contr_act_fun1 == 'sfpl':
                u1 = act_fun_sfpl(self.control1(x))
            else:
                u1 = self.control1(x)

            if self.contr_act_fun2 == 'tanh':
                u2 = act_fun_tanh(self.control2(u1))
            elif self.contr_act_fun2 == 'sfpl':
                u2 = act_fun_sfpl(self.control2(u1))
            else:
                u2 = self.control2(u1)
                
            if self.contr_act_fun3 == 'tanh':
                u = act_fun_tanh(self.control3(u2))
            elif self.contr_act_fun3 == 'sfpl':
                u = act_fun_sfpl(self.control3(u2))
            else:
                u = self.control3(u2)

        return V, u
        

#
# Class defining the ANN architecture 
# (version 0 -- original, only used from inverted pendulum scripts) 
#
class Net_v0(torch.nn.Module):

    def __init__(self, n_input, lyap_hid1, lyap_hid2, n_output, 
                 Lyap_act_fun1, Lyap_act_fun2, Lyap_act_fun3,
                 Lyap_bias1, Lyap_bias2, Lyap_bias3,
                 use_lin_ctr, lin_contr_bias, init_weight,
                 contr_hid1, contr_hid2, contr_out, 
                 contr_bias1, contr_bias2, contr_bias3, 
                 contr_act_fun1, contr_act_fun2, contr_act_fun3,
                 beta_sfpl):
        super(Net_v0, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, lyap_hid1, bias=Lyap_bias1)
        self.layer2 = torch.nn.Linear(lyap_hid1, lyap_hid2, bias=Lyap_bias2)
        self.layer3 = torch.nn.Linear(lyap_hid2, n_output, bias=Lyap_bias3)
        
        self.control1 = torch.nn.Linear(n_input, contr_hid1, bias=contr_bias1)
        self.control2 = torch.nn.Linear(contr_hid1, contr_hid2, bias=contr_bias2)
        self.control3 = torch.nn.Linear(contr_hid2, contr_out, bias=contr_bias3)
        
        self.control = torch.nn.Linear(n_input, contr_out, bias=lin_contr_bias)
        self.control.weight = torch.nn.Parameter(init_weight)  
        
        print('============================================================')
        print("New Lyapunov Control architecture instantiated: \n")
        print("Lyapunov ANN: ")
        print(f"s1 = {Lyap_act_fun1}, s2 = {Lyap_act_fun2}, s3 = {Lyap_act_fun3}")
        print(f"b1 = {Lyap_bias1}, b2 = {Lyap_bias2}, b3 = {Lyap_bias3}")
        print(f"{n_input} x {lyap_hid1} x {lyap_hid2} x {n_output} \n")    
        if use_lin_ctr:
            print("Linear control ANN: ")
            print(f"Use linear control bias = {lin_contr_bias}")
            print(f"{n_input} x {contr_out}")
        else:
            print("Nonlinear control ANN: ")
            print(f"s1 = {contr_act_fun1}, s2 = {contr_act_fun2} s3 = {contr_act_fun3}")
            print(f"b1 = {contr_bias1}, b2 = {contr_bias2}, b3 = {contr_bias3}")
            print(f"{n_input} x {contr_hid1} x {contr_hid2} x {contr_out}")
        print('============================================================\n')


        # Setting attributes for forward method
        self.Lyap_act_fun1 = Lyap_act_fun1
        self.Lyap_act_fun2 = Lyap_act_fun2
        self.Lyap_act_fun3 = Lyap_act_fun3
        self.use_lin_ctr = use_lin_ctr
        self.contr_act_fun1 = contr_act_fun1
        self.contr_act_fun2 = contr_act_fun2
        self.contr_act_fun3 = contr_act_fun3
        self.beta_sfpl = beta_sfpl
        
        

    def forward(self,x):
        act_fun_tanh = torch.nn.Tanh()
        act_fun_sfpl = torch.nn.Softplus(beta=self.beta_sfpl, threshold=50)
        
        # Lyapunov ANN
        if (self.Lyap_act_fun1 == 'tanh'):
            # using tanh activation function
            h_1 = act_fun_tanh(self.layer1(x))
        elif (self.Lyap_act_fun1 == 'pow2'): 
            # using quadratic polynomial activation function
            h_1 = self.layer1(x)**2
        elif (self.Lyap_act_fun1 == 'sfpl'): 
            # using softplus activation function
            h_1 = act_fun_sfpl(self.layer1(x))
        else: 
            # linear activation function
            h_1 = self.layer1(x)  

        
        if (self.Lyap_act_fun2 == 'tanh'):
            # using tanh activation function
            h_2 = act_fun_tanh(self.layer2(h_1))
        elif (self.Lyap_act_fun2 == 'pow2'): 
            # using quadratic polynomial activation function
            h_2 = self.layer2(h_1)**2
        elif (self.Lyap_act_fun2 == 'sfpl'): 
            # using softplus activation function
            h_2 = act_fun_sfpl(self.layer2(h_1))
        else:
            # linear activation function
            h_2 = self.layer2(h_1)
        

        # Output layer
        if (self.Lyap_act_fun3 == 'tanh'):
            # using tanh activation function
            V = act_fun_tanh(self.layer3(h_2))
        elif (self.Lyap_act_fun3 == 'pow2'): 
            # using quadratic polynomial activation function
            V = self.layer3(h_2)**2
        elif (self.Lyap_act_fun3 == 'sfpl'):
            # using softplus activation function
            V = act_fun_sfpl(self.layer3(h_2))
        else:
            # linear activation function
            V = self.layer3(h_2)  # no activation function on 2nd layer
        

        # Control ANN        
        if self.use_lin_ctr:
            u = self.control(x)
        else:
            # nonlinear control
            if self.contr_act_fun1 == 'tanh':
                u1 = act_fun_tanh(self.control1(x))
            elif self.contr_act_fun1 == 'sfpl':
                u1 = act_fun_sfpl(self.control1(x))
            else:
                u1 = self.control1(x)

            if self.contr_act_fun2 == 'tanh':
                u2 = act_fun_tanh(self.control2(u1))
            elif self.contr_act_fun2 == 'sfpl':
                u2 = act_fun_sfpl(self.control2(u1))         
            else:
                u2 = self.control2(u1)
                
            if self.contr_act_fun3 == 'tanh':
                u = act_fun_tanh(self.control3(u2))
            elif self.contr_act_fun3 == 'sfpl':
                u = act_fun_sfpl(self.control3(u2))         
            else:
                u = self.control3(u2)

        return V, u
