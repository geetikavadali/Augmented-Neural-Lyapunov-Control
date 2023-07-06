#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:33:29 2023

@author: Davide Grande
"""

import numpy as np
import matplotlib.pyplot as plt
import utilities.from_dreal_to_np as from_dreal_to_np
import re
from matplotlib import cm


def plot(sym_expression, n_points, gamma_up, normalise_plot, use_wireframe,
         title, iteration_no, folder_results_plots, save_pdf, dpi_):
    
    
    '''
    First variable
    '''    
    X = np.linspace(-gamma_up, gamma_up, n_points) 
    Y = np.linspace(-gamma_up, gamma_up, n_points)
    x1, x2 = np.meshgrid(X, Y)
    
    V_str = sym_expression.to_string() 
    V_sub = from_dreal_to_np.sub(V_str)  # substitute dreal functions
    try:
        out_str_x3 = re.sub(r'x3', r'0', V_sub)
    except:
        print("No x1 variable")
    else:
        in_str_x3 = out_str_x3
    V_eval_x3 = eval(in_str_x3)
    
    if normalise_plot:
        plot_ = V_eval_x3/V_eval_x3.max()
    else:
        plot_ = V_eval_x3
    
    
    # Plot 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Plot Valid region computed by dReal
    theta = np.linspace(0,2*np.pi,50)
    xc = gamma_up*np.cos(theta)
    yc = gamma_up*np.sin(theta)
    ax.plot(xc[:],yc[:],'r', linestyle='--', linewidth=2 ,label='$\mathscr{D}$')
    plt.legend(loc='upper right')
    
    if use_wireframe:
        surf = ax.plot_wireframe(x1, x2, plot_, rstride=5, cstride=5, alpha=0.8)
    else:
        surf = ax.plot_surface(x1, x2, plot_, rstride=5, cstride=5, alpha=0.5, cmap=cm.winter)        
        ax.contour(x1, x2, plot_, 10, zdir='z', offset=0, cmap=cm.winter)
    
    #cb = plt.colorbar(surf, pad=0.2)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    name_fig = title + "x1_x2_" + iteration_no
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    if save_pdf: plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()


    '''
    Second variable
    '''   
    X = np.linspace(-gamma_up, gamma_up, n_points) 
    Y = np.linspace(-gamma_up, gamma_up, n_points)
    x1, x3 = np.meshgrid(X, Y)
    
    V_str = sym_expression.to_string() 
    V_sub = from_dreal_to_np.sub(V_str)  # substitute dreal functions
    try:
        out_str_x2 = re.sub(r'x2', r'0', V_sub)
    except:
        print("No x2 variable")
    else:
        in_str_x2 = out_str_x2
    V_eval_x2 = eval(in_str_x2)
    
    if normalise_plot:
        plot_ = V_eval_x2/V_eval_x2.max()
    else:
        plot_ = V_eval_x2
  
    
    # Plot 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Plot Valid region computed by dReal
    theta = np.linspace(0,2*np.pi,50)
    xc = gamma_up*np.cos(theta)
    yc = gamma_up*np.sin(theta)
    ax.plot(xc[:],yc[:],'r', linestyle='--', linewidth=2 ,label='$\mathscr{D}$')
    plt.legend(loc='upper right')
    
    
    if use_wireframe:
        surf = ax.plot_wireframe(x1, x3, plot_, rstride=5, cstride=5, alpha=0.8)
    else:
        surf = ax.plot_surface(x1, x3, plot_, rstride=5, cstride=5, alpha=0.5, cmap=cm.winter)        
        ax.contour(x1, x3, plot_, 10, zdir='z', offset=0, cmap=cm.winter)
        
    #cb = plt.colorbar(surf, pad=0.2)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_3$')
    name_fig = title + "x1_x3_" + iteration_no
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    if save_pdf: plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    
    '''
    Third variable
    '''
    X = np.linspace(-gamma_up, gamma_up, n_points) 
    Y = np.linspace(-gamma_up, gamma_up, n_points)
    x2, x3 = np.meshgrid(X, Y)
    
    V_str = sym_expression.to_string() 
    V_sub = from_dreal_to_np.sub(V_str)  # substitute dreal functions
    try:
        out_str_x1 = re.sub(r'x1', r'0', V_sub)
    except:
        print("No x1 variable")
    else:
        in_str_x1 = out_str_x1
    V_eval_x1 = eval(in_str_x1)
    
    if normalise_plot:
        plot_ = V_eval_x1/V_eval_x1.max()
    else:
        plot_ = V_eval_x1
    
    # Plot 
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Plot Valid region computed by dReal
    theta = np.linspace(0,2*np.pi,50)
    xc = gamma_up*np.cos(theta)
    yc = gamma_up*np.sin(theta)
    ax.plot(xc[:],yc[:],'r', linestyle='--', linewidth=2 ,label='$\mathscr{D}$')
    plt.legend(loc='upper right')


    if use_wireframe:
        surf = ax.plot_wireframe(x1, x2, plot_, rstride=5, cstride=5, alpha=0.8)
    else:
        surf = ax.plot_surface(x2, x3, plot_, rstride=5, cstride=5, alpha=0.5, cmap=cm.winter)
        ax.contour(x2, x3, plot_, 10, zdir='z', offset=0, cmap=cm.winter)

    #cb = plt.colorbar(surf, pad=0.2)
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$x_3$')
    name_fig = title + "x2_x3_" + iteration_no
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    if save_pdf: plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    
    
    