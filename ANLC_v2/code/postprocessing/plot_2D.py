#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 23:44:05 2023

@authors: Davide Grande
          Andrea Peruffo

This function plots a 3D function, such as a control function u(x1, x2).
 
"""

import matplotlib.pyplot as plt
import numpy as np
import utilities.from_dreal_to_np as from_dreal_to_np
from matplotlib import cm


def plot(sym_expr, n_points, gamma_overbar, title_str, plot_title, 
         folder_results_plots, dpi_, Plot3D):
    
    if(n_points>1000):
        print("CAVEAT: the plot is being generated, please be patient ...")
    
    # inputs
    x_span = np.linspace(-gamma_overbar, gamma_overbar, n_points) 
    y_span = np.linspace(-gamma_overbar, gamma_overbar, n_points)
    x1, x2 = np.meshgrid(x_span, y_span)
    
    
    expr_str = sym_expr.to_string() 
    expr_sub = from_dreal_to_np.sub(expr_str)  # substitute dreal functions
    expr_eval = eval(expr_sub)
    
    # birdeye view (x1, x2)
    ax = Plot3D(x1, x2, expr_eval, gamma_overbar)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    if plot_title:
        plt.title(title_str)
    name_fig = f'{title_str}'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()


    # contour plot
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(x1, x2, expr_eval)
    fig.colorbar(cp)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    name_fig = f'{title_str}_contour'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    #plt.show()   
    
    
    # Plot mesh
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(elev=20, azim=-60)
    #surf = ax.plot_surface(x1, x2, expr_eval, rstride=5, cstride=5, alpha=0.5, cmap=cm.winter)
    surf = ax.plot_wireframe(x1, x2, expr_eval, rstride=5, cstride=5, alpha=0.8)
    theta = np.linspace(0,2*np.pi,50)      # Plot Valid region computed by dReal
    xc = gamma_overbar*np.cos(theta)
    yc = gamma_overbar*np.sin(theta)
    ax.plot(xc[:], yc[:],'r', linestyle='--', linewidth=2, label='$\mathscr{D}$')
    plt.legend(loc='upper right')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    name_fig = f'{title_str}_mesh'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    
    

