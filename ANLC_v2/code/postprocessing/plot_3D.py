#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:20:47 2023

@authors: Davide Grande
          Andrea Peruffo

Given a 4D function defined as f(x1, x2, x3), this function
iteratively sets one of the three variables to zero, and returns the 3D plot
of the remaining two. 

"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import re
import random
import utilities.from_dreal_to_np as from_dreal_to_np


def plot(sym_expr, n_points, ball_ub, title_str, plot_title, 
         folder_results_plots, dpi_, Plot3D):
    
    if(n_points>1000):
        print("CAVEAT: the plot is being generated, please be patient ...")
    
    # inputs
    x_span = np.linspace(-ball_ub, ball_ub, n_points) 
    y_span = np.linspace(-ball_ub, ball_ub, n_points)
    z_span = np.linspace(-ball_ub, ball_ub, n_points)
    
    expr_str = sym_expr.to_string() 
    expr_sub = from_dreal_to_np.sub(expr_str)  # substitute dreal functions
    

    # 1) removing x3
    x1, x2 = np.meshgrid(x_span, y_span)
    try:
        out_str_x3 = re.sub(r'x3' , r'0', expr_sub)
    except:
        print("No x3 variable")
    else:
        in_str_x3 = out_str_x3
    expr_eval_x3 = eval(in_str_x3)
    
    # birdeye view (x1, x2)
    ax = Plot3D(x1, x2, expr_eval_x3, ball_ub)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel(f'{title_str}')
    if plot_title:
        plt.title(title_str)
    name_fig = f'{title_str}_no_x3'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()


    # contour plot
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(x1, x2, expr_eval_x3)
    fig.colorbar(cp)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    name_fig = f'{title_str}_no_x3_contour'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    #plt.show()


    # 2) removing x2
    x1, x3 = np.meshgrid(x_span, z_span)
    try:
        out_str_x2 = re.sub(r'x2' , r'0', expr_sub)
    except:
        print("No x2 variable")
    else:
        in_str_x2 = out_str_x2
    expr_eval_x2 = eval(in_str_x2)
    
    # birdeye view (x1, x3)
    ax = Plot3D(x1, x3, expr_eval_x2, ball_ub)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_3$')
    ax.set_zlabel(f'{title_str}')
    if plot_title:
        plt.title(title_str)
    name_fig = f'{title_str}_no_x2'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    
    # contour plot
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(x1, x3, expr_eval_x2)
    fig.colorbar(cp)
    ax.set_xlabel('x1')
    ax.set_ylabel('x3')
    name_fig = f'{title_str}_no_x2_contour'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    #plt.show()
    

    # 3) removing x1
    x2, x3 = np.meshgrid(y_span, z_span)
    try:
        out_str_x1 = re.sub(r'x1' , r'0', expr_sub)
    except:
        print("No x1 variable")
    else:
        in_str_x1 = out_str_x1
    expr_eval_x1 = eval(in_str_x1)
    
    # birdeye view (x2, x3)
    ax = Plot3D(x2, x3, expr_eval_x1, ball_ub)
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$x_3$')
    ax.set_zlabel(f'{title_str}')
    if plot_title:
        plt.title(title_str)
    name_fig = f'{title_str}_no_x1'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    # contour plot
    fig,ax=plt.subplots(1,1)
    cp = ax.contourf(x2, x3, expr_eval_x2)
    fig.colorbar(cp)
    ax.set_xlabel('x2')
    ax.set_ylabel('x3')
    name_fig = f'{title_str}_no_x1_contour'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    #plt.show()



def plot_dimG2(expr_eval, a1, a2, n_points, title_str, plot_title, axis_1, axis_2,
               folder_results_plots, dpi_, parameters, Plot3D):

    # plot 3D-slices of higher dimensionality functions 

    if(n_points>1000):
        print("CAVEAT: the plot is being generated, please be patient ...")

    # birdeye view (a1, a2)
    ax = Plot3D(a1, a2, expr_eval, parameters['gamma_overbar'])
    ax.set_xlabel(f'${axis_1}$')
    ax.set_ylabel(f'${axis_2}$')
    ax.set_zlabel(f'{title_str}')
    if plot_title:
        plt.title(title_str)
    name_fig = f'{title_str}'
    plt.savefig(folder_results_plots + '/' +name_fig + "_" + axis_1 + "_" + axis_2 +".png", dpi=dpi_)
    plt.close()

