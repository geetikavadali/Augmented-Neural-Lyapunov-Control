#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:33:56 2022

@author: Davide Grande
"""

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def plot_order_2(n_input, folder_results_plots, ce_found, x_dataset_init, x, 
                 dpi_, gamma_overbar):    


    # Generating CEs result folder
    try:
        folder_ds_plots = folder_results_plots + "/dataset_plots"
        current_dir = os.getcwd()
        final_dir_plots_ds = current_dir + "/" + folder_ds_plots + "/"
        os.mkdir(final_dir_plots_ds)
    except OSError:
        print("Creation of the dataset plot result directory %s failed" % final_dir_plots_ds)
    else:
        print("Plot dataset directory successfully created as: \n %s \n" % final_dir_plots_ds)


    # # Plot initial dataset
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    # ax.set_zlabel('$x_3$')
    # #plt.title('Counter examples')
    # ax.scatter3D(x_dataset_init[:,0], x_dataset_init[:,1], c='limegreen', s=15)
    # name_fig = "dataset_init"
    # plt.savefig(final_dir_plots_ds + name_fig + ".png", dpi=dpi_)
    # plt.savefig(final_dir_plots_ds + name_fig + ".pdf", format='pdf')
    # plt.close()    

    # # Plot final dataset (sum of CE and initial)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.set_xlabel('$x_1$')
    # ax.set_ylabel('$x_2$')
    # ax.set_zlabel('$x_3$')
    # ax.set_xlim(-ball_ub, ball_ub)
    # ax.set_ylim(-ball_ub, ball_ub)
    # ax.set_zlim(-ball_ub, ball_ub)
    # ax.scatter3D(ce_found[:, 0], ce_found[:, 1], s=15, label='$CE_{SMT}$')
    # ax.scatter3D(x_final[:,0], x_final[:,1], s=15, label='$CE_{DF}$')
    # ax.scatter3D(x_dataset_init[:, 0], x_dataset_init[:, 1], 
    #              c='limegreen', s=15, label='Intial points')
    # ax.legend()
    # name_fig = "dataset_init_plus_CEs"
    # plt.savefig(final_dir_plots_ds + name_fig+".png", dpi=dpi_)
    # plt.savefig(final_dir_plots_ds + name_fig+".pdf", format='pdf')
    # plt.close()
    
    
    # plot initial dataset
    plt.figure()
    plt.scatter(x_dataset_init[:,0], x_dataset_init[:,1], c='limegreen', s=15) 
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    plt.xlim(-gamma_overbar, gamma_overbar)
    plt.ylim(-gamma_overbar, gamma_overbar)
    name_fig = "dataset_init"
    plt.savefig(final_dir_plots_ds + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    # plot dataset CE
    plt.figure()
    plt.scatter(ce_found[:,0], ce_found[:,1], c='r', s=15) 
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    plt.xlim(-gamma_overbar, gamma_overbar)
    plt.ylim(-gamma_overbar, gamma_overbar)
    name_fig = "dataset_CE_SMT"
    plt.savefig(final_dir_plots_ds + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    # plot final dataset
    plt.figure()
    plt.scatter(x[:,0], x[:,1], s=15) 
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    name_fig = "dataset_final"
    plt.savefig(final_dir_plots_ds + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    # plot final dataset (sum of CE and initial)
    plt.figure()
    ax = plt.subplot(111)
    plt.scatter(ce_found[:,0], ce_found[:,1], s=15, label='$CE_{SMT}$')  
    plt.scatter(x[:,0], x[:,1], s=15, label='$CE_{DF}$') 
    plt.scatter(x_dataset_init[:,0], x_dataset_init[:,1], c='limegreen', s=15, label='Intial dataset') 
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    plt.legend(bbox_to_anchor=(-0.02, 1.02), loc="lower left", ncol=3)
    plt.xlim(-gamma_overbar, gamma_overbar)
    plt.ylim(-gamma_overbar, gamma_overbar)
    name_fig = "dataset_init_plus_CEs"
    plt.savefig(final_dir_plots_ds + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + '/' +name_fig+".pdf", format='pdf')
    plt.close()
    
    
    # plot final dataset (sum of CE and initial) inverted order
    plt.figure()
    ax = plt.subplot(111)
    plt.scatter(x_dataset_init[:,0], x_dataset_init[:,1], c='limegreen', s=15, label='Intial dataset') 
    plt.scatter(x[:,0], x[:,1], s=15, label='$CE_{DF}$') 
    plt.scatter(ce_found[:,0], ce_found[:,1], s=15, label='$CE_{SMT}$')  
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid()
    plt.legend(bbox_to_anchor=(-0.02, 1.02), loc="lower left", ncol=3)
    plt.xlim(-gamma_overbar, gamma_overbar)
    plt.ylim(-gamma_overbar, gamma_overbar)
    name_fig = "dataset_init_plus_CEs_other_order"
    plt.savefig(final_dir_plots_ds + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + '/' +name_fig+".pdf", format='pdf')
    plt.close()

    


def plot_order_3(n_input, folder_results_plots, ce_found, x_dataset_init, x_final, 
                 dpi_, ball_ub):    


    # Generating CEs result folder
    try:
        folder_ds_plots = folder_results_plots + "/dataset_plots"
        current_dir = os.getcwd()
        final_dir_plots_ds = current_dir + "/" + folder_ds_plots + "/"
        os.mkdir(final_dir_plots_ds)
    except OSError:
        print("Creation of the dataset plot result directory %s failed" % final_dir_plots_ds)
    else:
        print("Plot dataset directory successfully created as: \n %s \n" % final_dir_plots_ds)


    # Plot initial dataset
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    #plt.title('Counter examples')
    ax.scatter3D(x_dataset_init[:,0], x_dataset_init[:,1], x_dataset_init[:,2], c='limegreen', s=15)
    name_fig = "dataset_init"
    plt.savefig(final_dir_plots_ds + name_fig + ".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + name_fig + ".pdf", format='pdf')
    plt.close()    

    # Plot final dataset (sum of CE and initial)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_xlim(-ball_ub, ball_ub)
    ax.set_ylim(-ball_ub, ball_ub)
    ax.set_zlim(-ball_ub, ball_ub)
    ax.scatter3D(ce_found[:, 0], ce_found[:, 1], ce_found[:, 2], s=15, label='$CE_{SMT}$')
    ax.scatter3D(x_final[:,0], x_final[:,1], x_final[:,2], s=15, label='$CE_{DF}$')
    ax.scatter3D(x_dataset_init[:, 0], x_dataset_init[:, 1], x_dataset_init[:, 2], 
                 c='limegreen', s=15, label='Intial points')
    ax.legend()
    name_fig = "dataset_init_plus_CEs"
    plt.savefig(final_dir_plots_ds + name_fig+".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + name_fig+".pdf", format='pdf')
    plt.close()



def plot_order_3_v2(n_input, folder_results_plots, ce_found, x_dataset_init, x_final, 
                    dpi_, low_boundary, upper_boundary):    

    # Plotting dataset with axis bounded between specified lower and upper 
    # boundaries
    
    # Generating CEs result folder
    try:
        folder_ds_plots = folder_results_plots + "/dataset_plots"
        current_dir = os.getcwd()
        final_dir_plots_ds = current_dir + "/" + folder_ds_plots + "/"
        os.mkdir(final_dir_plots_ds)
    except OSError:
        print("Creation of the dataset plot result directory %s failed" % final_dir_plots_ds)
    else:
        print("Plot dataset directory successfully created as: \n %s \n" % final_dir_plots_ds)


    # Plot initial dataset
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    #plt.title('Counter examples')
    ax.scatter3D(x_dataset_init[:,0], x_dataset_init[:,1], x_dataset_init[:,2], c='limegreen', s=15)
    name_fig = "dataset_init"
    plt.savefig(final_dir_plots_ds + name_fig + ".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + name_fig + ".pdf", format='pdf')
    plt.close()    

    # Plot final dataset (sum of CE and initial)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_xlim(low_boundary, upper_boundary)
    ax.set_ylim(low_boundary, upper_boundary)
    ax.set_zlim(low_boundary, upper_boundary)
    ax.scatter3D(ce_found[:, 0], ce_found[:, 1], ce_found[:, 2], s=10, c='r', 
                 marker='x', alpha=1.0, label='$CE_{SMT}$')
    ax.scatter3D(x_final[:,0], x_final[:,1], x_final[:,2], s=10, label='$CE_{DF}$')
    ax.scatter3D(x_dataset_init[:, 0], x_dataset_init[:, 1], x_dataset_init[:, 2], 
                 c='limegreen', s=10, label='Intial points')
    ax.legend(loc='best')
    name_fig = "dataset_init_plus_CEs"
    plt.savefig(final_dir_plots_ds + name_fig+".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + name_fig+".pdf", format='pdf')
    plt.close()



def plot_order_4(n_input, folder_results_plots, ce_found, x_dataset_init, x_final, 
                 dpi_, ball_ub):    


    # Generating CEs result folder
    try:
        folder_ds_plots = folder_results_plots + "/dataset_plots"
        current_dir = os.getcwd()
        final_dir_plots_ds = current_dir + "/" + folder_ds_plots + "/"
        os.mkdir(final_dir_plots_ds)
    except OSError:
        print("Creation of the dataset plot result directory %s failed" % final_dir_plots_ds)
    else:
        print("Plot dataset directory successfully created as: \n %s \n" % final_dir_plots_ds)


    # 1) x1, x2, x3
    # Plot initial dataset
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    #plt.title('Counter examples')
    ax.scatter3D(x_dataset_init[:,0], x_dataset_init[:,1], x_dataset_init[:,2], c='limegreen', s=15)
    name_fig = "dataset_init_x1_x2_x3"
    plt.savefig(final_dir_plots_ds + name_fig + ".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + name_fig + ".pdf", format='pdf')
    plt.close()    

    # Plot final dataset (sum of CE and initial)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')
    ax.set_xlim(-ball_ub, ball_ub)
    ax.set_ylim(-ball_ub, ball_ub)
    ax.set_zlim(-ball_ub, ball_ub)
    ax.scatter3D(ce_found[:, 0], ce_found[:, 1], ce_found[:, 2], s=15, 
                 marker='x', alpha=1.0, label='$CE_{SMT}$')
    ax.scatter3D(x_final[:,0], x_final[:,1], x_final[:,2], s=15, label='$CE_{DF}$')
    ax.scatter3D(x_dataset_init[:, 0], x_dataset_init[:, 1], x_dataset_init[:, 2], 
                 c='limegreen', s=15, label='Intial points')
    ax.legend()
    name_fig = "dataset_init_plus_CEs_x1_x2_x3"
    plt.savefig(final_dir_plots_ds + name_fig+".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + name_fig+".pdf", format='pdf')
    plt.close()


    # 2) x1, x2, x4   
    # Plot initial dataset
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_4$')
    #plt.title('Counter examples')
    ax.scatter3D(x_dataset_init[:,0], x_dataset_init[:,1], x_dataset_init[:,3], c='limegreen', s=15)
    name_fig = "dataset_init_x1_x2_x4"
    plt.savefig(final_dir_plots_ds + name_fig + ".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + name_fig + ".pdf", format='pdf')
    plt.close()    

    # Plot final dataset (sum of CE and initial)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_4$')
    ax.set_xlim(-ball_ub, ball_ub)
    ax.set_ylim(-ball_ub, ball_ub)
    ax.set_zlim(-ball_ub, ball_ub)
    ax.scatter3D(ce_found[:, 0], ce_found[:, 1], ce_found[:, 3], s=15, 
                 marker='x', alpha=1.0, label='$CE_{SMT}$')
    ax.scatter3D(x_final[:,0], x_final[:,1], x_final[:,3], s=15, label='$CE_{DF}$')
    ax.scatter3D(x_dataset_init[:, 0], x_dataset_init[:, 1], x_dataset_init[:, 3], 
                 c='limegreen', s=15, label='Intial points')
    ax.legend()
    name_fig = "dataset_init_plus_CEs_x1_x2_x4"
    plt.savefig(final_dir_plots_ds + name_fig+".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + name_fig+".pdf", format='pdf')
    plt.close()


    # 3) x2, x3, x4   
    # Plot initial dataset
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$x_3$')
    ax.set_zlabel('$x_4$')
    #plt.title('Counter examples')
    ax.scatter3D(x_dataset_init[:,1], x_dataset_init[:,2], x_dataset_init[:,3], c='limegreen', s=15)
    name_fig = "dataset_init_x2_x3_x4"
    plt.savefig(final_dir_plots_ds + name_fig + ".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + name_fig + ".pdf", format='pdf')
    plt.close()    

    # Plot final dataset (sum of CE and initial)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('$x_2$')
    ax.set_ylabel('$x_3$')
    ax.set_zlabel('$x_4$')
    ax.set_xlim(-ball_ub, ball_ub)
    ax.set_ylim(-ball_ub, ball_ub)
    ax.set_zlim(-ball_ub, ball_ub)
    ax.scatter3D(ce_found[:, 0], ce_found[:, 1], ce_found[:, 3], s=15, 
                 marker='x', alpha=1.0, label='$CE_{SMT}$')
    ax.scatter3D(x_final[:,1], x_final[:,2], x_final[:,3], s=15, label='$CE_{DF}$')
    ax.scatter3D(x_dataset_init[:, 0], x_dataset_init[:, 1], x_dataset_init[:, 3], 
                 c='limegreen', s=15, label='Intial points')
    ax.legend()
    name_fig = "dataset_init_plus_CEs_x2_x3_x4"
    plt.savefig(final_dir_plots_ds + name_fig+".png", dpi=dpi_)
    plt.savefig(final_dir_plots_ds + name_fig+".pdf", format='pdf')
    plt.close()    
    
    