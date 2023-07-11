#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:20:50 2021

@author: Davide Grande

Plotting evolution of the weights of the ANN during the training.
"""
import matplotlib.pyplot as plt
import os


def plot_w(w1_hist, w2_hist, b1_hist, b2_hist, save_figures, folder_results):

    # plot layer 'input' to layer 'hidden'
    for i in range(w1_hist.shape[2]):  # input dimension

        for j in range(w1_hist.shape[1]):  # output dimension

            plt.figure()
            plt.plot(w1_hist[:, j, i])
            plt.xlabel('Iterations')
            title_fig = "Layer input - Weight Neuron 0," + str(i) +\
                " to Neuron 1," + str(j)
            plt.title(title_fig)
            plt.grid()
            if save_figures:
                name_fig = '/weight_layer_1_neuron_0_' + str(i) + '_to_1_' +\
                    str(j) + '.png'
                plt.savefig(folder_results + name_fig, dpi=300)
            plt.close()


    # plot layer 'hidden' to 'ouput' - bias
    for i in range(b1_hist.shape[1]):  # input dimension

        plt.figure()
        plt.plot(b1_hist[:, i])
        plt.xlabel('Iterations')
        title_fig = "Layer input - Bias Neuron 1," + str(i) +\
            " to Neuron 2"
        plt.title(title_fig)
        plt.grid()
        if save_figures:
            name_fig = '/bias_layer_1_neuron_0_' + str(i) + '_to_1_' + '.png'
            plt.savefig(folder_results + name_fig, dpi=300)
        plt.close()

    # plot layer 'hidden' to layer 'output'
    for i in range(w2_hist.shape[2]):  # input dimension

        for j in range(w2_hist.shape[1]):  # output dimension

            plt.figure()
            plt.plot(w2_hist[:, j, i])
            plt.xlabel('Iterations')
            title_fig = "Layer hidden - Weight Neuron 1," + str(i) +\
                " to Neuron 2," + str(j)
            plt.title(title_fig)
            plt.grid()
            if save_figures:
                name_fig = '/weight_layer_2_neuron_0_' + str(i) + '_to_2_' + '.png'
                plt.savefig(folder_results + name_fig, dpi=300)
            plt.close()


    # plot layer 'hidden' to 'ouput' - bias
    for i in range(b2_hist.shape[1]):  # input dimension

        plt.figure()
        plt.plot(b2_hist[:, i])
        plt.xlabel('Iterations')
        title_fig = "Layer input - Bias Neuron 1," + str(i) +\
            " to Neuron 2"
        plt.title(title_fig)
        plt.grid()
        if save_figures:
            name_fig = '/bias_layer_2_neuron_0_' + str(i) + '_to_1_' + '.png'
            plt.savefig(folder_results + name_fig, dpi=300)
        plt.close()



def plot_w_control(wc1_hist, wc2_hist, wc3_hist, wc4_hist, bc1_hist, bc2_hist, bc3_hist, save_figures, folder_results):

    # plot layer 1 'input' to layer 'hidden' - weights
    for i in range(wc1_hist.shape[2]):  # input dimension

        for j in range(wc1_hist.shape[1]):  # output dimension

            plt.figure()
            plt.plot(wc1_hist[:, j, i])
            plt.xlabel('Iterations')
            title_fig = "Layer 1 (input) control - Weight Neuron " + str(i) +\
                " to Neuron " + str(j)
            plt.title(title_fig)
            plt.grid()
            if save_figures:
                name_fig = '/weight_control_layer_1_neuron_' + str(i) + '_to_' +\
                    str(j) + '.png'
                plt.savefig(folder_results + name_fig, dpi=300)
            plt.close()

    # plot layer 1 input - bias
    for i in range(bc1_hist.shape[1]):  # input dimension

        plt.figure()
        plt.plot(bc1_hist[:, i])
        plt.xlabel('Iterations')
        title_fig = "Layer 1 (input) control - Bias Neuron " + str(i)
        plt.title(title_fig)
        plt.grid()
        if save_figures:
            name_fig = '/bias_control_layer_1_neuron_' + str(i) + '.png'
            plt.savefig(folder_results + name_fig, dpi=300)
        plt.close()


    # plot layer 2 'hidden' to layer 'hidden' - weights
    for i in range(wc2_hist.shape[2]):  # input dimension

        for j in range(wc2_hist.shape[1]):  # output dimension

            plt.figure()
            plt.plot(wc2_hist[:, j, i])
            plt.xlabel('Iterations')
            title_fig = "Layer 2 (hidden) control - Weight Neuron " + str(i) +\
                " to Neuron " + str(j)
            plt.title(title_fig)
            plt.grid()
            if save_figures:
                name_fig = '/weight_control_layer_2_neuron_' + str(i) + '_to_'\
                    + str(j) + '.png'
                plt.savefig(folder_results + name_fig, dpi=300)
            plt.close()

    # plot layer 2 'hidden' to layer 'hidden' - bias
    for i in range(bc2_hist.shape[1]):  # input dimension

        plt.figure()
        plt.plot(bc2_hist[:, i])
        plt.xlabel('Iterations')
        title_fig = "Layer 2 (hidden) control - Bias Neuron " + str(i)
        plt.title(title_fig)
        plt.grid()
        if save_figures:
            name_fig = '/bias_control_layer_2_neuron_' + str(i) + '.png'
            plt.savefig(folder_results + name_fig, dpi=300)
        plt.close()


    # plot layer 3 'hidden' to layer 'hidden' - weights
    for i in range(wc3_hist.shape[2]):  # input dimension

        for j in range(wc3_hist.shape[1]):  # output dimension

            plt.figure()
            plt.plot(wc3_hist[:, j, i])
            plt.xlabel('Iterations')
            title_fig = "Layer 3 (hidden) control - Weight Neuron " + str(i) +\
                " to Neuron " + str(j)
            plt.title(title_fig)
            plt.grid()
            if save_figures:
                name_fig = '/weight_control_layer_3_neuron_' + str(i) + '_to_'\
                    + str(j) + '.png'
                plt.savefig(folder_results + name_fig, dpi=300)
            plt.close()


    # plot layer 3 'hidden' to layer 'hidden' - bias
    for i in range(bc3_hist.shape[1]):  # input dimension

        plt.figure()
        plt.plot(bc3_hist[:, i])
        plt.xlabel('Iterations')
        title_fig = "Layer 3 (hidden) control - Bias Neuron " + str(i) 
        plt.title(title_fig)
        plt.grid()
        if save_figures:
            name_fig = '/bias_control_layer_3_neuron_' + str(i) + '.png'
            plt.savefig(folder_results + name_fig, dpi=300)
        plt.close()
        
        
    # plot layer 'hidden' to layer 'output' - weights
    for i in range(wc4_hist.shape[2]):  # input dimension

        for j in range(wc4_hist.shape[1]):  # output dimension

            plt.figure()
            plt.plot(wc4_hist[:, j, i])
            plt.xlabel('Iterations')
            title_fig = "Layer 4 (output) control - Weight Neuron " + str(i) +\
                " to Neuron " + str(j)
            plt.title(title_fig)
            plt.grid()
            if save_figures:
                name_fig = '/weight_control_layer_4_neuron_' + str(i) + '_to_'\
                    + str(j) + '.png'
                plt.savefig(folder_results + name_fig, dpi=300)
            plt.close()
        
        
        
def plot_w_generic(w_hist, layer_number, save_figures, folder_results):
    # plotting weight of a generic layer

    # plot layer 'input' to layer 'hidden'
    for i in range(w_hist.shape[2]):  # input dimension

        for j in range(w_hist.shape[1]):  # output dimension

            plt.figure()
            plt.plot(w_hist[:, j, i])
            plt.xlabel('Iterations')
            title_fig = "Layer input - Weight Neuron " + str(i) +\
                " to Neuron " + str(j)
            plt.title(title_fig)
            plt.grid()
            if save_figures:
                name_fig = '/weight_layer_' + str(layer_number) + '_neuron_' + str(i) + '_to_' +\
                    str(j) + '.png'
                plt.savefig(folder_results + name_fig, dpi=300)
            plt.close()
            
            
def plot_b_generic(b_hist, layer_number, save_figures, folder_results):
    # plotting bias of a generic layer
    
    # plot layer 'input' to layer 'hidden'

    for j in range(b_hist.shape[1]):  # output dimension

        plt.figure()
        plt.plot(b_hist[:, j])
        plt.xlabel('Iterations')
        title_fig = "Layer input - Bias Neuron " + str(j)
        plt.title(title_fig)
        plt.grid()
        if save_figures:
            name_fig = '/bias_layer_' + str(layer_number) + '_neuron_' +\
                str(j) + '.png'
            plt.savefig(folder_results + name_fig, dpi=300)
        plt.close()
