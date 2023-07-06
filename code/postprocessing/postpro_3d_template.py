#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 22:22:39 2022

A postprocessing script for the Lorenz system ANLC campaign.

@author: Davide Grande

"""

import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import functions.plot_and_save as plot_and_save
import re
import postprocessing.plot_3D as plot_3D
import postprocessing.plot_in_3D_4D_function as plot_in_3D_4D_function
import postprocessing.plot_dataset as plot_dataset

print(f"\n\nPostprocessing '{system_name}'...")


def Plot3D(X, Y, V, r):
    # Plot 3D Lyapunov functions
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot Valid region computed by dReal
    theta = np.linspace(0,2*np.pi,50)
    xc = r*np.cos(theta)
    yc = r*np.sin(theta)
    ax.plot(xc[:],yc[:],'r',linestyle='--', linewidth=2 ,label='$\mathscr{D}$')
    plt.legend(loc='upper right')

    surf = ax.plot_surface(X,Y,V, rstride=5, cstride=5, alpha=0.5, cmap=cm.winter)
    ax.contour(X,Y,V,10, zdir='z', offset=0, cmap=cm.winter)

    cb = plt.colorbar(surf, pad=0.2)

    return ax


def save_params_to_txt(params_dict, file):
    # Saving all the parameters in a dictionary to a txt file.
    
    for key, value in params_dict.items():
        if isinstance(value, dict):
            file.write(f"{key}:\n")
            save_params_to_txt(value, file)
        else:
            file.write(f"{key}: {value}\n")


# Generating figures result folder
try:
    folder_results_plots = final_dir_run + "/figures"
    current_dir = os.getcwd()
    final_dir_plots = current_dir + "/" + folder_results_plots + "/"
    os.mkdir(final_dir_plots)
except OSError:
    print("Creation of the plot result directory %s failed" % final_dir_plots)
else:
    print("Plot result directory successfully created as: \n %s \n" % final_dir_plots)

if parameters['plot_ctr_weights']:  # TODO include this in a separate script
    print("Plotting ANN weights...")

    # 3) Plotting ANN weights
    # Generating control ANN weight and bias result folder
    try:
        folder_ann_plots = folder_results_plots + "/ANN_control"
        current_dir = os.getcwd()
        final_dir_plots_ann_control = current_dir + "/" + folder_ann_plots + "/"
        os.mkdir(final_dir_plots_ann_control)
    except OSError:
        print("Creation of the ANN plot result directory %s failed" % final_dir_plots_ann_control)
    else:
        print("Plot ANN result directory successfully created as: \n %s \n" % final_dir_plots_ann_control)


    if parameters['use_lin_ctr']:
        for i_p in range(q_hist.shape[1]):
          for j_p in range(q_hist.shape[2]):
              plt.figure()
              plt.plot(q_hist[:, i_p, j_p])
              plt.xlabel('Epochs')
              plt.grid()
              name_fig = '/weight_control_layer_1_neuron_' + str(i_p) + '_to_' +\
                  str(j_p) + '.png'
              plt.savefig(final_dir_plots_ann_control + name_fig, dpi=parameters['dpi_'])
              plt.close()

    else:

        for i_p in range(wc1_hist.shape[1]):
            for j_p in range(wc1_hist.shape[2]):
                plt.figure()
                plt.plot(wc1_hist[:, i_p, j_p])
                plt.xlabel('Epochs')
                plt.grid()
                name_fig = '/weight_control_layer_1_neuron_' + str(i_p) + '_to_' +\
                    str(j_p) + '.png'
                plt.savefig(final_dir_plots_ann_control + name_fig, dpi=parameters['dpi_'])
                plt.close()

        for i_p in range(wc2_hist.shape[1]):
            for j_p in range(wc2_hist.shape[2]):
                plt.figure()
                plt.plot(wc2_hist[:, i_p, j_p])
                plt.xlabel('Epochs')
                plt.grid()
                name_fig = '/weight_control_layer_2_neuron_' + str(i_p) + '_to_' +\
                    str(j_p) + '.png'
                plt.savefig(final_dir_plots_ann_control + name_fig, dpi=parameters['dpi_'])
                plt.close()

        for i_p in range(wc3_hist.shape[1]):
            for j_p in range(wc3_hist.shape[2]):
                plt.figure()
                plt.plot(wc3_hist[:, i_p, j_p])
                plt.xlabel('Epochs')
                plt.grid()
                name_fig = '/weight_control_layer_3_neuron_' + str(i_p) + '_to_' +\
                    str(j_p) + '.png'
                plt.savefig(final_dir_plots_ann_control + name_fig, dpi=parameters['dpi_'])
                plt.close()
       
        if parameters['contr_bias1']:
            for i_p in range(bc1_hist.shape[1]):
                for j_p in range(bc1_hist.shape[2]):
                    plt.figure()
                    plt.plot(bc1_hist[:, i_p, j_p])
                    plt.xlabel('Epochs')
                    plt.grid()
                    name_fig = '/bias_control_layer_3_neuron_' + str(i_p) + '_to_' +\
                        str(j_p) + '.png'
                    plt.savefig(final_dir_plots_ann_control + name_fig, dpi=parameters['dpi_'])
                    plt.close()

        if parameters['contr_bias2']:
            for i_p in range(bc2_hist.shape[1]):
                for j_p in range(bc2_hist.shape[2]):
                    plt.figure()
                    plt.plot(bc2_hist[:, i_p, j_p])
                    plt.xlabel('Epochs')
                    plt.grid()
                    name_fig = '/bias_control_layer_2_neuron_' + str(i_p) + '_to_' +\
                        str(j_p) + '.png'
                    plt.savefig(final_dir_plots_ann_control + name_fig, dpi=parameters['dpi_'])
                    plt.close()

        if parameters['contr_bias3']:
            for i_p in range(bc3_hist.shape[1]):
                for j_p in range(bc3_hist.shape[2]):
                    plt.figure()
                    plt.plot(bc3_hist[:, i_p, j_p])
                    plt.xlabel('Epochs')
                    plt.grid()
                    name_fig = '/bias_control_layer_3_neuron_' + str(i_p) + '_to_' +\
                        str(j_p) + '.png'
                    plt.savefig(final_dir_plots_ann_control + name_fig, dpi=parameters['dpi_'])
                    plt.close()


# Plot dataset distribution
x_final =  x.clone()  # copying final dataset   

if parameters['plot_V_weights']:

    # 3) Plotting Lyapunov ANN weights
    # Generating Lyapunov ANN weight and bias result folder
    try:
        folder_ann_lyap_plots = folder_results_plots + "/ANN_lyap"
        current_dir = os.getcwd()
        final_dir_plots_ann_lyap = current_dir + "/" + folder_ann_lyap_plots + "/"
        os.mkdir(final_dir_plots_ann_lyap)
    except OSError:
        print("Creation of the Lyap. ANN plot result directory %s failed" % final_dir_plots_ann_lyap)
    else:
        print("Plot Lyap. ANN result directory successfully created as: \n %s \n" % final_dir_plots_ann_lyap)


    for i_p in range(w1_hist.shape[1]):
      for j_p in range(w1_hist.shape[2]):
          plt.figure()
          plt.plot(w1_hist[:, i_p, j_p])
          plt.xlabel('Epochs')
          plt.grid()
          name_fig = '/weight_lyap_layer_1_neuron_' + str(i_p) + '_to_' +\
              str(j_p) + '.png'
          plt.savefig(final_dir_plots_ann_lyap + name_fig, dpi=parameters['dpi_'])
          plt.close()

    # Plotting only first layer as it only serves as a flag for possible
    # training stalls

# Saving weights
# Generating ANN result folder
try:
    folder_results_ann = folder_results_plots + "/ANN_params/"
    current_dir = os.getcwd()
    final_dir_ann = current_dir + "/" + folder_results_ann
    os.mkdir(folder_results_ann)

except OSError:
    print(f"\nCreation of the ANN params directory: \n{folder_results_ann} \nFAILED!!\n")
else:
    print(f"\nANN params directory SUCCESSFULLY created as: \n{folder_results_ann}\n")

np.savetxt(folder_results_ann + "w1.txt", model.layer1.weight.data, fmt="%s")
np.savetxt(folder_results_ann + "w2.txt", model.layer2.weight.data, fmt="%s")
np.savetxt(folder_results_ann + "w3.txt", model.layer3.weight.data, fmt="%s")
if parameters['Lyap_bias1']:
    np.savetxt(folder_results_ann + "b1.txt", model.layer1.bias.data, fmt="%s")
if parameters['Lyap_bias2']:
    np.savetxt(folder_results_ann + "b2.txt", model.layer2.bias.data, fmt="%s")
np.savetxt(folder_results_ann + "wc1.txt", model.control1.weight.data, fmt="%s")
np.savetxt(folder_results_ann + "wc2.txt", model.control2.weight.data, fmt="%s")
np.savetxt(folder_results_ann + "wc3.txt", model.control3.weight.data, fmt="%s")
np.savetxt(folder_results_ann + "q.txt", model.control.weight.data, fmt="%s")
if parameters['contr_bias1']:
    np.savetxt(folder_results_ann + "bc1.txt", model.control1.bias.data, fmt="%s")
if parameters['contr_bias2']:
    np.savetxt(folder_results_ann + "bc2.txt", model.control2.bias.data, fmt="%s")


# Plot loss function
try:
    folder_loss = folder_results_plots + "/loss_function/"
    current_dir = os.getcwd()
    final_dir_loss = current_dir + "/" + folder_loss
    os.mkdir(final_dir_loss)
except OSError:
    print("Creation of the 'Loss function result' directory %s failed" % final_dir_loss)
else:
    print("'Loss function result' directory successfully created as: \n %s \n" % final_dir_loss)


plot_and_save.plot_and_save(data=Lyap_risk_ELR_hist,
                            save_figures=True,
                            save_fig_folder=final_dir_loss,
                            save_fig_file="lyapunov_risk_ELR.png",
                            title=None, 
                            xlabel='Epochs',
                            ylabel=None,
                            caption=None)

plot_and_save.plot_and_save(data=Lyap_risk_SLR_hist,
                            save_figures=True,
                            save_fig_folder=final_dir_loss,
                            save_fig_file="lyapunov_risk_SLR.png",
                            title=None, 
                            xlabel='Epochs',
                            ylabel=None,
                            caption=None)


# Plot loss functions comparison
plt.figure()
plt.plot(Lyap_risk_ELR_hist, label='$L_{ELR}$')
plt.plot(Lyap_risk_SLR_hist, label='$L_{SLR}$')
plt.xlabel('Epochs')
plt.ylabel(None)
save_fig_file="Lyapunov_risk_comparison.png"
save_fig_folder=final_dir_loss
plt.grid()
plt.legend(loc='best')
plt.savefig(save_fig_folder + save_fig_file, dpi=parameters['dpi_'])
plt.close()



# Plotting learning rate
plot_and_save.plot_and_save(data=learning_rate_history,
                            save_figures=True,
                            save_fig_folder=folder_results_plots + "/",
                            save_fig_file="learning_rate.png",
                            title=None, 
                            xlabel='Epochs',
                            ylabel=None,
                            caption=None)

plot_and_save.plot_and_save(data=learning_rate_c_history,
                            save_figures=True,
                            save_fig_folder=folder_results_plots + "/",
                            save_fig_file="learning_rate_c.png",
                            title=None, xlabel='Epochs',
                            ylabel=None,
                            caption=None)
    
# Plotting Lyapuonov function
if parameters['plot_V']:

    # Generating result folder
    try:
        folder_V = folder_results_plots + "/V_function/"
        current_dir = os.getcwd()
        final_dir_V = current_dir + "/" + folder_V
        os.mkdir(final_dir_V)
    except OSError:
        print("Creation of the 'V-function result' directory %s failed" % final_dir_V)
    else:
        print("'V-function result' directory successfully created as: \n %s \n" % final_dir_V)
    
    if (parameters['n_input'] == 3):
        
        iteration_no = '_final'
        title = 'Lyapunov_function_'
        plot_in_3D_4D_function.plot(V_learn, 100, 
                                    gamma_overbar, False, False,
                                    title, iteration_no, final_dir_V, 
                                    parameters['save_pdf'], parameters['dpi_'])
     
        
        if parameters['compare_first_last_iters']:
            iteration_no = '_init'
            title = 'Lyapunov_function_'
            plot_in_3D_4D_function.plot(V_learn0, 100, 
                                        gamma_overbar, False, False,
                                        title, iteration_no, final_dir_V, 
                                        parameters['save_pdf'], parameters['dpi_'])
    
    
    if parameters['compare_first_last_iters']:
        print("Plotting comparison V, Vdot first and last iterations")
        
        # Generating result folder
        try:
            folder_V_Vdot = folder_results_plots + "/V_Vdot_function/"
            current_dir = os.getcwd()
            final_dir_V_Vdot = current_dir + "/" + folder_V_Vdot
            os.mkdir(final_dir_V_Vdot)
        except OSError:
            print("Creation of the 'V/Vdot-function result' directory %s failed" % folder_V_Vdot)
        else:
            print("'V/Vdot-function result' directory successfully created as: \n %s \n" % folder_V_Vdot)
            

# Plotting Lie derivative
if parameters['plot_Vdot']:
   
    # Generating result folder
    try:
        folder_V_lie = folder_results_plots + "/V_lie_function/"
        current_dir = os.getcwd()
        final_dir_V_lie = current_dir + "/" + folder_V_lie
        os.mkdir(final_dir_V_lie)
    except OSError:
        print("Creation of the 'V-lie function result' directory %s failed" % final_dir_V_lie)
    else:
        print("'V-lie function result' directory successfully created as: \n %s \n" % final_dir_V_lie)
   
    # evaluate Lyapunov function expression
    V_lie_str = lie_derivative_of_V.to_string() 
    V_lie_sub = from_dreal_to_np.sub(V_lie_str)  # substitute dreal functions
    
    
    if (parameters['n_input'] == 3):

        iteration_no = '_final'
        title = 'Lie_derivative_'
        plot_in_3D_4D_function.plot(lie_derivative_of_V, 100, 
                                    gamma_overbar, False, True,
                                    title, iteration_no, final_dir_V_lie, 
                                    parameters['save_pdf'], parameters['dpi_'])

        if parameters['compare_first_last_iters']:
            iteration_no = '_init'
            title = 'Lie_derivative_'
            plot_in_3D_4D_function.plot(lie_derivative_of_V0, 100, 
                                        gamma_overbar, False, True,
                                        title, iteration_no, final_dir_V_lie, 
                                        parameters['save_pdf'], parameters['dpi_'])

# Plotting control function
if parameters['plot_u']:

    print("Plotting control function ... ")    

    # Generating result folder
    try:
        folder_u = folder_results_plots + "/control_function/"
        current_dir = os.getcwd()
        final_dir_u = current_dir + "/" + folder_u
        os.mkdir(final_dir_u)
    except OSError:
        print("Creation of the 'Control function result' directory %s failed" % final_dir_u)
    else:
        print("'Control function result' directory successfully created as: \n %s \n" % final_dir_u)

    ctr_input = np.zeros((parameters['contr_out'], 1))
    for iC in range(parameters['contr_out']):
        # retrieving control variables
        str_u = "u_NN" + str(iC)
        
        plot_3D.plot(locals()[str_u], parameters['n_points_3D'], gamma_overbar, 
                     f'Control function {iC}', False,
                     final_dir_u, parameters['dpi_'], Plot3D) 
            

# 0) Save ANN 
ann_file = final_dir_run + "/trained_ann"
ann_file_h5 = final_dir_run + "/trained_ann.h5" 
ann_file_onnx = final_dir_run+"/trained_ann.onnx"
ann_file_onnx_no_names = final_dir_run +"/trained_ann_no_names.onnx"
ann_file_tf = final_dir_run +"/trained_ann.pb"

input_names = [ "x_%d" % i for i in range(1,4) ]
output_names = [ "V", "u"]
input_onnx = x_0  # arbitrary choice

torch.save(model.state_dict(), ann_file)
torch.save(model, ann_file_h5)

# Plot dataset with CEs
plot_dataset.plot_order_3_v2(parameters['n_input'], folder_results_plots, 
                             ce_found, x_dataset_init, x_final,
                             parameters['dpi_'], -gamma_overbar, gamma_overbar)

# Plotting Discrete Falsifier number of violations
plt.figure()
plt.plot(disc_viol_found, marker='x')
plt.xlabel("Iterations")
plt.ylabel(None)
plt.grid()
name_fig = "DF_violations"
plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=parameters['dpi_'])
plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
plt.close()


# Automatic log report
saving_log.gen_log(system_name, found_lyap_f, seed_, max_loop_number, max_iters, 
                   parameters, x, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, 
                   alpha_roa, epsilon, gamma_underbar, gamma_overbar, config, 
                   model, to_fals, to_learner, seconds_elapsed, minutes_elapsed, 
                   hours_elapsed, out_iters, i_epoch, start, init_date, 
                   end_date, falsifier_elapsed, final_dir_run, x_star)

# Dump all the paramters of the dictionary
with open(final_dir_run + "/params_raw.txt", "w") as file:
    save_params_to_txt(parameters, file)

# Saving ANN symbolic expressions
try:
    folder_expressions = final_dir_run + "/expressions/"
    current_dir = os.getcwd()
    final_dir_expre = current_dir + "/" + folder_expressions
    os.mkdir(final_dir_expre)
except OSError:
    print("Creation of the 'Expressions' directory %s failed" % final_dir_expre)
else:
    print("'Expressions' directory successfully created as: \n %s \n" % final_dir_expre)

np.savetxt(final_dir_expre + "Expression_V.txt", [V_learn.to_string()], fmt="%s")
np.savetxt(final_dir_expre + "Expression_Vdot.txt", [lie_derivative_of_V.to_string()], fmt="%s")
ctr_input = np.zeros((parameters['contr_out'], 1))
for iC in range(parameters['contr_out']):
    # retrieving control variables
    str_x_i = "u_NN" + str(iC)
    np.savetxt(final_dir_expre + "Expression_u" + str(iC) + ".txt", [locals()[str_x_i].to_string()], fmt="%s")

