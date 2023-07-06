#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 23:58:44 2022

@author: Davide Grande

Postprocessing file for the ANLC of an inverted pendulum. 

"""

#from numpy import *
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import functions.plot_and_save as plot_and_save
import functions.plot_weights as plot_weights
import re

print("\n\nPostprocessing 'Control inverted pendulum 2DOF'...")


def Plot3D(X, Y, V, r):
    # Plot Lyapunov functions
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot Valid region computed by dReal
    theta = np.linspace(0,2*np.pi,50)
    xc = r*np.cos(theta)
    yc = r*np.sin(theta)
    ax.plot(xc[:],yc[:],'r',linestyle='--', linewidth=2 ,label=r'domain ($\mathscr{D}$)')
    plt.legend(loc='upper right')

    surf = ax.plot_surface(X,Y,V, rstride=5, cstride=5, alpha=0.5, cmap=cm.winter)
    ax.contour(X,Y,V,10, zdir='z', offset=0, cmap=cm.winter)

    cb = plt.colorbar(surf, pad=0.2)

    return ax


def Plotflow(Xd, Yd, t, dyn_sys_params, close_loop):
    # Plot phase plane 
    DX, DY = f([Xd, Yd], t, dyn_sys_params, close_loop)
    DX=DX/np.linalg.norm(DX, ord=2, axis=1, keepdims=True)
    DY=DY/np.linalg.norm(DY, ord=2, axis=1, keepdims=True)
    plt.streamplot(Xd,Yd,DX,DY, color=('gray'), linewidth=0.5,
                  density=0.5, arrowstyle='-|>', arrowsize=1.5)


def f(y,t, dyn_sys_params, close_loop) :
    # parameters
    G = dyn_sys_params.G
    L = dyn_sys_params.L
    m = dyn_sys_params.m
    b = dyn_sys_params.b

    x1, x2 = y    
    
    u_NN_str = u_NN.to_string() 
    u_NN_sub = from_dreal_to_np.sub(u_NN_str)  # substitute dreal functions
    u_NN_eval = eval(u_NN_sub)
    
    if close_loop:
        
        #u_NN_eval = (+0*x1 +0*x2)
        
        dydt =[x2, 
               (m*G*L*np.sin(x1) - b*x1 + u_NN_eval) / (m*L**2)]

    else:  
        # open loop 
        dydt =[x2, 
               (m*G*L*np.sin(x1) - b*x1) / (m*L**2)]

    return dydt


# 1 Plotting ANN weights
if plot_ctr_weights:
    print("Plotting ANN weights...")
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

    if use_lin_ctr:
        for i_p in range(q_hist.shape[1]):  # input dimension
          for j_p in range(q_hist.shape[2]):  # input dimension
              plt.figure()
              plt.plot(q_hist[:, i_p, j_p])
              plt.xlabel('Iterations')
              plt.grid()
              #if save_figures:
              plt.title("Control weight")
              name_fig = '/weight_control_layer_1_neuron_' + str(i_p) + '_to_' +\
                  str(j_p) + '.png'
              plt.savefig(final_dir_plots_ann_control + name_fig, dpi=dpi_)
              plt.close()

    else:

        for i_p in range(wc1_hist.shape[1]):  # input dimension
            for j_p in range(wc1_hist.shape[2]):  # input dimension
                plt.figure()
                plt.plot(wc1_hist[:, i_p, j_p])
                plt.xlabel('Iterations')
                #title_fig = "Layer 1 (input) control - Weight Neuron " + str(i) +\
                #    " to Neuron " + str(j)
                #plt.title(title_fig)
                plt.grid()
                #if save_figures:
                name_fig = '/weight_control_layer_1_neuron_' + str(i_p) + '_to_' +\
                    str(j_p) + '.png'
                plt.savefig(final_dir_plots_ann_control + name_fig, dpi=dpi_)
                plt.close()

        for i_p in range(wc2_hist.shape[1]):  # input dimension
            for j_p in range(wc2_hist.shape[2]):  # input dimension
                plt.figure()
                plt.plot(wc2_hist[:, i_p, j_p])
                plt.xlabel('Iterations')
                plt.grid()
                #if save_figures:
                name_fig = '/weight_control_layer_2_neuron_' + str(i_p) + '_to_' +\
                    str(j_p) + '.png'
                plt.savefig(final_dir_plots_ann_control + name_fig, dpi=dpi_)
                plt.close()

        for i_p in range(wc3_hist.shape[1]):  # input dimension
            for j_p in range(wc3_hist.shape[2]):  # input dimension
                plt.figure()
                plt.plot(wc3_hist[:, i_p, j_p])
                plt.xlabel('Iterations')
                plt.grid()
                #if save_figures:
                name_fig = '/weight_control_layer_3_neuron_' + str(i_p) + '_to_' +\
                    str(j_p) + '.png'
                plt.savefig(final_dir_plots_ann_control + name_fig, dpi=dpi_)
                plt.close()

if plot_lyap_weights:

    print("Plotting Lyapunov weights...")
    # Generating control ANN weight and bias result folder
    try:
        folder_ann_plots_lyap = folder_results_plots + "/ANN_lyap"
        current_dir = os.getcwd()
        final_dir_plots_ann_lyap = current_dir + "/" + folder_ann_plots_lyap + "/"
        os.mkdir(final_dir_plots_ann_lyap)
    except OSError:
        print("Creation of the Lyap. ANN plot result directory %s failed" % final_dir_plots_ann_lyap)
    else:
        print("Plot Lyap. ANN result directory successfully created as: \n %s \n" % final_dir_plots_ann_lyap)

    plot_weights.plot_w_generic(w1_hist, 1, True, final_dir_plots_ann_lyap)
    plot_weights.plot_w_generic(w2_hist, 2, True, final_dir_plots_ann_lyap)
    plot_weights.plot_w_generic(w3_hist, 3, True, final_dir_plots_ann_lyap)


# 2) Save weights
# Generating ANN result folder
try:
    from pathlib import Path
    folder_results_ann = folder_results + "/ANN_params/"
    current_dir = os.getcwd()
    final_dir_ann = current_dir + "/" + folder_results_ann
    os.mkdir(folder_results_ann)

except OSError:
    print(f"\nCreation of the ANN result directory: \n{folder_results_ann} \nFAILED!!\n")
else:
    print(f"\nANN result directory SUCCESSFULLY created as: \n{folder_results_ann}\n")

np.savetxt(folder_results_ann + "w1.txt", model.layer1.weight.data, fmt="%s")
np.savetxt(folder_results_ann + "w2.txt", model.layer2.weight.data, fmt="%s")
np.savetxt(folder_results_ann + "w3.txt", model.layer3.weight.data, fmt="%s")
if Lyap_bias1:
    np.savetxt(folder_results_ann + "b1.txt", model.layer1.bias.data, fmt="%s")
if Lyap_bias2:
    np.savetxt(folder_results_ann + "b2.txt", model.layer2.bias.data, fmt="%s")
if Lyap_bias3:
    np.savetxt(folder_results_ann + "b3.txt", model.layer3.bias.data, fmt="%s")
np.savetxt(folder_results_ann + "wc1.txt", model.control1.weight.data, fmt="%s")
np.savetxt(folder_results_ann + "wc2.txt", model.control2.weight.data, fmt="%s")
np.savetxt(folder_results_ann + "wc3.txt", model.control3.weight.data, fmt="%s")
np.savetxt(folder_results_ann + "q.txt", model.control.weight.data, fmt="%s")
if contr_bias1:
    np.savetxt(folder_results_ann + "bc1.txt", model.control1.bias.data, fmt="%s")
if contr_bias2:
    np.savetxt(folder_results_ann + "bc2.txt", model.control2.bias.data, fmt="%s")

# 3) Plot Lyapunov risk
    
# 0) Generating result folder
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
                            xlabel='Iterations',
                            ylabel=None,
                            caption=None)

plot_and_save.plot_and_save(data=Lyap_risk_SLR_hist,
                            save_figures=True,
                            save_fig_folder=final_dir_loss,
                            save_fig_file="lyapunov_risk_SLR.png",
                            title=None, 
                            xlabel='Iterations',
                            ylabel=None,
                            caption=None)

plot_and_save.plot_and_save(data=V_hist,
                            save_figures=True,
                            save_fig_folder=final_dir_loss,
                            save_fig_file="V_contribution_history.png",
                            title=None, 
                            xlabel='Iterations',
                            ylabel=None,
                            caption=None)

plot_and_save.plot_and_save(data=V_dot_hist,
                            save_figures=True,
                            save_fig_folder=final_dir_loss,
                            save_fig_file="Vdot_contribution_history.png",
                            title=None, 
                            xlabel='Iterations',
                            ylabel=None,
                            caption=None)

plot_and_save.plot_and_save(data=V0_hist,
                            save_figures=True,
                            save_fig_folder=final_dir_loss,
                            save_fig_file="V0_contribution_history.png",
                            title=None, 
                            xlabel='Iterations',
                            ylabel=None,
                            caption=None)

plot_and_save.plot_and_save(data=V_tuning_hist,
                            save_figures=True,
                            save_fig_folder=final_dir_loss,
                            save_fig_file="V_tuning_contribution_history.png",
                            title=None, 
                            xlabel='Iterations',
                            ylabel=None,
                            caption=None)



# Plot contribution comparison
plt.figure()
plt.plot(V_hist, label='$V$-term')
plt.plot(V_dot_hist, label='$\dot{V}$-term')
plt.plot(V0_hist, label='${V(0)}$-term')  
plt.plot(V_tuning_hist, label='${V_{tune}}$-term')  
plt.xlabel('Iterations')
plt.ylabel(None)
plt.title("Loss function contributions")
save_fig_file="Loss_function_contributions.png"
save_fig_folder=final_dir_loss
plt.grid()
plt.legend(loc='best')
plt.savefig(save_fig_folder + save_fig_file, dpi=dpi_)
#plt.close()

# Plot contribution comparison
plt.figure()
plt.plot(Lyap_risk_ELR_hist, label='$L_{ELR}$')
plt.plot(Lyap_risk_SLR_hist, label='$L_{SLR}$')
plt.xlabel('Iterations')
plt.ylabel(None)
#plt.title("Loss function contributions")
save_fig_file="Lyapunov_risk_comparison.png"
save_fig_folder=final_dir_loss
plt.grid()
plt.legend(loc='best')
plt.savefig(save_fig_folder + save_fig_file, dpi=dpi_)
#plt.close()    


# 4) Plotting learning rate
plot_and_save.plot_and_save(data=learning_rate_history,
                            save_figures=True,
                            save_fig_folder=folder_results_plots + "/",
                            save_fig_file="learning_rate.png",
                            title=None, 
                            xlabel='Iterations',
                            ylabel=None,
                            caption=None)

plot_and_save.plot_and_save(data=learning_rate_c_history,
                            save_figures=True,
                            save_fig_folder=folder_results_plots + "/",
                            save_fig_file="learning_rate_c.png",
                            title=None, xlabel='Iterations',
                            ylabel=None,
                            caption=None)

# plot initial dataset
plt.figure()
plt.scatter(x_dataset_init[:,0], x_dataset_init[:,1], c='limegreen', s=15) 
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid()
plt.xlim(-gamma_overbar, gamma_overbar)
plt.ylim(-gamma_overbar, gamma_overbar)
name_fig = "dataset_init"
plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
plt.close()

# plot dataset CE
plt.figure()
plt.scatter(ce_found[:,0], ce_found[:,1], c='r', s=15) 
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid()
plt.xlim(-gamma_overbar, gamma_overbar)
plt.ylim(-gamma_overbar, gamma_overbar)
name_fig = "dataset_CE"
plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
plt.close()

# plot final dataset
plt.figure()
plt.scatter(x[:,0], x[:,1], s=15) 
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid()
name_fig = "dataset_final"
plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
plt.close()

# plot final dataset (sum of CE and initial)
plt.figure()
ax = plt.subplot(111)
plt.scatter(ce_found[:,0], ce_found[:,1], s=15, label='$CE_{SMT}$')  
plt.scatter(x[:,0], x[:,1], s=15, label='$CE_{DF}$') 
plt.scatter(x_dataset_init[:,0], x_dataset_init[:,1], c='limegreen', s=15, label='Intial points') 
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid()
plt.legend(bbox_to_anchor=(-0.02, 1.02), loc="lower left", ncol=3)
plt.xlim(-gamma_overbar, gamma_overbar)
plt.ylim(-gamma_overbar, gamma_overbar)
name_fig = "dataset_init_plus_CEs"
plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
plt.close()


# plot final dataset (sum of CE and initial) inverted order
plt.figure()
ax = plt.subplot(111)
plt.scatter(x_dataset_init[:,0], x_dataset_init[:,1], c='limegreen', s=15, label='Intial points') 
plt.scatter(x[:,0], x[:,1], s=15, label='Final points') 
plt.scatter(ce_found[:,0], ce_found[:,1], s=15, label='CEs')  
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid()
plt.legend(bbox_to_anchor=(-0.02, 1.02), loc="lower left", ncol=3)
plt.xlim(-gamma_overbar, gamma_overbar)
plt.ylim(-gamma_overbar, gamma_overbar)
name_fig = "dataset_init_plus_CEs_other_order"
plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
plt.close()


# 5) Plotting Lyapuonov function
if plot_V:
    print("\nPlotting V ...\n")
    X = np.linspace(-gamma_overbar, gamma_overbar, 100)
    Y = np.linspace(-gamma_overbar, gamma_overbar, 100)
    x1, x2 = np.meshgrid(X, Y)

    V_str = V_learn.to_string()
    V_sub = from_dreal_to_np.sub(V_str)  # substitute dreal functions
    V_eval = eval(V_sub)
     
   
    # birdeye view (x1, x2)
    ax = Plot3D(x1, x2, V_eval, gamma_overbar)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('V')
    #plt.title('Lyapunov Function')
    name_fig = 'Lyapunov_function'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.savefig(folder_results_plots + '/' +name_fig+".eps", format='eps')
    plt.close()


    # side view (x1, x2)
    ax = Plot3D(x1, x2, V_eval, gamma_overbar)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('V')
    ax.view_init(elev=0, azim=90)
    #plt.title('Lyapunov Function')
    name_fig = 'Lyapunov_function_side'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()

    V_min = np.min(V_eval)
    V_max = np.max(V_eval)

# 6) Plotting Lie derivative
if plot_Vdot:

    print("\nPlotting Lie derivative...\n")
    
    X = np.linspace(-gamma_overbar, gamma_overbar, 100)
    Y = np.linspace(-gamma_overbar, gamma_overbar, 100)
    x1, x2 = np.meshgrid(X, Y)
    V_lie_str = lie_derivative_of_V.to_string() 
    V_lie_sub = from_dreal_to_np.sub(V_lie_str)  # substitute dreal functions
    V_lie_eval = eval(V_lie_sub)


    # birdeye view (x1, x2)
    ax = Plot3D(x1, x2, V_lie_eval, gamma_overbar)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel(''r'$\nabla V_x \cdot f$')
    name_fig = 'Lie_derivative_x1_x2'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()


    # side view (x1, x2)
    ax = Plot3D(x1, x2, V_lie_eval, gamma_overbar)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel(''r'$\nabla V_x \cdot f$')
    #plt.title('Lyapunov Function')
    ax.view_init(elev=0, azim=90)
    name_fig = 'Lie_derivative_x1_x2_side'
    plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
    plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
    plt.close()

    Lie_V_max = np.max(V_lie_eval)



# 7) Saving ANN file
ann_file = final_dir + "trained_ann"
ann_file_h5 = final_dir + "trained_ann.h5" 
torch.save(model.state_dict(), ann_file)
torch.save(model, ann_file_h5)



# 10) Plot CE history
plt.figure()
plt.plot(ce_found[:,0], ce_found[:,1], marker='x') 
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
#plt.title('Counter examples')
plt.grid()
name_fig = "counter_examples"
plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
plt.close()

# plotting Euclidean distance of the CEs over time
euc_dist_ce = np.zeros(ce_found[:,0].size)
domain_limit = np.zeros(ce_found[:,0].size)
for iCE in range(len(ce_found)):
    euc_dist_ce[iCE] = np.linalg.norm(ce_found[iCE,:])
    domain_limit[iCE] = gamma_overbar

plt.figure()
plt.plot(euc_dist_ce, marker='x', label='E.D. CE')
plt.plot(domain_limit, 'r', linestyle='--', linewidth=2, label='Valid region')
plt.xlabel("Iterations")
plt.ylabel(None)
plt.legend(loc="best")
plt.grid()
name_fig = "counter_examples_euclidean_distance"
plt.savefig(folder_results_plots + '/' +name_fig+".png", dpi=dpi_)
plt.savefig(folder_results_plots + '/' +name_fig+".pdf", format='pdf')
plt.close()


# 14) Saving symbolic expressions
np.savetxt(final_dir + "Expression_V.txt", [V_learn.to_string()], fmt="%s")
np.savetxt(final_dir + "Expression_Vdot.txt", [lie_derivative_of_V.to_string()], fmt="%s")
np.savetxt(final_dir + "Expression_u.txt", [u_NN.to_string()], fmt="%s")


# 15) Automatic report
print("Saving logs ...\n")

dt_string_init = init_date.strftime("%d/%m/%Y %H:%M:%S")  # date init training
dt_string_end = end_date.strftime("%d/%m/%Y %H:%M:%S")  # date end training

result_report = [f"Run of the {system_name} system.\n" + 
                 "Convergence reached = " + str(found_lyap_f) + "\n\n" +
                 "TRAINING PARAMS: \n" + 
                 "Seed = " + str(seed_) + "\n" +
                 "max_loop_number = " + str(max_loop_number) + "\n" +
                 "max_epochs_per_loop = " + str(max_iters_per_loop) + "\n" +
                 "Initial dataset dimension = " + str(N) + "\n" +
                 "Final dataset dimension = " + str(len(x)) + "\n" + 
                 "Using a sliding window = " + str(sliding_window) + "\n" +
                 "Maximum dataset dimension (if using sliding wind) = " + str(N_max) + "\n" +
                 "Re-initialise dataset = " + str(use_reinit_dataset) + "\n" +
                 "\n\n" +
                 "LYAPUNOV ANN: \n" + 
                 "layer 1 dim. = " + str(H1) + "\n" +
                 "layer 1 act. f. = " + str(Lyap_act_fun1) + "\n" +
                 "layer 1 has bias = " + str(Lyap_bias1) + "\n" +
                 "layer 2 dim. = " + str(H2) + "\n" +
                 "layer 2 has act. f. = " + str(Lyap_act_fun2) + "\n" +
                 "layer 2 has bias = " + str(Lyap_bias2) + "\n" +
                 "layer 3 dim. = " + str(D_out) + "\n" +
                 "layer 3 has act. f. = " + str(Lyap_act_fun3) + "\n" +
                 "layer 3 has bias = " + str(Lyap_bias3) + "\n" +
                 "\n\n" +
                 "CONTROL ANN: \n" +
                 "Use linear control = " + str(use_lin_ctr) + "\n" + 
                 "If nonlinear control law is used, then:\n" + 
                 "dim. layer 1 = " +str(contr_hid1) + "\n" +
                 "dim layer 2 = " +str(contr_hid2) + "\n" +
                 "use bias layer 1 = " + str(contr_bias1) + "\n" +
                 "use bias layer 2 = " + str(contr_bias2) + "\n" +
                 "\n\n" +
                 "LEARNER: \n" + 
                 "Learning rate Lyap. = " + str(learning_rate) + "\n" +
                 "Learning rate control = " + str(learning_rate_c) + "\n" + 
                 "\nLYAPUNOV RISK:\n" +
                 "alpha_1 (Weight V) = " + str(alpha_1) + "\n" +
                 "alpha_2 (Weight V_dot) = " + str(alpha_2) + "\n" +
                 "alpha_3 (Weight V0) = " + str(alpha_3) + "\n" +
                 "alpha_4 (Weight V tuning) = " + str(alpha_4) + "\n" +
                 "alpha_5(overall weight) = " + str(alpha_5) + "\n" +
                 "\n\n" +
                 "TRAINING: \n" +
                 "ANN control initialised (LQR) = " + str(use_old_solution) + "\n" +
                 "ANN control initial weights = " + str(init_control) + "\n" +
                 "\n\n" +
                 "FALSIFIER (SMT): \n" +  
                 "Epsilon = " + str(epsilon) + "\n" +
                 "Falsifier domain = " + str(gamma_underbar) + " - " +str(gamma_overbar) + "\n" +
                 "config.precision = " + str(config.precision) + "\n"
                 "zeta_SMT (SMT CE point cloud) = " + str(zeta_SMT) +
                 "\n\n" + 
                 "DISCRETE FALSIFIER (DF): \n" +  
                 f"use adaptive grid = {adaptive_grid}\n" +
                 f"maximum grid points = {max_grid_points}\n" +
                 f"Final grid points = {grid_points}\n" +
                 "\n\n" +                  
                 "POSTPROCESSING: \n" + 
                 "Testing close-loop system = " + str(test_closed_loop_dynamics) + "\n" +
                 "Dt = " + str(Dt) + "\n" +
                 "end_time = " + str(end_time) +
                 "\n\n" +
                 "RESULTS: \n" +
                 "Time elapsed: "+ "\n" +
                 "seconds = " + str(seconds_elapsed) + "\n" +
                 "minutes = " + str(minutes_elapsed) + "\n" +
                 "hours = " + str(hours_elapsed)+ "\n" +
                 "Falsifier time [']: " + str(falsifier_elapsed) + "\n" +
                 f"Falsifier time [%]: {falsifier_elapsed/minutes_elapsed*100}\n" +
                 f"Training iterations completed = {(out_iters)}\n" +
                 f"Training epochs (last iteration) = {i_epoch}\n\n" + 
                 f"Training start = {dt_string_init}\n" +
                 f"Training end = {dt_string_end}"                 
                 ]

np.savetxt(final_dir + "logs.txt", result_report, fmt="%s")


# 16) Dynamic parameters
dynamic_param_save = []
for att in vars(dyn_sys_params):
    dynamic_param_save.append([att, getattr(dyn_sys_params,att)])
np.savetxt(final_dir + "dyn_system_params.txt", dynamic_param_save, fmt="%s")

