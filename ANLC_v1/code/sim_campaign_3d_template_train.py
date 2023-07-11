#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:49:27 2022

@author: Davide Grande


This script aims at designing an ANLC for a generic 3-dimensional system.

"""


##############################################################################

#System Dynamics
def f_value(x, u, parameters, x_star):
    x_dot = []

    sigma = parameters['sigma']
    b = parameters['b']
    r = parameters['r']

    x_shift=(x+x_star)  # coordinates transformation

    x_dot = [-sigma*(x_shift[:, 0] - x_shift[:, 1]) + u[:, 0],
             r*x_shift[:, 0] - x_shift[:, 1] - x_shift[:, 0]*x_shift[:, 2] + u[:, 1],
             x_shift[:, 0] * x_shift[:, 1] - b*x_shift[:, 2] + u[:, 2]]

    x_dot = torch.transpose(torch.stack(x_dot), 0, 1)  

    return x_dot


# Loading training parameters
parameters = config_file.get_params()


# Symbolic variables
x1 = dreal.Variable("x1")
x2 = dreal.Variable("x2")
x3 = dreal.Variable("x3")
vars_ = [x1, x2, x3]
                             

'''
Variables definition
'''
torch.manual_seed(seed_)

# instantiating variables
dim1 = (max_loop_number) * (max_iters)
x = torch.Tensor(parameters['N'], parameters['n_input']).uniform_(-gamma_overbar, gamma_overbar)
x_dataset_init = x.detach().clone()  # initial dataset          
x_0 = torch.zeros([1, parameters['n_input']])

wc1_hist = np.empty((dim1, parameters['contr_hid1'], parameters['n_input']))  # 1st control layer weight
wc2_hist = np.empty((dim1, parameters['contr_hid2'], parameters['contr_hid1']))  # 2nd control layer weight
wc3_hist = np.empty((dim1, parameters['contr_out'], parameters['contr_hid2']))  # 3rd control layer weight

bc1_hist = np.empty((dim1, 1, parameters['contr_hid1']))  # 1st control layer bias
bc2_hist = np.empty((dim1, 1, parameters['contr_hid2']))  # 2nd control layer bias
bc3_hist = np.empty((dim1, 1, parameters['contr_out']))  # 3rd control layer bias

q_hist = np.empty((dim1, parameters['contr_out'], parameters['n_input']))
w1_hist = np.empty((dim1, parameters['lyap_hid1'], parameters['n_input']))

Lyap_risk_ELR_hist = np.empty(dim1)
Lyap_risk_SLR_hist = np.empty(dim1)

V_dot_hist = np.empty(dim1)
V_hist = np.empty(dim1)
V0_hist = np.empty(dim1)

V_tuning_hist = np.empty(dim1)
Vdot_tuning_hist = np.empty(dim1)

learning_rate_history = np.empty(dim1)
learning_rate_c_history = np.empty(dim1)

ce_found = np.empty((dim1, parameters['n_input']))
disc_viol_found = np.empty((dim1, parameters['n_input']))


# Assigning vectors values to NaN
wc1_hist[:] = np.NaN
wc2_hist[:] = np.NaN
wc3_hist[:] = np.NaN
bc1_hist[:] = np.NaN
bc2_hist[:] = np.NaN
bc3_hist[:] = np.NaN
q_hist[:] = np.NaN
w1_hist[:] = np.NaN
Lyap_risk_ELR_hist[:] = np.NaN
Lyap_risk_SLR_hist[:] = np.NaN
V_dot_hist[:] = np.NaN
V_hist[:] = np.NaN
V0_hist[:] = np.NaN
V_tuning_hist[:] = np.NaN
Vdot_tuning_hist[:] = np.NaN
ce_found[:] = np.NaN
learning_rate_history[:] = np.NaN
learning_rate_c_history[:] = np.NaN
disc_viol_found[:] = np.NaN


out_iters = 0
found_lyap_f = False
start = timeit.default_timer()
t_falsifier = 0
init_date = datetime.now()

# campaign-specific variables
tot_iters = 0      # total number of iterations
to_fals = False    # time out falsifier
to_learner = False # time out learner


'''
Main
'''
while out_iters < max_loop_number and not found_lyap_f: 

    if (parameters['control_initialised']):
        lqr = init_control.detach().clone() 
    else:
        lqr = torch.rand([parameters['contr_out'], parameters['n_input']])-0.5

    # Instantiating ANN architecture
    model = Net(parameters, seed_)

    i_epoch = 0  # epochs counter

    # setting up optimiser
    list_c = ['control.weight', 
              'control1.weight', 'control2.weight', 'control3.weight',
              'control1.bias', 'control2.bias', 'control3.bias']
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in list_c, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in list_c, model.named_parameters()))))
    optimizer = torch.optim.Adam([{'params': base_params}, 
                                  {'params': params, 'lr': parameters['learning_rate_c']}], 
                                     lr=parameters['learning_rate'])

    # setting up LR scheduluer
    if parameters['use_scheduler']: 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=parameters['sched_T'], eta_min=0)        

    # computing the system (symbolic) dynamics at the first iteration
    if parameters['compare_first_last_iters']:
        
        u_NN0, u_NN1, u_NN2, V_learn0, f_out_sym =\
            sym_dyn.symbolic_sys(parameters,
                                 vars_,
                                 model, x_star)
    
        lie_derivative_of_V0 = Functions.LieDerivative(vars_, f_out_sym, 
                                                        V_learn0, 
                                                        gamma_underbar, 
                                                        gamma_overbar, 
                                                        config, 
                                                        epsilon)

    # training cycle
    while i_epoch < max_iters and not found_lyap_f:

        # sliding window
        if (parameters['sliding_window'] and x.shape[0] > parameters['N_max']):

            if parameters['debug_info']:
                print("DEBUG 'sliding wind': Sliding x.")
                print(f"DEBUG 'sliding wind': x_old size = {x.shape[0]}")

            # removing initial points from the current dataset
            x_noinit = x[x_dataset_init.shape[0] : x.shape[0], :]

            # sliding oldest CEs points out 
            x_sw = x_noinit[(x.shape[0]-parameters['N_max']):, :]

            # restoring initial points
            x = torch.cat((x_dataset_init, x_sw), 0)

            if parameters['debug_info']:
                print(f"DEBUG 'sliding wind': x_new size = {x.shape[0]}\n")    

        # Forward ANN pass
        V_candidate, u = model(x) 
        V0, u0 = model(x_0)  # evaluating model at the origin

        # Forward dynamics pass
        f = f_value(x, u, parameters, x_star)

        # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ -- Translator
        Lie_V = compute_Lie_der.compute_v2(model, f, x,
                                           parameters)


        f0 = f_value(x_0, u0, parameters, x_star)  # evaluating system in f(0,0)
        ref = torch.zeros(1, parameters['n_input'])  # reference value in the origin                            
        Circle_Tuning = Functions.Norm2(x)

        # Extended Lyapunov risk (training loss function)
        Lyap_risk_ELR = alpha_5 * (\
                        alpha_1 * F.relu(-V_candidate).sum() +\
                        alpha_2 * F.relu(Lie_V).sum() +\
                        alpha_3 * (V0).pow(2) +\
                        alpha_4 * ((Circle_Tuning - alpha_roa*(V_candidate)).pow(2)).mean())


        # Loss function related to the first 3 Lyapunov conditions
        Lyap_risk_SLR = alpha_1 * F.relu(-V_candidate).sum() +\
                        alpha_2 * F.relu(Lie_V).sum() +\
                        alpha_3 * (V0).pow(2)

        message_update = f"(#{out_iters + 1}/{max_loop_number}, #{i_epoch}/{max_iters})\
         L. Risk = {Lyap_risk_ELR.item()}"
        print(message_update)

        #Saving computed cost function values
        Lyap_risk_ELR_hist[out_iters * max_iters + i_epoch] = \
            Lyap_risk_ELR.item()
        Lyap_risk_SLR_hist[out_iters * max_iters + i_epoch] = \
            Lyap_risk_SLR.item()
        V_hist[out_iters * max_iters + i_epoch] = \
            alpha_1*F.relu(-V_candidate).sum()
        V_dot_hist[out_iters * max_iters + i_epoch] = \
            alpha_2*F.relu(Lie_V).sum()
        V0_hist[out_iters * max_iters + i_epoch] = \
            alpha_3*(V0).pow(2)
        V_tuning_hist[out_iters * max_iters + i_epoch] = \
            alpha_4 * ((Circle_Tuning - alpha_roa*(V_candidate)).pow(2)).mean()


        # SGD step
        optimizer.zero_grad()
        Lyap_risk_ELR.backward()
        optimizer.step()
        if parameters['use_scheduler']:
            scheduler.step()  #Lyap_risk_ELR
        
        # clipping weight
        if parameters['clipping_V']:
            model.layer1.weight.data = model.layer1.weight.data.clamp_min(np.finfo(float).eps)
            model.layer2.weight.data = model.layer2.weight.data.clamp_min(np.finfo(float).eps)
            model.layer3.weight.data = model.layer3.weight.data.clamp_min(np.finfo(float).eps)

        # saving learning rate
        learning_rate_history[out_iters * max_iters + i_epoch] =\
            optimizer.param_groups[0]["lr"]
        learning_rate_c_history[out_iters * max_iters + i_epoch] =\
            optimizer.param_groups[1]["lr"]

        # Saving model weight (for postprocessing)
        w1_hist, q_hist, wc1_hist, wc2_hist, wc3_hist, bc1_hist, bc2_hist, bc3_hist = \
            Functions.SaveWeightHist(model, parameters, w1_hist, q_hist, 
                                     wc1_hist, wc2_hist, wc3_hist,
                                     bc1_hist, bc2_hist, bc3_hist,
                                     out_iters, max_iters, i_epoch)

        # Augmented falsifier
        if (Lyap_risk_SLR==0):
            
            print('\n=========== Augmented Falsifier ==========')        
            start_ = timeit.default_timer() 
            
            # Computing the system symbolic dynamics                
            u_NN0, u_NN1, u_NN2, V_learn, f_out_sym =\
                sym_dyn.symbolic_sys(parameters,
                                     vars_,
                                     model, x_star)

            print("\nDiscrete Falsifier computing CEs ...")
            lie_derivative_of_V = Functions.LieDerivative(vars_, f_out_sym, 
                                                          V_learn, 
                                                          gamma_underbar, 
                                                          gamma_overbar, 
                                                          config, 
                                                          epsilon)

            x, disc_viol_found[out_iters * max_iters + i_epoch] = \
                Functions.AddLieViolationsOrder3_v4(x, 
                                                    gamma_overbar, 
                                                    parameters['grid_points'],
                                                    parameters['zeta_D'],
                                                    parameters['debug_info'],
                                                    V_learn,
                                                    lie_derivative_of_V)

            # If no CE is found, invoke the SMT Falsifier
            if (disc_viol_found[out_iters * max_iters + i_epoch, 0]==0):

                print("\nSMT Falsifier computing CE ...")
                try:
                    start_ = timeit.default_timer() 

                    CE_SMT, \
                    lie_derivative_of_V = Functions.CheckLyapunov(vars_, 
                                                                  f_out_sym, 
                                                                  V_learn, 
                                                                  gamma_underbar, 
                                                                  gamma_overbar, 
                                                                  config, 
                                                                  epsilon)


                except TimeoutError:
                    print("SMT Falsifier Timed Out")
                    tot_falsifier_to += 1  # increasing counter of falsifier time out
                    to_fals = True
                    stop_ = timeit.default_timer()
                    fals_to_check = stop_ - start_
                    time.sleep(5)
                    break


                if (CE_SMT):
                    # if a counterexample is found
                    print("SMT found a CE: ")
                    print(CE_SMT)

                    # Adding midpoint of the CE_SMT to the history
                    for i_dim_ce in range(parameters['n_input']):  
                        ce_found[out_iters * max_iters + i_epoch, i_dim_ce] = CE_SMT[i_dim_ce].mid()    

                    x = Functions.AddCounterexamples(x, CE_SMT, parameters['zeta_SMT'])
                    if parameters['debug_info']:
                        print(f"LOG 'SMT Falsifier': Added {parameters['zeta_SMT']} points in the vicinity of the CE.\n")

                else:
                    # no CE_SMT is returned hence V_learn is a valid Lyapunov
                    # function
                    print("NO CE_SMT found.")
                    found_lyap_f = True
                    print("\nControl Lyapunov Function satisfying conditions!")
                    print(V_learn, " is a Lyapunov function.")

            else:
                print(f"Skipping SMT callback.\n") 

            stop_ = timeit.default_timer() 
            t_falsifier += (stop_ - start_)
            print('================================') 

        i_epoch += 1
        tot_iters += 1

    out_iters+=1
    if (out_iters == max_loop_number and not found_lyap_f):
        print('============================================================')
        print('Lyapunov function NOT FOUND within the max number')
        print('of iterations.')
    print('============================================================')

# Saving times for statistics generation
stop = timeit.default_timer()
seconds_elapsed = stop - start
minutes_elapsed = seconds_elapsed / 60
hours_elapsed = minutes_elapsed / 60
falsifier_elapsed = t_falsifier / 60
end_date = datetime.now()

print('\n')
print("Total time [s]: ", seconds_elapsed)
print("Total time [']: ", minutes_elapsed)
print("Total time [h]: ", hours_elapsed)
print("Falsifier time [']: ", falsifier_elapsed)
print(f"Falsifier time [%]: {falsifier_elapsed/minutes_elapsed*100}")
print("\n")

if not found_lyap_f:

    # Computing the system symbolic dynamics used during postprocessing           
    u_NN0, u_NN1, u_NN2, V_learn, f_out_sym =\
        sym_dyn.symbolic_sys(parameters,
                             vars_,
                             model, x_star)

    lie_derivative_of_V = Functions.LieDerivative(vars_, f_out_sym, 
                                                  V_learn, 
                                                  gamma_underbar, 
                                                  gamma_overbar, 
                                                  config, 
                                                  epsilon)

    if not to_fals:
        # if the test was completed and the Falsifier was not the cause of TO
        tot_learner_to += 1  # increasing counter of learner time out
        to_learner = True


'''
Postprocessing
'''

# 0) Generating result folder
try:
    final_dir_run = final_dir_campaign + "/" + str(i_loop)
    os.mkdir(final_dir_run)
except OSError:
    print("Creation of the result directory %s failed" % final_dir_run)
else:
    print("Result directory successfully created as: \n %s \n" % final_dir_run)


if execute_postpro:

    # Executing postprocessing
    exec(open("postprocessing/postpro_3d_template.py").read())

    # Perform closed loop (convergence) tests    
    if(parameters['test_closed_loop_dynamics']):
    
        print("Run closed-loop test.")
        samples_number = int(parameters['end_time'] / parameters['Dt'])
    
        # Setting up three closed-loop tests
        initial_x1 =     [gamma_overbar,      -gamma_overbar/2,      gamma_overbar]
        initial_x2 =     [gamma_overbar,       gamma_overbar/2,      gamma_overbar]
        initial_x3 =     [gamma_overbar,      -gamma_overbar/2,      gamma_overbar]
        desired_x1 =     [x_star[0].item(),   x_star[0].item(),   x_star[0].item()]
        desired_x2 =     [x_star[1].item(),   x_star[1].item(),   x_star[1].item()]
        desired_x3 =     [x_star[2].item(),   x_star[2].item(),   x_star[2].item()]
        control_active = [1,                                  1,                 0]
    
        # Creating closed-loop folder
        final_dir_test = final_dir_plots + "closed_loop_tests/"
        os.mkdir(final_dir_test)

        for iTest in range(len(initial_x1)):
            message = "Closed-loop " + str(iTest + 1) + "/" + str(len(initial_x1))
            print(message)

            final_dir_ = final_dir_test + "test_#" + str(iTest) + "/"
            os.mkdir(final_dir_)

            des_x1 = desired_x1[iTest]
            des_x2 = desired_x2[iTest]
            des_x3 = desired_x3[iTest]
            init_x1 = initial_x1[iTest]
            init_x2 = initial_x2[iTest]
            init_x3 = initial_x3[iTest]
            control_active_test = control_active[iTest]

            # closed-loop test callback
            cl.closed_loop_system(samples_number, model,
                                  des_x1, des_x2, des_x3,
                                  control_active_test, 
                                  init_x1, init_x2, init_x3,
                                  parameters,
                                  final_dir_,
                                  dynamic_sys, 
                                  gamma_overbar, gamma_underbar)

            # saving test report
            result_report = [f"Initial x1 = {init_x1} [m/s]\
                             \nInitial x2 = {init_x2} [m/s]\
                             \nInitial x3 = {init_x3} [rad/s]\
                             \nDesired x1 = {des_x1} [m/s]\
                             \nDesired x2 = {des_x2} [m/s]\
                             \nDesired x3 = {des_x3} [rad/s]\
                             \nIs control active = {control_active_test}\
                             "]
            np.savetxt(final_dir_ + "test_report.txt", result_report, fmt="%s")

        # Plot 3D trajectories  
        if parameters['trajectory_closed_loop'] and found_lyap_f:
        
            samples_number = int(parameters['end_time_traj'] / parameters['Dt'])

            final_dir_traj = final_dir_plots + "closed_loop_traj/"
            os.mkdir(final_dir_traj)
    
            sol_x = np.zeros((parameters['traj_no'], samples_number, parameters['n_input']))
            sol_u = np.zeros((parameters['traj_no'], samples_number, parameters['contr_out']))
            sol_V = np.zeros((parameters['traj_no'], samples_number, 1))
    
            for iTest in range(parameters['traj_no']):
                message = "Trajectory " + str(iTest + 1) + "/" + str(parameters['traj_no'])
                print(message)
    
                # initialisation
                des_x1 = x_star[0].item()
                des_x2 = x_star[1].item()
                des_x3 = x_star[2].item()
                sign_1 = random.randint(0,1)
                sign_2 = random.randint(0,1)
                sign_3 = random.randint(0,1)
                if (sign_1==0):
                    sign_1 = -1
                if (sign_2==0):
                    sign_2 = -1
                if (sign_3==0):
                    sign_3 = -1

                    
                init_x1 = ((random.random()*(gamma_overbar-gamma_underbar))+gamma_underbar)*sign_1
                init_x2 = ((random.random()*(gamma_overbar-gamma_underbar))+gamma_underbar)*sign_2
                init_x3 = ((random.random()*(gamma_overbar-gamma_overbar))+gamma_overbar)*sign_3
                control_active_test = 1
    
                
                sol_x[iTest, :, :], sol_u[iTest, :, :], sol_V[iTest, :, :] = \
                    cl.traj_system(samples_number, model,
                                   des_x1, des_x2, des_x3,
                                   control_active_test, 
                                   init_x1, init_x2, init_x3,
                                   parameters,
                                   dynamic_sys)
                
            cl.plot_traj(sol_x, sol_V, final_dir_traj, samples_number, 
                         parameters)


else:
    # Automatic log report
    saving_log.gen_log(system_name, found_lyap_f, seed_, max_loop_number, max_iters, 
                       parameters, x, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, 
                       alpha_roa, epsilon, gamma_underbar, gamma_overbar, config, 
                       model, to_fals, to_learner, seconds_elapsed, minutes_elapsed, 
                       hours_elapsed, out_iters, i_epoch, start, init_date,
                       end_date, falsifier_elapsed, final_dir_run)
    
print(f"\n{i_loop+1}/{tot_runs} training run terminated.\n\n")
