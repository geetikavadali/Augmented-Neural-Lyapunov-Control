#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:37:07 2022

@author: Davide Grande

This script callback is executed by the sim_campaign_ANLC_main.py that 
iteratively runs this file to generated the simulation campaign runs.

"""

class dyn_sys_params():
    # A generic class to store the dynamics parameters
    pass


def f_value(x, u, dyn_sys_params):
    #Inverted pendulum Dynamics
    x_dot = []
    
    G = dyn_sys_params.G
    L = dyn_sys_params.L
    m = dyn_sys_params.m
    b = dyn_sys_params.b

    for r in range(0,len(x)): 
        f = [ x[r][1], 
              (m*G*L*np.sin(x[r][0]) - b*x[r][1]) / (m*L**2)]
        x_dot.append(f)

    x_dot = torch.tensor(x_dot)
    x_dot[:, 1] = x_dot[:, 1] + (u[:,0]/(m*L**2))

    return x_dot


'''
Training parameters
'''
skip_training = False

N = 500                # initial dataset size
N_max = 1000            # maximum dataset size (if using a sliding window)
use_reinit_dataset = False  # re-initialise the first random points of the
                            # dataset at each callback
# Training params
max_loop_number = max_loops     # passed from the looper file
max_iters_per_loop = max_iters  # passed from the looper file
use_old_solution = control_initialised  # passed from the looper file


'''
Learner
'''
# Lyapunov ANN params
D_in = 2             # input dimension (do not change!!)
H1 = 10              # hidden dimension Lyapunov layer 1
H2 = 10             # hidden dimension Lyapunov layer 2
D_out = 1            # output dimension Lyapunov (do not change this!)
Lyap_act_fun1 = 'linear'  # activation function options:  'tanh', 'pow2', 'sfpl', 'linear' 
Lyap_act_fun2 = 'pow2'
Lyap_act_fun3 = 'linear'
Lyap_bias1 = False  # use bias on 1st layer
Lyap_bias2 = False  # use bias on 2nd layer
Lyap_bias3 = False  # use bias on 3rd layer
beta_sfpl = 2  # the higher, the steeper the Softplus, the closer sfpl(0) = 0


# Control ANN params
use_lin_ctr = True  # use linear control law
init_control = torch.tensor([[-23.58639732,  -5.31421063]])  # initial control solution
contr_hid1 = 2  # hidden dimension control layer 1
contr_hid2 = 2 # hidden dimension Lyapunov layer 2
contr_out = 1  # output dimension control ANN (do not change this!)
contr_bias1 = False  # use bias on 1st layer (if nonlinear ctr is used)
contr_bias2 = False  # use bias on 2nd layer (if nonlinear ctr is used)
contr_bias3 = False  # use bias on 3rd layer (if nonlinear ctr is used)
contr_act_fun1 = 'sfpl' # activation function options:  'tanh', 'sfpl', 'linear'
contr_act_fun2 = 'sfpl' # activation function options:  'tanh', 'sfpl', 'linear'
contr_act_fun3 = 'linear' # activation function options:  'tanh', 'sfpl', 'linear'
lin_contr_bias = False  # use bias on linear control layer


# Learning rate params
learning_rate = 0.01   # learning rate Lyapunov branch
learning_rate_c = 1.0  # learning rate control branch
T_scheduler = 100  # half period of the scheduler (if used)


# Loss function
alpha_1 = 1.0  # weight V
alpha_2 = 1.0  # weight V_dot
alpha_3 = 1.0  # weight V0
alpha_4 = 0.0  # weight tuning term V
alpha_roa = gamma_overbar  # Lyapunov function steepness
alpha_5 = 1.0  # general scaling factor

'''
Falsifier (SMT)
'''
# Checking candidate V within a ball around the origin (gamma_underbar ≤ sqrt(∑xᵢ²) ≤ gamma_overbar)
zeta_SMT = 10     # how many points are added to the dataset after a CE box 
                  # is found


'''
Discrete falsifier (DF)
'''
grid_points = 500              # sampling size grid
max_grid_points = 1000         # maximum grid points
adaptive_grid = True           # using grid with adaptive number of points
zeta_D = 300                    # how many points are added at each DF callback


# Instantiating symbolic variables and dynamic parameters
x1 = dreal.Variable("x1") 
x2 = dreal.Variable("x2")
vars_ = [x1, x2]
dyn_sys_params.G = 9.81  # gravity constant
dyn_sys_params.L = 0.5   # length of the pole 
dyn_sys_params.m = 0.15  # ball mass
dyn_sys_params.b = 0.1   # friction


'''
Postprocessing
'''
dpi_ = 300  # DPI number for plots
debug_info = True  # print debug info
plot_V = True
plot_Vdot = True
plot_ctr_weights = True
plot_lyap_weights = False


'''
Variables definition
'''
torch.manual_seed(seed_)  
x = torch.Tensor(N, D_in).uniform_(-gamma_overbar, gamma_overbar)  # initial dataset 
x_0 = torch.zeros([1, D_in])
x_dataset_init =  x.detach().clone()  # initial dataset 
dim1 = (max_loop_number) * (max_iters_per_loop)
w1_hist = np.empty((dim1, H1, D_in))  # 1st Lyap. layer weight
w2_hist = np.empty((dim1, H2, H1))  # 2nd Lyap. layer weight
w3_hist = np.empty((dim1, 1, H2))  # 3rd Lyap. layer weight
wc1_hist = np.empty((dim1, contr_hid1, D_in))  # 1st control layer weight
wc2_hist = np.empty((dim1, contr_hid2, contr_hid1))  # 2nd control layer weight
wc3_hist = np.empty((dim1, contr_out, contr_hid2))  # 3rd control layer weight
q_hist = np.empty((dim1, contr_out, D_in))
Lyap_risk_ELR_hist = np.empty(dim1)
Lyap_risk_SLR_hist = np.empty(dim1)
V_dot_hist = np.empty(dim1)
V_hist = np.empty(dim1)
V0_hist = np.empty(dim1)
V_tuning_hist = np.empty(dim1)
learning_rate_history = np.empty(dim1)
learning_rate_c_history = np.empty(dim1)
ce_found = np.empty((dim1, D_in))
disc_viol_found = np.empty((dim1, D_in))


# Assigning vectors values to NaN
w1_hist[:] = np.NaN
w2_hist[:] = np.NaN
w3_hist[:] = np.NaN
wc1_hist[:] = np.NaN
wc2_hist[:] = np.NaN
wc3_hist[:] = np.NaN
q_hist[:] = np.NaN
Lyap_risk_ELR_hist[:] = np.NaN
Lyap_risk_SLR_hist[:] = np.NaN
V_dot_hist[:] = np.NaN
V_hist[:] = np.NaN
V0_hist[:] = np.NaN
V_tuning_hist[:] = np.NaN
ce_found[:] = np.NaN
learning_rate_history[:] = np.NaN
learning_rate_c_history[:] = np.NaN
disc_viol_found[:] = np.NaN


out_iters = 0
found_lyap_f = False
start = timeit.default_timer()
t_falsifier = 0
init_date = datetime.now()
stop_adaptive_grid = False
tot_iters = 0      # total number of iterations
to_fals = False    # time out falsifier
to_learner = False # time out learner

'''
Main
'''
while out_iters < max_loop_number and not found_lyap_f: 

    if (use_old_solution):
        lqr = init_control.detach().clone()    # lqr solution  
    else:
        lqr = torch.rand([contr_out, D_in])-0.5

    model = Net_v0(D_in, H1, H2, D_out,
                   Lyap_act_fun1, Lyap_act_fun2, Lyap_act_fun3,
                   Lyap_bias1, Lyap_bias2, Lyap_bias3,
                   use_lin_ctr, lin_contr_bias, lqr,
                   contr_hid1, contr_hid2, contr_out,
                   contr_bias1, contr_bias2, contr_bias3,
                   contr_act_fun1, contr_act_fun2, contr_act_fun3, 
                   beta_sfpl)
    

    i_epoch = 0  # epochs counter

    # setting up optimiser
    list_c = ['control.weight', 
              'control1.weight', 'control2.weight', 'control3.weight',
              'control1.bias', 'control2.bias', 'control3.bias']
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in list_c, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in list_c, model.named_parameters()))))
    optimizer = torch.optim.Adam([{'params': base_params}, {'params': params, 'lr': learning_rate_c}], lr=learning_rate)
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # basic declaration

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_scheduler, eta_min=0)


    while i_epoch < max_iters_per_loop and not found_lyap_f:

        if use_reinit_dataset:
            # re-initialising initial dataset points
            x_dataset_init = torch.Tensor(N, D_in).uniform_(-gamma_overbar, gamma_overbar)

        if (sliding_window and x.shape[0] > N_max):

            if debug_info:
                print("DEBUG 'sliding wind': Sliding x.")
                print(f"DEBUG 'sliding wind': x_old size = {x.shape[0]}")

            # removing initial points from the current dataset
            x_noinit = x[x_dataset_init.shape[0] : x.shape[0], :]

            # sliding oldest CEs points out 
            x_sw = x_noinit[(x.shape[0]-N_max) : x_noinit.shape[0], :]

            # restoring initial points
            x = torch.cat((x_dataset_init, x_sw), 0)

            if debug_info:
                print(f"DEBUG 'sliding wind': x_new size = {x.shape[0]}\n") 

        # Forward ANN pass
        V_candidate, u = model(x) 
        V0, u0 = model(x_0)  # evaluating model at the origin

        # Forward dynamics pass
        f = f_value(x, u, dyn_sys_params)

        # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
        Lie_V = compute_Lie_der.compute(model, f, x,
                                        Lyap_bias1, Lyap_bias2, Lyap_bias3,
                                        Lyap_act_fun1, Lyap_act_fun2, Lyap_act_fun3,
                                        D_in, H1, H2, beta_sfpl)


        f0 = f_value(x_0, u0, dyn_sys_params)  # evaluating system in f(0,0)
        ref = torch.zeros(1, D_in)  # reference value in the origin                            
        Circle_Tuning = Functions.Tune(x)

        # Extended Lyapunov risk (training loss function)
        Lyap_risk_ELR = alpha_5 * (\
                        alpha_1 * (F.relu(-V_candidate)).sum() +\
                        alpha_2 * (F.relu(Lie_V)).sum() +\
                        alpha_3 * (V0).pow(2) +\
                        alpha_4 * ((Circle_Tuning - alpha_roa*(V_candidate)).pow(2)).mean())

        # Loss function related to the first 3 Lyapunov conditions
        Lyap_risk_SLR = alpha_1 * (F.relu(-V_candidate)).sum() +\
                        alpha_2 * (F.relu(Lie_V)).sum() +\
                        alpha_3 * (V0).pow(2)

        message_update = f"(#{out_iters + 1}/{max_loop_number}, #{i_epoch}/{max_iters_per_loop})\
         loss = {Lyap_risk_ELR.item()}"
        print(message_update)

        #Saving computed cost function values
        Lyap_risk_ELR_hist[out_iters * max_iters_per_loop + i_epoch] = \
            Lyap_risk_ELR.item()
        Lyap_risk_SLR_hist[out_iters * max_iters_per_loop + i_epoch] = \
            Lyap_risk_SLR.item()
        V_hist[out_iters * max_iters_per_loop + i_epoch] = \
            alpha_1*F.relu(-V_candidate).sum()
        V_dot_hist[out_iters * max_iters_per_loop + i_epoch] = \
            alpha_2*F.relu(Lie_V).sum()
        V0_hist[out_iters * max_iters_per_loop + i_epoch] = \
            alpha_3*(V0).pow(2)
        V_tuning_hist[out_iters * max_iters_per_loop + i_epoch] = \
            alpha_4 * ((Circle_Tuning - alpha_roa*(V_candidate)).pow(2)).mean()

        optimizer.zero_grad()
        Lyap_risk_ELR.backward()
        optimizer.step()
        scheduler.step()

        # saving learning rate
        learning_rate_history[out_iters * max_iters_per_loop + i_epoch] =\
            optimizer.param_groups[0]["lr"]
        learning_rate_c_history[out_iters * max_iters_per_loop + i_epoch] =\
            optimizer.param_groups[1]["lr"]

        # Parameters Lyapunov ANN
        w1 = model.layer1.weight.data.numpy()
        w2 = model.layer2.weight.data.numpy()
        w3 = model.layer3.weight.data.numpy()
        if Lyap_bias1:
            b1 = model.layer1.bias.data.numpy()
        if Lyap_bias2:
            b2 = model.layer2.bias.data.numpy()
        if Lyap_bias3:
            b3 = model.layer3.bias.data.numpy()

        w1_hist[out_iters * max_iters_per_loop + i_epoch] = w1
        w2_hist[out_iters * max_iters_per_loop + i_epoch] = w2
        w3_hist[out_iters * max_iters_per_loop + i_epoch] = w3

        # Parameters control ANN
        if use_lin_ctr:
            q = model.control.weight.data.numpy()
            q_hist[out_iters * max_iters_per_loop + i_epoch] = q
        else:
            wc1 = model.control1.weight.data.numpy()
            wc2 = model.control2.weight.data.numpy()
            wc3 = model.control3.weight.data.numpy()
            if contr_bias1:
                bc1 = model.control1.bias.data.numpy()
            if contr_bias2:
                bc2 = model.control2.bias.data.numpy()

            wc1_hist[out_iters * max_iters_per_loop + i_epoch] = wc1
            wc2_hist[out_iters * max_iters_per_loop + i_epoch] = wc2
            wc3_hist[out_iters * max_iters_per_loop + i_epoch] = wc3

        # Falsification
        if (Lyap_risk_SLR==0):

            print('\n=========== Augmented Falsifier ==========')        
            start_ = timeit.default_timer() 

            # Computing the system symbolic dynamics
            u_NN, V_learn, f_out_sym, \
            f_closed_1, f_closed_2, f_open_1, f_open_2 =\
                sym_dyn.symbolic_sys(use_lin_ctr, 
                                     contr_act_fun1, contr_act_fun2, contr_act_fun3, 
                                     contr_bias1, contr_bias2, contr_bias3, lin_contr_bias,
                                     Lyap_bias1, Lyap_bias2, Lyap_bias3,
                                     Lyap_act_fun1, Lyap_act_fun2, Lyap_act_fun3, 
                                     beta_sfpl,
                                     x1, x2, vars_,
                                     model,
                                     dyn_sys_params)

            # 1) Discrete Falsifier
            print("\nDiscrete Falsifier computing CEs ...")
            
            lie_derivative_of_V = Functions.LieDerivative(vars_, 
                                                          f_out_sym, 
                                                          V_learn, 
                                                          gamma_underbar, 
                                                          gamma_overbar, 
                                                          config, 
                                                          epsilon)

            x, grid_points, \
            disc_viol_found[out_iters * max_iters_per_loop + i_epoch] = \
                Functions.AddLieViolationsOrder2_v3(x, 
                                                    gamma_overbar, 
                                                    grid_points,
                                                    config,
                                                    lie_derivative_of_V,
                                                    zeta_D,
                                                    debug_info,
                                                    False,
                                                    V_learn
                                                    )  


            if (disc_viol_found[out_iters * max_iters_per_loop + i_epoch, 0]==0):
            # 1) SMT falsifier
            # Check the positivity of V and negativity of the Lie derivative
            # returning up to one CE at each callback

                try:
                    start_ = timeit.default_timer() 
                    print("SMT Falsifier computing CE ...")
                    CE_SMT, lie_derivative_of_V = Functions.CheckLyapunov(vars_, 
                                                                          f_out_sym, 
                                                                          V_learn, 
                                                                          gamma_underbar, 
                                                                          gamma_overbar, 
                                                                          config, 
                                                                          epsilon)
                except TimeoutError:
                    print("\nSMT Falsifier Timed Out")
                    tot_falsifier_to += 1  # increasing counter of falsifier time out
                    to_fals = True
                    stop_ = timeit.default_timer()
                    fals_to_check = stop_ - start_
                    time.sleep(5)
                    break


                if (CE_SMT):
                    # if a counterexample is found
                    print("\nSMT found a CE: ")
                    print(CE_SMT)
    
                    # Adding midpoint of the CE_SMT to the history
                    for i_dim_ce in range(D_in):  
                        ce_found[out_iters * max_iters_per_loop + i_epoch, i_dim_ce] = CE_SMT[i_dim_ce].mid()    

                    x = Functions.AddCounterexamples(x, CE_SMT, zeta_SMT)
                    if debug_info:
                        print(f"LOG 'SMT Falsifier': Added {zeta_SMT} points in the vicinity of the CE.\n")

                else:
                    # no CE_SMT is returned hence V_learn is a valid Lyapunov
                    # function

                    found_lyap_f = True
                    print("NO CE_SMT found")
                    print("\nControl Lyapunov Function satisfying conditions!")
                    print(V_learn, " is a Lyapunov function.")


            else:
                print(f"\nSkipping SMT callback.\n") 

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


if not found_lyap_f and not to_fals:
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
    print("Creation of the result directory %s failed" % final_dir)
else:
    print("Result directory successfully created as: \n %s \n" % final_dir)


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
                 "If linear control law is used then: \n" + 
                 "Use bias = " + str(lin_contr_bias) + "\n" +
                 "If nonlinear control law is used, then:\n" + 
                 "dim. layer 1 = " +str(contr_hid1) + "\n" +
                 "use bias layer 1 = " + str(contr_bias1) + "\n" +
                 "layer 1 act. f. = " + str(contr_act_fun1) + "\n" +
                 "dim layer 2 = " +str(contr_hid2) + "\n" +
                 "use bias layer 2 = " + str(contr_bias2) + "\n" +
                 "layer 2 act. f. = " + str(contr_act_fun2) + "\n" +
                 "use bias output layer = " + str(contr_bias3) + "\n" +
                 "layer out act. f. = " + str(contr_act_fun3) + "\n" +
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
                 "alpha_roa (ROA tuning) = " + str(alpha_roa) + "\n" +
                 "\n\n" +
                 "TRAINING: \n" +
                 "ANN control initialised (LQR) = " + str(use_old_solution) + "\n" +
                 "ANN control initial weights = " + str(init_control) + "\n" +
                 "ANN control final weights = " + str(model.control.weight.data.detach()) + "\n" +
                 "\n\n" +
                 "FALSIFIER (SMT): \n" +
                 "Epsilon = " + str(epsilon) + "\n" +
                 "Falsifier domain = " + str(gamma_underbar) + " - " +str(gamma_overbar) + "\n" +
                 "config.precision = " + str(config.precision) + "\n"
                 "zeta_SMT (SMT CE point cloud) = " + str(zeta_SMT) +
                 "\n\n" + 
                 "DISCRETE FALSIFIER (DF): \n" +  
                 "zeta_D (DF CEs added at each callback) = " + str(zeta_D) + "\n" +
                 f"use adaptive grid = {adaptive_grid}\n" +
                 f"maximum grid points = {max_grid_points}\n" +
                 f"Final grid points = {grid_points}\n" +
                 "\n\n" +
                 "RESULTS: \n" +
                 "Falsifier Time Out = " + str(to_fals) + "\n" +
                 "Learner Time Out = " + str(to_learner) + "\n" +
                 "Time elapsed: "+ "\n" +
                 "seconds = " + str(seconds_elapsed) + "\n" +
                 "minutes = " + str(minutes_elapsed) + "\n" +
                 "hours = " + str(hours_elapsed)+ "\n" +
                 "Falsifier time [']: " + str(falsifier_elapsed) + "\n" +
                 f"Falsifier time [%]: {falsifier_elapsed/minutes_elapsed*100}\n" +
                 f"Training iterations completed = {(out_iters)}\n" +
                 f"Training epochs (last iteration) = {i_epoch}\n\n"
                 ]


np.savetxt(final_dir_run + "/logs.txt", result_report, fmt="%s")


if (found_lyap_f and execute_postpro):
    
    test_closed_loop_dynamics = False
    end_time = 20
    Dt = 0.01
    load_ann = False
    path_ann_to_load = None


    if not found_lyap_f:
        # Computing the system symbolic dynamics
        u_NN, V_learn, f_out_sym, \
        f_closed_1, f_closed_2, f_open_1, f_open_2 =\
            sym_dyn.symbolic_sys(use_lin_ctr, 
                                 contr_act_fun1, contr_act_fun2, contr_act_fun3, 
                                 contr_bias1, contr_bias2, contr_bias3, lin_contr_bias,
                                 Lyap_bias1, Lyap_bias2, Lyap_bias3,
                                 Lyap_act_fun1, Lyap_act_fun2, Lyap_act_fun3, 
                                 beta_sfpl,
                                 x1, x2, vars_,
                                 model,
                                 dyn_sys_params)

        lie_derivative_of_V = Functions.LieDerivative(vars_, 
                                                      f_out_sym, 
                                                      V_learn, 
                                                      gamma_underbar, 
                                                      gamma_overbar, 
                                                      config, 
                                                      epsilon)
    
    # 0) Generating result folder
    try:
        folder_results = final_dir_run + "/results"
        current_dir = os.getcwd()
        final_dir = current_dir + "/" + folder_results + "/"
        os.mkdir(final_dir)
    except OSError:
        print("Creation of the result directory %s failed" % final_dir)
    else:
        print("Result directory successfully created as: \n %s \n" % final_dir)
    
    
    # Generating figures result folder
    try:
        folder_results_plots = folder_results + "/figures"
        current_dir = os.getcwd()
        final_dir_plots = current_dir + "/" + folder_results_plots + "/"
        os.mkdir(final_dir_plots)
    except OSError:
        print("Creation of the plot result directory %s failed" % final_dir_plots)
    else:
        print("Plot result directory successfully created as: \n %s \n" % final_dir_plots)
    
    
    exec(open("postprocessing/postpro_sim_campaign_ANLC.py").read())

