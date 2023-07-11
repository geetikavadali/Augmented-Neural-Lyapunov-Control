#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 20:07:22 2022

@author: docautomata


This script callback is executed by the sim_campaign_NLC_main.py that iteratively 
runs this file to generated the simulation campaign runs.

"""

class Net(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output,lqr):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden,n_output)
        self.control = torch.nn.Linear(n_input,1,bias=False)
        self.control.weight = torch.nn.Parameter(lqr)

    def forward(self,x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        out = sigmoid(self.layer2(h_1))
        u = self.control(x)
        return out,u
    
    
    
def f_value(x,u):
    #Dynamics
    y = []
    G = 9.81  # gravity
    L = 0.5   # length of the pole 
    m = 0.15  # ball mass
    b = 0.1   # friction coefficient
    
    for r in range(0,len(x)): 
        f = [ x[r][1], 
              (m*G*L*np.sin(x[r][0])- b*x[r][1]) / (m*L**2)]
        y.append(f) 
    y = torch.tensor(y)
    y[:,1] = y[:,1] + (u[:,0]/(m*L**2))
    return y



'''
For learning 
'''
N = 500             # sample size
D_in = 2            # input dimension
H1 = 6              # hidden dimension
D_out = 1           # output dimension
torch.manual_seed(seed_)  # seed passed from the looper file  
x = torch.Tensor(N, D_in).uniform_(-6, 6)           
x_0 = torch.zeros([1, 2])


'''
For verifying 
'''
mu = 10  # intervals of Falsifier callback
zeta_SMT = 10     # how many points are added to the dataset after a CE box 
                  # is found
x1 = Variable("x1")
x2 = Variable("x2")
vars_ = [x1,x2]
G = 9.81 
l = 0.5  
m = 0.15
b = 0.1


'''
Service variables
'''
dim1 = (max_loops) * (max_iters)
ce_found = np.empty((dim1, D_in))
ce_found[:] = np.NaN


out_iters = 0
valid = False
to_fals = False     # time out falsifier
to_learner = False  # time out learner
tot_iters = 0       # total number of iterations
fals_to_check = -1  # resetting falsifier timeout time. -1 is a convention to 
                    # spot when the 

while out_iters < max_loops and not valid: 
    start = timeit.default_timer()
    
    if control_initialised:  
        lqr = torch.tensor([[-23.58639732,  -5.31421063]])    # lqr solution
    else:
        lqr = torch.rand([1, 2])-0.5
        
    weight_init = copy.deepcopy(lqr)
    model = Net(D_in,H1, D_out,lqr)
    L = []
    i = 0 
    t = 0
   
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    while i < max_iters and not valid: 
        V_candidate, u = model(x)
        X0,u0 = model(x_0)
        f = f_value(x,u)
        Circle_Tuning = Functions.Tune(x)
        # Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
        L_V = torch.diagonal(torch.mm(torch.mm(torch.mm(Functions.dtanh(V_candidate),model.layer2.weight)\
                            *Functions.dtanh(torch.tanh(torch.mm(x,model.layer1.weight.t())+model.layer1.bias)),model.layer1.weight),f.t()),0)

        
        # Loss function
        if loss_function_with_tuning:
            # With tuning term 
            Lyapunov_risk = (F.relu(-V_candidate)+ 1.5*F.relu(L_V+0.5)).mean()\
                        +2.2*((Circle_Tuning-6*V_candidate).pow(2)).mean()+(X0).pow(2) 
        else:
            # Without tuning term
            #Lyapunov_risk = (F.relu(-V_candidate)+ 1.5*F.relu(L_V+0.5)).mean()+ 1.2*(X0).pow(2)
            Lyapunov_risk = (F.relu(-V_candidate)+ F.relu(L_V)).mean()+ (X0).pow(2)


        print(i, "Lyapunov Risk=",Lyapunov_risk.item()) 
        L.append(Lyapunov_risk.item())
        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 

        w1 = model.layer1.weight.data.numpy()
        w2 = model.layer2.weight.data.numpy()
        b1 = model.layer1.bias.data.numpy()
        b2 = model.layer2.bias.data.numpy()
        q = model.control.weight.data.numpy()

        # Falsification
        if i % mu == 0:
            u_NN = (q.item(0)*x1 + q.item(1)*x2) 
            f_sym = [ x2,
                    (m*G*l*sin(x1) + u_NN - b*x2) /(m*l**2)]

            # Candidate V
            z1 = np.dot(vars_,w1.T)+b1

            a1 = []
            for j in range(0,len(z1)):
                a1.append(tanh(z1[j]))
            z2 = np.dot(a1,w2.T)+b2
            V_learn = tanh(z2.item(0))

            print('===========Verifying==========')        
            start_ = timeit.default_timer() 
            '''
            result, lie_derivative_of_V = \
                Functions.CheckLyapunov(vars_, f_sym, V_learn, 
                                        ball_lb, ball_ub, 
                                        config,epsilon)
            '''

            try:
                print("SMT Falsifier computing CE ...")
                result, lie_derivative_of_V = Functions.CheckLyapunov(vars_, 
                                                                      f_sym, 
                                                                      V_learn, 
                                                                      ball_lb, 
                                                                      ball_ub, 
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
            
            stop_ = timeit.default_timer()

            if (result): 
                print("Not a Lyapunov function. Found counterexample: ")
                print(result)
                x = Functions.AddCounterexamples(x,result,zeta_SMT)
                
                # Adding midpoint of the CE_SMT to the history
                for i_dim_ce in range(D_in):  
                    ce_found[out_iters * max_loops + i, i_dim_ce] = result[i_dim_ce].mid()    

            else:  
                valid = True
                print("Satisfy conditions!!")
                print(V_learn, " is a Lyapunov function.")
            t += (stop_ - start_)
            print('==============================') 
        i += 1
        tot_iters += 1

    stop = timeit.default_timer()

    print('\n')
    print("Total time: ", stop - start)
    print("Verified time: ", t)
    
    out_iters+=1
    
if not valid and not to_fals:
    # if the test was completed and the Falsifier was not the cause of TO
    tot_learner_to += 1  # increasing counter of learner time out
    # parameter passed from looper file
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


result_report = ["Run of the Inverted Pendulum system (NLC).\n" + 
                 "Convergence reached = " + str(valid) + "\n\n" +
                 "Seed = " + str(seed_) + "\n" +
                 "LYAPUNOV ANN: \n" + 
                 "layer 1 dim. = " + str(H1) + "\n" +
                 "layer 2 activation f = tanh \n" +
                 "layer 1 has bias = True \n" +
                 "layer 2 activation f = tanh \n" +
                 "layer 2 has bias = True \n" +
                 "\n\n" +
                 "LEARNER: \n" + 
                 "Learning rate = " + str(learning_rate) + "\n" +
                 "Initial control weights = " + str(weight_init) + "\n" +
                 "Final control weights = " + str(model.control.weight.data.detach()) + "\n" +
                 "TRAINING: \n" +
                 "max_loop_number = " + str(max_loops) + "\n" +
                 "max_epochs_per_loop = " + str(max_iters) + "\n" +
                 "Initial dataset dimension = " + str(N) + "\n" +
                 "Final dataset dimension = " + str(len(x)) + "\n" + 
                 "Target ROA = " + str(ball_lb) + " - " +str(ball_ub) + "\n" +
                 "Loss function with tuning term = " + str(loss_function_with_tuning) + "\n" +
                 "\n\n" +
                 "FALSIFIER: \n" +  
                 "epsilon = " + str(epsilon) + "\n" +
                 "delta = " + str(config.precision) + "\n" +
                 "Falsifier interval (mu) = " + str(mu) + "\n" + 
                 "zeta_SMT = " + str(zeta_SMT) + "\n" +
                 "\n\n" +
                 "RESULTS: \n" +
                 "Falsifier Time Out = " + str(to_fals) + "\n" +
                 "Learner Time Out = " + str(to_learner) + "\n" +
                 "Time elapsed: "+ "\n" +
                 f"seconds = {stop - start}\n" +
                 "Falsifier time [']: " + str(t) + "\n" +
                 f"Training iterations completed = {(out_iters)}\n" +
                 f"Training epochs (last iteration) = {i}\n\n"
                 ]

np.savetxt(final_dir_run + "/logs.txt", result_report, fmt="%s")

    
