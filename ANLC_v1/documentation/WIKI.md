# Augmented Neural Lyapunov Control 
## A step-by-step guide

"I have a 3-dimensional system that I want to control, how can I set the code up?"

This tutorial is a walk-through the 5 quick steps needed to start your control synthesis in a matter of few minutes.

The system chosen for this example is described as:

$$
\begin{cases}
\dot{x}_1 = -\alpha (x_2^2) + 3 u_1  \\
\dot{x}_2 = x_1 x_3^3 + u_2 cos(\alpha) \\
\dot{x}_3 = 7 cos(x_2)+ \beta sin(x_1 x_2) + u_3
\end{cases}
$$

where $\alpha, \beta$ represent (scalar) dynamic parameters, $x=[x_1, x_2, x_3]^T$ the state-space vector and $u=[u_1, u_2, u_3]^T$ the control vector.

  
## Overall view
To use the framework, the following steps are recommended:

0. **Minimal working example**;
1. Set up the campaign *parameters*;
2. Set up the *Neural Network* architecture;  
3. Set up the configuration conditions for the *Falsifier*;
4. *Start* the training;  
5. The *procedure stops* when no counterexample is found or when a timeout is hit.  
  
The overall code architecture can be visualised as:
<img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/ANLC_v1/documentation/images/Code_modules.png" width=70% height=70%>



### Step-by-step example
As a 3-dimensional model must be controlled, the files can be modify accordingly:  
   
0. **Minimal working example**:
	1. Select the template relevant to the system dimension: for a 3-dimensional system the files are marked with ' _3d_ ', e.g. the main is `sim_campaign_3d_template_main.py`.
	2. If your dynamic system has parameters, set them up in `config/config_3d_template.py`:
	```python
	dyn_sys_params = {
		'alpha': 2.1,
		'beta': 5.3
	}
	```
		
	3. Define the dynamic system to be controlled:
		1. within `sim_campaign_3d_template_train/f_value()` function define the new dynamics (used to compute the Lie Derivative values in the loss function):
		```python
		def f_value(x, u, parameters):
			x_dot = []

			alpha = parameters['alpha']
			beta = parameters['beta']

			x_dot = [-alpha * x[:, 1]**2 + 3*u[:, 0],
				x[:, 0]*x[:, 2]**3 + u[:, 1]*torch.cos(alpha),
				7*torch.cos(x[:,1]) + beta*torch.sin(x[:, 0]*x[:, 1]) + u[:, 2]]

			x_dot = torch.transpose(torch.stack(x_dot), 0, 1)

	    	return x_dot
		```
		  
		2. within `utilities/sym_dyn_3d_template.py` define the new dynamics expressed with symbolic variables (used by dReal):
		```python
		f_out_sym = [-alpha * x2**2 + 3*u1,
			     x1*x2**3 + u2*dreal.cos(alpha),
			     7*dreal.cos(x2) + beta*dreal.sin(x1*x2) + u3]
		```
		
		
		3. if the synthesised closed-loop system should be tested upon convergence, the dynamics must be defined in `closed_loop_testing/cl_3d_template.py`.
	 	```python
	 	dydt = [(-alpha * x2**2 + 3*u1)*Dt + x1,
			(x1*x3**3 + u2*np.cos(alpha))*Dt + x2,
			(7*np.cos(x2) + beta*np.sin(x1*x2) + u3)*Dt + x3]
	 	```  
  
1. Set up the campaign *parameters*:
	1. select the number of training runs and the maximum learning iterations: (`sim_campaign_3d_template_main.py`);
	```python
	# Campaign parameters
	campaign_run = 1470  # number of the run.
		             # The results will be saved in sim_campaign/results/campaign_'campaign_run'
	tot_runs = 10        # total number of run of the campaigns (each with different seed)
	max_loop_number = 1  # number of loops per run (>1 means that the weights will be re-initialised).
		             # default = 1
	max_iters = 1000     # number of maximum learning iterations per run
	system_name = "control_3d_yoursystem_campaign"
	execute_postpro = True  #execute postprocessing
	```  
	This will launch a simulation campaign composed of 10 runs of 1000 learning iterations each. 
	The results will be generated in: `/sim_campaign/results/campaign_1470`
	
	2. select the loss function tuning parameters (`sim_campaign_3d_template_main.py`):
	```python
	# Loss function
	alpha_1 = 1.0   # weight V
	alpha_2 = 1.0   # weight V_dot
	alpha_3 = 0.0   # weight V0
	alpha_4 = 0.1   # weight tuning term V
	alpha_roa = 1.0  # Lyapunov function steepness
	alpha_5 = 1.0   # general scaling factor
	```  
	
	3. select dataset initial and final dimensions (`configuration/config_3d_template.py`):
	```python
	# Parameters for learner
	learner_params = {
    	'N': 50,                        # initial dataset size
    	'N_max': 1000                  # maximum dataset size (if using a sliding window)
    	}
	```  
	
2. Set up the *Neural Network* architecture: 
	1. select either the linear or nonlinear control laws (`configuration/config_3d_template.py`);
	```python
	# Parameters for control ANN
	control_params = {
        'use_lin_ctr': False,           # use linear control law  -- defined as 'phi' in the publication
	...
	}
	```  
	2. select the dimension of the hidden layers, the activation functions and learning rate values (`configuration/config_3d_template.py`);  
	

3.  Set configuration conditions for the *Falsifier*:
	1. select the SMT domain boundaries (gamma_underbar, gamma_overbar) and delta-precision (`sim_campaign_template_main.py`).     
4. *Start* the training:  
	1. launch the `sim_campaign_3d_template_main.py` files.
5. The procedure *stops* when no *counterexample* is found or when a *timeout* is hit:
	1. the timeout refers to either the maximum number of learning iterations (`sim_campaign_3d_template_main.py/max_iters`) OR to the maximum time dReal is allowed to search for a solution. 
	A value of 3 minutes is suggested for the latter and it is currently implemented. To modify it, change the value of `@timeout(180, use_signals=True)` in file `utilities/Functions/CheckLyapunov()`. 
	 

