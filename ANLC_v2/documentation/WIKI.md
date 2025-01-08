# Augmented Neural Lyapunov Control WIKI
## A step-by-step guide

"I have a 3-dimensional system that I want to control, how can I set the code up?"

This tutorial will guide you through the 6 steps encompassing what you need to start the synthesis of your control functions.

Let us consider the following system:

$$
\begin{cases}
\dot{x}_1 = -\alpha \cdot x_2^2 + 3 u_1  \\
\dot{x}_2 = x_1 x_3^3 + u_2 cos(\alpha) \\
\dot{x}_3 = 7 cos(x_2)+ \beta sin(x_1 x_2) + u_3
\end{cases}
$$

where $\alpha, \beta$ represent two (scalar) dynamic parameters, $x=[x_1, x_2, x_3]^T$ the state-space vector and $u=[u_1, u_2, u_3]^T$ the control vector.

  
## Overall view
To use the framework, the following steps are recommended:

1. Define the *system dynamics*;
2. Set up the training *parameters*;
3. Set up the *Neural Networks* architecture;
4. Set up the configuration conditions for the *Falsifier*;
5. *Start* the training;  
6. The *procedure stops* when no counterexample is found or when a timeout is reached.  
  

## Step-by-step example
Only 4 files need to be modified to run a custom example: `main_3d_template.py`, `utilities/models`, `configuration/config_3d_template.py` and `systems_dynamics/dynamics_3d_template.py`.  

The files can be modified as follows:    
   
1. **Define the system dynamics**:
    
	1. Define your dynamic system in `utilities/models.py`, taking `class ControlledLorenz()` as an example. 
	Create a new class called, for instance, `class YourNewModel()`, and define the dynamics as:  	
	```python
		class YourNewModel():

			def __init__(self):
				self.vars_ = [dreal.Variable("x1"), dreal.Variable("x2"), dreal.Variable("x3")]
        
			def f_torch(x, u, parameters):

				x1, x2, x3 = x[:,0], x[:,1], x[:,2]
				x1_shift = x1 + parameters['x_star'][0]
				x2_shift = x2 + parameters['x_star'][1]
				x3_shift = x3 + parameters['x_star'][2]
				u_NN0 = u[:,0]
				u_NN1 = u[:,1]
				u_NN2 = u[:,2]

				alpha = parameters['alpha']
				beta = parameters['beta']

				x_dot = [-alpha * x2_shift**2 + 3*u_NN0,
					x1_shift*x3_shift**3 + u_NN1*torch.cos(alpha),
					7*torch.cos(x2_shift) + beta*torch.sin(x1_shift*x2_shift) + u_NN2]

				x_dot = torch.transpose(torch.stack(x_dot), 0, 1)

			return x_dot
	```

	In the system dynamics, `x_star` represents the value of the equilibrium to be stabilised. Do not modify the lines related to `x1_shift`, `x2_shift`, etc.  
	  
	2. Similarly, the symbolic dynamics for the dReal verification is defined in `def f_symb()`.  
  	```python
		def f_symb(x, u, parameters, x_star):
			x1, x2, x3 = x[0], x[1], x[2]
			x1_shift = x1 + parameters['x_star'].numpy()[0]
			x2_shift = x2 + parameters['x_star'].numpy()[1]
			x3_shift = x3 + parameters['x_star'].numpy()[2]
			u_NN0 = u[0]
			u_NN1 = u[1]
			u_NN2 = u[2]
			
			alpha = parameters['alpha']
			beta = parameters['beta']
			
			x_dot = [-alpha * x2_shift**2 + 3*u_NN0,
				x1_shift*x3_shift**3 + u_NN1*dreal.cos(alpha),
				7*dreal.cos(x2_shift) + beta*dreal.sin(x1_shift*x2_shift) + u_NN2]
				
			return x_dot
	```
	   
	
	3. If your dynamics has scalar parameters, define them in `configuration/config_3d_template.py/def set_params():`
	```python
		dyn_sys_params = {
			'alpha': 2.1,
			'beta': 5.3,
		}
	```

 	4. In the `main_3d_template.py` file, import the model class defined in step 1.,  as:
   	```python
	from utilities.models import YourNewModel as UsedModel
	import configuration.config_3d_template as config_file
	import closed_loop_testing.cl_3d_template as cl
	import systems_dynamics.dynamics_3d_template as dynamic_sys
 	```  
 	where *YourNewModel* should match your new 3-dimensional model name.  
  
  	5. Optional: if you want to test the closed-loop system upon convergence, the dynamics should also be defined in `systems_dynamics/dynamics_3d_template.py` (currently a Forward Euler integrator is implemented).
 	```python
		dydt = [(-alpha * x2**2 + 3*u1)*Dt + x1,
			(x1*x3**3 + u2*np.cos(alpha))*Dt + x2,
			(7*np.cos(x2) + beta*np.sin(x1*x2) + u3)*Dt + x3]
 	```  
  
2. **Set up the training *parameters***:  
	All the parameters are included in the `configuration/config_3d_template.py` file.

	1. Select the number of training runs (`tot_runs`) and the maximum learning iterations per run (`max_iters`) as:
	```python
		campaign_params = {
			
			'init_seed': 1,         # initial campaign seed
			'campaign_run': 3000,  # number of the run.
									# The results will be saved in /results/campaign_'campaign_run'
			'tot_runs': 10,        # total number of runs of the campaigns (each one with a different seed)
			'max_loop_number': 1,  # number of loops per run (>1 means that the weights will be re-initialised).
									# default value = 1.
			'max_iters': 1000,     # number of maximum learning iterations per run
			'system_name': "your_model_3d",  # name of the systems to be controlled
			'x_star': torch.tensor([0.0, 0.0, 0.0]),  # target equilibrium point
		}    
	```  
	This parameters configuration will launch a simulation campaign composed of 10 runs of 1000 learning iterations each. 
	The results will be generated in: `/results/campaign_3000`.
	
	2. Select the loss function tuning parameters as:
	```python
		loss_function = {
			# Loss function tuning
			'alpha_1': 1.0,  # weight V
			'alpha_2': 1.0,  # weight V_dot
			'alpha_3': 1.0,  # weight V0
			'alpha_4': 0.1,  # weight tuning term V
			'alpha_roa': falsifier_params['gamma_overbar'],  # Lyapunov function steepness
			'alpha_5': 1.0,  # general scaling factor    
		}
	```  
	
	3. Select the dataset initial and final dimensions as:
	```python
		# Parameters for learner
		learner_params = {
				'N': 500,                      # initial dataset size
				'N_max': 1000,                 # maximum dataset size (if using a sliding window)
				...
			}
	```  
	  
3. **Set up the *Neural Networks* architecture**:   
	1. Select the structure of the Lyapunov ANN:  
	```python
		# Parameters for Lyapunov ANN
		lyap_params = {
			'n_input': 3, # input dimension (n = n-dimensional system)
			'beta_sfpl': 2,  # the higher, the steeper the Softplus, the better approx. sfpl(0) ~= 0
			'clipping_V': True,  # clip weight of Lyapunov ANN
			'size_layers': [10, 10, 1],  # CAVEAT: the last entry needs to be = 1 (this ANN outputs a scalar)!
			'lyap_activations': ['pow2', 'linear', 'linear'],
			'lyap_bias': [False, False, False],
		}
	```  
	This configuration defines a Lyapunov ANN composed of 2 hidden layers of 10 neurons each, with no bias, and with quadratic activation function on the first hidden layer. Make sure to leave the last entry of `size_layers` set to 1, as the output of the Lyapunov ANN is by definition a scalar.
	
	2. Select the structure of the control ANN:
	```python
    	# Parameters for control ANN
    	control_params = {
			'use_lin_ctr': True,  # use linear control law  -- defined as 'phi' in the publication
			'lin_contr_bias': False,  # use bias on linear control layer
			'control_initialised': False,  # initialised control ANN with pre-computed LQR law
			'init_control': torch.tensor([[-3.0, -1.7, 24.3]]),  # initial control solution
			'size_ctrl_layers': [50, 3],  # CAVEAT: the last entry is the number of control actions!
			'ctrl_bias': [True, False],
			'ctrl_activations': ['tanh', 'linear'],
			'use_saturation': False,        	# use saturations in the control law.
			'ctrl_sat': [18.3, 11.2, 4.6],		# actuator saturation values:
												# this vector needs to be as long as 'size_ctrl_layers[-1]' (same size as the control vector).
    	}
	```  
	This configuration defines a linear control ANN, with the control gain initialised to the value defined in `init_control`. If `control_initialised` was set to `False` instead, random initial weight would be seleced.  
	If a nonlinear control law is to be used instead, set `use_lin_ctr=False`; the nonlinear control law weight is by default initialised randomly.  
	The size of the control gain (both linear and nonlinear) is defined by the last entry of `size_ctrl_layers`; with the choice above, three control signals will be output.  
	If you intend to use a saturated control input, set `use_saturation=True`, and select the corresponding values of the input saturation with `ctrl_sat`.  
	  

4.  **Set up the configuration conditions for the *Falsifier***:
	1. Select the SMT domain boundaries and the additional configuration parameters as:
    	```python
    	    falsifier_params = {
				# a) SMT parameters
				'gamma_underbar': 0.1,  # domain lower boundary
				'gamma_overbar': 2.0,   # domain upper boundary
				'zeta_SMT': 200,  # how many points are added to the dataset after a CE box
								# is found
				'epsilon': 0.0,   # parameters to further relax the SMT check on the Lie derivative conditions.
								# default value = 0 (inspect utilities/Functions/CheckLyapunov for further info).
				
				# b) Discrete Falsifier parameters
				'grid_points': 50,  # sampling size grid
				'zeta_D': 50,       # how many points are added at each DF callback
			}
    	``` 
	  
5. *Start* the training by launching the `main_3d_template.py` script.  

6. The procedure *stops* when no *counterexample* is found or when a *timeout* is hit:
	1. the timeout refers to either the maximum number of learning iterations (`max_iters`) OR to the maximum time dReal is allowed to search for a solution. 
	A value of 180 seconds is suggested for the latter and it is currently implemented. To modify it, change the value of `@timeout(180, use_signals=True)` in file `utilities/Functions/CheckLyapunov()`. 
	 

