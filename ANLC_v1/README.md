# Augmented-Neural-Lyapunov-Control
This repository contains the code for the paper:  
**Augmented Neural Lyapunov Control** (ANLC)   
  
The work can be read open-access on the [IEEE webpage](https://ieeexplore.ieee.org/document/10171339).   
  
  
## Scope of the code
ANCL is a software framework to **automatically** synthesise:  
1. a **stabilising controller** for a desired equilibrium of a nonlinear system;  
2. a **Control Lyapunov Function**(CLF) to certify its stability.    
  
The code is based on a loop between a *Learner* and a *Falsifier*. Starting from a finite set of state-space samples, the *Learner* trains two Neural Networks, one representing a control law and the other a CLF. In parallel, the *Falsifier* is tasked with verifying whether the candidate CLF satisfies the theoretical Lyapunov conditions within a domain of Reals. If the conditions are satisifed, the learning is halted and the resulting control law and CLF are returned. On the contrary, if the conditions are not satisfied, the Falsifier returns a set of points (denoted as counterexample) where the Lyapunov conditions are violated. These points are added to the dataset and the learning process is further iterated.    
   
A schematic of the learning architecture is hereby illustrated:  
<img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/ANLC_v1/documentation/images/learning_scheme.png" width=100% height=100%>


This repository was initially forked from the original **Neural Lyapunov Control** (NLC):  
- [Neural-Lyapunov-Control](https://github.com/YaChienChang/Neural-Lyapunov-Control)  
  
With respect to the NLC, the key ANLC upgrades consist of:
- [x] Option to synthesised both linear and nonlinear control laws;  
- [x] Synthesis of control laws without requiring the initialisation of the control weights.
  
## Installation

The code was tested on OS: Linux Ubuntu 20.04  

The most up-to-date installation intructions are available at:
- [INSTALLATION](https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/ANLC_v2/INSTALLATION.md)  
  
 
## How to use me
To test the synthesis of control and Lyapunov functions with your own dynamics, a step-by-step example is illustrated in the ![wiki_example](./documentation/WIKI.md) file.
  
  
## Published results
The results reported in the reference publication [1] are reported in the subdirectory:
![/results_publication](./results_publication)
and logged in the file:
 ![log_results](./results_publication/README_log_results.md).  
To reproduce the results, the following scripts can be run:
1. NLC results:  ![NLC-inverted pendulum main](./code/sim_campaign_NLC_main.py) (requires Python3.7)
2. ANLC results:  ![ANLC-inverted pendulum main](./code/sim_campaign_ANLC_main.py) 
3. ANLC Lorenz system results: ![ANLC-Lorenz main](./code/sim_campaign_3d_Lorenz_main.py) 


## Framework limitations
1. The results presented in [1] cover 2- and 3-dimensional (nonlinear) systems. 
The Falsifier, representing the bottleneck of this procedure, is known to scale poorly as the system dimension increases.  
The code currently supports:
- [x] 1-dimensional systems  
- [x] 2-dimensional systems  
- [x] 3-dimensional systems  
- [x] 4-dimensional systems  
- [ ] >= 5-dimensional systems  
To scale up to higher dimensional systems, additional **Discrete Falsifier** functions (included within `utilities/Function.py`) need to be developed.
As reference, use `utilities/Function/AddLieViolationsOrder3_v4`.
  
2. The released framework can currently stabilise systems around the state-space *origin* only. An updated version, allowing users to specify non-trivial equilibria, will be released. 
- [ ] Stabilising generic equilibria (not necessarily the origin)
  
  
## Code -- known issues
1) the `ipython` module is known to have compatibility issues with some IDEs such as PyCharm. Try by selecting (within the `sim_campaign_main.py` file): 
```  
clear_ws = False  
```
If the issue is not fixed, remove completely the `ipython`-related code from the `sim_campaign_main.py` file, as it is not critical.


## Reference
The article can be accessed open-access on the [IEEE webpage](https://ieeexplore.ieee.org/document/10171339/).
  
This work can be cited with the following BibTeX entry:  

```bibtex
@article{grande2023augmented,
  title={Augmented Neural Lyapunov Control},
  author={Grande, Davide and Peruffo, Andrea and Anderlini, Enrico and Salavasidis, Georgios},
  journal={IEEE Access},
  volume={11},
  pages={67979--67986},
  year={2023},
  publisher={IEEE},
  doi={10.1109/ACCESS.2023.3291349}
}
``` 

