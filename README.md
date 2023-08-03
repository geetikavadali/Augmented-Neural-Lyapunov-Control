# Augmented-Neural-Lyapunov-Control
This repository contains the code for the paper:  
**Augmented Neural Lyapunov Control** (ANLC)  
  
The work can be read open-access on the [IEEE webpage](https://ieeexplore.ieee.org/document/10171339).  
  
  
## Scope of the code
ANCL is a software framework to **automatically** synthesise:  
1. a **stabilising controller** for a desired equilibrium of a nonlinear system;  
2. a **Control Lyapunov Function** (CLF) to certify its stability.    
  
The code is based on a loop between a *Learner* and a *Falsifier*. Starting from a finite set of state-space samples, the *Learner* trains two Neural Networks, one representing a control law and the other a CLF. In parallel, the *Falsifier* is tasked with verifying whether the candidate CLF satisfies the theoretical Lyapunov conditions within a domain of Reals. If the conditions are satisifed, the learning is halted and the resulting control law and CLF are returned. On the contrary, if the conditions are not satisfied, the Falsifier returns a set of points (denoted as counterexample) where the Lyapunov conditions are violated. These points are added to the dataset and the learning process is further iterated.    
   
A schematic of the learning architecture is hereby illustrated:  
<img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/ANLC_v2/documentation/images/learning_scheme.png" width=100% height=100%>


This repository was initially branched from the original **Neural Lyapunov Control** (NLC):  
- [Neural-Lyapunov-Control](https://github.com/YaChienChang/Neural-Lyapunov-Control)  
  
With respect to the NLC, the key ANLC upgrades consist of:
- [x] Option to synthesised both linear and nonlinear control laws;  
- [x] Control functions (linear and nonlinear) synthesised without initialising the control weights;
- [x] Option to stabilise generic equilibria (not necessarily the origin).  

## Repository structure  
The repository contains two versions of the code:    
- [x] ![ANLC_v1](./ANLC_v1): v1 is used to generate the results of the paper;    
- [x] ![ANLC_v2](./ANLC_v2): v2 is the version actively developed and the one *recommended*. With respect to v1, v2 entails:
	- [x] Refactored and optimised code;  
	- [x] Option to stabilise generic equilibria (not necessarily the origin).  
  
## Step-by-step guide
To synthesise control and Lyapunov functions for your own dynamics, a step-by-step example is described in the ![WIKI](./ANLC_v2/documentation/WIKI.md) file.  
  
Results, encompassing 3D visualisation plots as follows, will be generated:  

Lyapunov function example |  Lie derivative function example                    
:-------------------------:|:-------------------------:
<img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/ANLC_v2/documentation/images/V_3D.gif"> | <img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/ANLC_v2/documentation/images/Lie_der_3D.gif">
  
  
## Installation  
Instructions on installation are available within the ![INSTALLATION](./ANLC_v2/INSTALLATION.md/) file.    


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

