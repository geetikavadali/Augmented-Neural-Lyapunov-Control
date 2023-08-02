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
  
  
## Installation  
Instructions on installation are available within the ![INSTALLATION](INSTALLATION.md) file.    
  
## Step-by-step guide
To synthesise control and Lyapunov functions for your own dynamics, a step-by-step example is reported in the ![WIKI](./documentation/WIKI.md) file.  
  
## Library architecture
The library architecture is composed of three main modules:  
    
1. the main file loading the configuration and the system dynamics;  
2. the CEGIS file training the ANNs and calling the Falsifier;  
3. the postprocessing subroutines to plot the synthesised functions and to run the closed-loop tests;  
  
The architecture is summarised as:  
<img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/ANLC_v2/documentation/images/code_modules.png" width=80% height=80%>

More in detail, the *library call graph* is illustrated as:  
<img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/ANLC_v2/documentation/images/call_graph.png" width=100% height=100%>  
  
## Framework limitations
1. The results presented in [1] cover 2- and 3-dimensional (nonlinear) systems. 
The Falsifier, representing the bottleneck of this procedure, is known to scale poorly as the system dimension increases.  
The code currently supports:
- [x] 1-dimensional systems  
- [x] 2-dimensional systems  
- [x] 3-dimensional systems  
- [x] 4-dimensional systems  
- [ ] >= 5-dimensional systems  
To scale up to higher dimensional systems, additional **Discrete Falsifier** functions need to be developed.
As reference, use `utilities/Function/AddLieViolationsOrder3_v4`.
  
## Reference  
The article can be accessed open-access on the [IEEE webpage](https://ieeexplore.ieee.org/document/10171339/).
  
This work can be cited with the following BibTeX entry:  

```bibtex
@article{grande2023augmented,
  title={Augmented Neural Lyapunov Control},
  author={Grande, Davide and Peruffo, Andrea and Anderlini, Enrico and Salavasidis, Georgios},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE},
  doi={10.1109/ACCESS.2023.3291349}
}
``` 

