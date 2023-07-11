# Augmented-Neural-Lyapunov-Control
This repository contains the code for the paper:  
**Augmented Neural Lyapunov Control** (ANLC)  
  
## Scope of the code
ANCL is a software framework to **automatically** synthesise:  
1. a **stabilising controller** for a desired equilibrium of a nonlinear system;  
2. a **Lyapunov function** to certify its stability.    
  
The code is based on a loop between a *Learner* and a *Falsifier*. The Learner trains a control and a Lyapunov function, both represented by a Neural Network.    
The Falsifier takes candidate Lyapunov Functions and verifies whether they satisfy the Lyapunov conditions over the desired domain (over the Reals). If the conditions are not satisfied, the Falsifier returns several points (denoted as counterexample) where there conditions are violated. If that occurrs, the new counterexamples are added to the dataset and the learning procedure is iterated.  
A schematic of the learning architecture is hereby illustrated:  
<img src="https://github.com/dave-ai/test_deplo1/blob/master/ANLC_v1/documentation/images/ANN_architecture.png" width=50% height=20%>


This repository was initially branched from the original **Neural Lyapunov Control** (NLC):  
- [Neural-Lyapunov-Control](https://github.com/YaChienChang/Neural-Lyapunov-Control)  
  
With respect to the NLC, the key ANLC upgrades consist of:
- [x] Option to synthesised both linear and nonlinear control laws;  
- [x] Control functions (linear and nonlinear) synthesised without initialising the control weights.
  

## Repository structure  
The repository contains two versions of the code:    
- [x] ![ANLC_v1](./ANLC_v1): v1 is used to generate the results of the paper;    
- [x] ![ANLC_v2](./ANLC_v2): v2 is the version actively developed. With respect to v1, the code was refactored and optimised (-soon to be released-).  
  
  
## Installation  
Instructions on installation are available within the ![README](./ANLC_v1/README.md/).    


## Reference  
The code of this repository can be cited with the following BibTeX entry:  

```bibtex
@article{grande2023augmented,
  title={Augmented Neural Lyapunov Control},
  author={Davide Grande, Andrea Peruffo, Enrico Anderlini, Georgios Salavasidis},
  journal={IEEE Access},
  year={2023}
}
``` 

