# Augmented-Neural-Lyapunov-Control
This repository contains the code for the paper:  
**Augmented Neural Lyapunov Control** (ANLC)  
  
## Scope of the code
ANCL is a software framework to **automatically** synthesise:  
1. a **stabilising controller** for a desired equilibrium of a nonlinear system;  
2. a **Control Lyapunov Function**(CLF) to certify its stability.    
  
The code is based on a loop between a *Learner* and a *Falsifier*. The Learner trains both a control and a Lyapunov function, each represented by a Neural Network.    
The Falsifier verifies whether the candidate CLFs satisfy the theoretical Lyapunov conditions within a specified domain (over the Reals). If the conditions are not satisfied, the Falsifier returns several points (denoted as counterexample) where the Lyapunov conditions are violated. Next, the new counterexamples are added to the dataset and the learning procedure is iterated.  
A schematic of the learning architecture is hereby illustrated:  
<img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/documentation/images/ANN_architecture.png" width=50% height=20%>


This repository was initially forked from the original **Neural Lyapunov Control** (NLC):  
- [Neural-Lyapunov-Control](https://github.com/YaChienChang/Neural-Lyapunov-Control)  
  
With respect to the NLC, the key ANLC upgrades consist of:
- [x] Option to synthesised both linear and nonlinear control laws;  
- [x] Control functions (linear and nonlinear) synthesised without initialising the control weights.
  
## Installation

The code was tested on OS: Linux Ubuntu 20.04  
Three methods are hereby provided to install the following main dependencies:

- Python3.7 or Python3.9
- [dReal4: v4.21.6.2](https://github.com/dreal/dreal4)
- [PyTorch: 1.4.0 or 1.7.1](https://pytorch.org/get-started/locally/)
- [Numpy: 1.21.5](https://pytorch.org/get-started/locally/)
  
First, clone the repository:
```
git clone https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control.git
cd Augmented-Neural-Lyapunov-Control
```

  
### Approach 1: install the requirements at system level (not recommended)
If Python3.7 is available on your machine, you can install the required packages at system level with:
```  
pip3 install -r ./requirements.txt  
```

If Python3.9 is available on your machine, you can install the required packages at system level with:
```  
pip3 install -r ./requirements_v39.txt  
```


### Approach 2: clone the Anaconda environment
If [Anaconda](https://docs.anaconda.com/free/anaconda/install/) is installed on your system, you can clone the environment with: 

```
conda env create -f ./inter_platform/env_anlc.yml
conda activate env_anlc
```

(use `conda deactivate` upon completion.)


### Approach 3: create a Python virtual environment
  
If Python7 or Python9 are installed on your system, the code can be run in a [virtual environment](https://docs.python.org/3/library/venv.html). Start as follows:
```
pip3 install virtualenv
python3 -m venv anlc_venv
source anlc_venv/bin/activate
python -V
```

If Python7:
```
pip3.7 install -r ./requirements.txt  
```

If Python9:
```
pip3.9 install -r ./requirements_v39.txt  
```

(use `deactivate` upon completion.)

### Test succesful installation
These commands will run the training for the *controlled Lorenz system*:
```
cd code
python3 sim_campaign_3d_Lorenz_main.py  
```
Upon completion, you should expect to find in the 'code/sim_campaign/results/campaign_1475/0' figures such as:   
<img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/documentation/images/Lyapunov_function_example.png" width=50% height=20%>
  
 
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
The code of this repository can be cited with the following BibTeX entry:

<a id="1">[1]</a> 
```bibtex
@article{grande2023augmented,
  title={Augmented Neural Lyapunov Control},
  author={Davide Grande, Andrea Peruffo, Enrico Anderlini, Georgios Salavasidis},
  journal={IEEE Access},
  year={2023}
}
``` 

