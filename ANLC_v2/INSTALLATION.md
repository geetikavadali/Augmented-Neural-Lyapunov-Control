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

  
### 1.1) Approach 1: install the requirements at system level (not recommended)
If Python3.7 is available on your machine, you can install the required packages at system level with:
```  
cd ANLC_v2
pip3 install -r ./requirements.txt  
```

If Python3.9 is available on your machine, you can install the required packages at system level with:
```  
cd ANLC_v2
pip3 install -r ./requirements_v39.txt  
```
Move to step 2).


### 1.2) Approach 2: clone the Anaconda environment
If [Anaconda](https://docs.anaconda.com/free/anaconda/install/) is installed on your system, you can clone the environment with: 

```
conda env create -f ANLC_v1/inter_platform/env_anlc.yml
conda activate env_anlc
```

(use `conda deactivate` upon completion.)

Move to step 2).



### 1.3) Approach 3: create a Python virtual environment
  
If Python3.7 or Python3.9 are installed on your system, the code can be run in a [virtual environment](https://docs.python.org/3/library/venv.html). Start as follows:
```
pip3 install virtualenv
python3 -m venv anlc_venv
source anlc_venv/bin/activate
python -V
```

If Python3.7:
```
pip3.7 install -r ANLC_v2/requirements.txt  
```

If Python3.9:
```
pip3.9 install -r ANLC_v2/requirements_v39.txt  
```

(use `deactivate` upon completion.)

Move to step 2).


### 2) Test *dreal* installation
Depending upon your OS, some of the instructions above were reported returning issues linked to the *dreal* library. To check whether your *dreal* installation is set up correctly, run:

```
python3
from dreal import *
```

If you experience any error at this stage, please install *dreal* with the most up to date instructions, available at: 
https://github.com/dreal/dreal4  
If you are using Ubuntu 20.04, for instance, you can amend the installation with:
```
sudo apt-get install curl
curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/20.04/install.sh | sudo bash
pip3 install dreal
```

### 3) Test ANLC successful installation
These commands will run the training for the *controlled Lorenz system*:
```
cd ANLC_v2/code
python3 main_3d_template.py  
```
Upon completion, you should expect to find in 'ANLC_v2/code/results/campaign_3000/0' figures such as:   
<img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/ANLC_v2/documentation/images/Lyapunov_function_example.png" width=30% height=30%>
