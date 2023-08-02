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
  
If Python3.7 or Python3.9 are installed on your system, the code can be run in a [virtual environment](https://docs.python.org/3/library/venv.html). Start as follows:
```
pip3 install virtualenv
python3 -m venv anlc_venv
source anlc_venv/bin/activate
python -V
```

If Python3.7:
```
pip3.7 install -r ./requirements.txt  
```

If Python3.9:
```
pip3.9 install -r ./requirements_v39.txt  
```

(use `deactivate` upon completion.)

### Test succesful installation
These commands will run the training for the *controlled Lorenz system*:
```
cd code
python3 main_3d_template.py  
```
Upon completion, you should expect to find in 'code/results/campaign_3000/0' figures such as:   
<img src="https://github.com/grande-dev/Augmented-Neural-Lyapunov-Control/blob/master/ANLC_v2/documentation/images/Lyapunov_function_example.png" width=30% height=30%>
