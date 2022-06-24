This repo contains source code for our paper "Model checking for Entanglment Swapping"  
Instructions to use this repo are as follows  
# Prerequisites  

---
OS : Ubuntu  
python>=3.8  
Java>=9  
# Creating a virtual environment  
We highly recommend using [virtual environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system.) while using this repo. Virtual environments helps to isolate versions of Python and associated pip packages, thus avoiding any dependency issues.  
Command to create a virtual environment named ``testenv``
```
python3 -m venv testenv
```
Command to activate the virtual environment
```
source testenv/bin/activate
```
# Installing PRISM model checker  

---
Installation procedure for PRISM can be found in their [website](http://prismmodelchecker.org/manual/InstallingPRISM/Instructions).
The same bash script for this is also added to ``prism`` directory for easy access.  Inside prism directory, run  

```
bash install.sh
```
PATH_TO_PRISM variable in run.sh shall be changed if the installation is done not using our script.  

# Installing SeQUeNCe simulator  

---
[SeQUeNCe](https://sequence-toolbox.github.io/) simulator from the original project is reconfigured to suit our requirements. The modified code is given in ``SeQUeNCe`` directory.  
To install, go to SeQUeNCe directory and run the following commands. 
```
pip3 install -r requirements.txt  
pip3 install .  
```

# Running the experiments  
We provide a bash script to run all the experiments. As mentioned above, verify that the PATH_TO_PRISM variable points to bin of PRISM installed.  
Then run the following command  
```
bash run.sh
```
On a normal computer with 8gb RAM and i7 10th Gen processor, it takes around 90 min to complete the whole thing.  
Majority of the time is needed for simulator part as we each experiment is run for 2000 trails to mimic the probabilistic behaviour of the model.  
Once complete two directories are generated ``results`` and ``figures``  
- ``results`` directory contains all the probability values in csv formats seperated into different files for model and simulator  
- ``figures`` directory contains the six plots shown in the paper. These are named as 6a-6f as given in the paper.
