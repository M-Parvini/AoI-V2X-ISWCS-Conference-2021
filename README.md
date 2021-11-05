 # AoI-V2X-ISWCS-Conference-2021

Simulation code of the paper:
    "AoI Aware Radio Resource Management of Autonomous Platoons via Multi Agent Reinforcement Learning"

Written by  : Mohammad Parvini, M.Sc. student at Tarbiat Modares University.

### If you want to cite: 
>M. Parvini, M. R. Javan, N. Mokari, B. A. Arand and E. A. Jorswieck, "AoI Aware Radio Resource Management of Autonomous Platoons via Multi Agent Reinforcement Learning," 2021 17th International Symposium on Wireless Communication Systems (ISWCS), 2021, pp. 1-6, doi: 10.1109/ISWCS49558.2021.9562190.
---------------------------------------------------------------------------------------
* We have built our simulation following the urban case defined in Annex A of 
     3GPP, TS 36.885, "Study on LTE-based V2X Services".

* The vehicular simulation environment of this paper is coded on top of the simulation environment designed by **_[Le Liang](https://github.com/le-liang)_** from Southeast University. 

* The twin delayed deep deterministic policy gradient (TD3) framework is built on top of the materials developed by  **_[Phil Tabor](https://github.com/philtabor)_**, Physicist, and Machine Learning Engineer.
---------------------------------------------------------------------------------------
### prerequisites:
* python 3.7 or higher
* PyTorch 1.7 or higher + CUDA
* It is recommended that the latest drivers be installed for the GPU.
---------------------------------------------------------------------------------------

In order to run the code:

* Please make sure that you have created the following directories:
	1) ...\Classes\tmp\ddpg
	2) ...\model\marl_model

The final results and the network weights will be saved in these directories.


1- Change the number of vehicles, platoon sizes, and intra-platoon distance

2- Once you run the code, simulation results will be saved into the directory: 
   ...\model\marl_model. You can import these data wherever you want (Matlab, python, etc.) 
   and plot the results. Furthermore, the weights of the neural networks will be saved into 
   the directory: ...\Classes\tmp\ddpg. 

Instructions: 

Please make sure that the following prerequisites are met:

python 3.7 or higher

PyTorch 1.7 or higher + CUDA

It is recommended that the latest drivers be installed for the GPU.

In order to run the code:

***

Please make sure that you have created the following directories:

    1) ...\Classes\tmp\ddpg

    2) ...\model\marl_model

The final results and the network weights will be saved in these directories.
 
***
