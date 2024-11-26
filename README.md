# Basal ganglia models

A repository for my BSc thesis "Reinforcement learning in artificial neural network inspired by dopaminergic pathway" @ UW.


## Setup

For the sake of reproducability as well as ease of development for cross-platform usage (local and cluster) we use conda to manage enviorments. 

### Local 

1. Install miniconda if you don't have it yet:
    https://docs.anaconda.com/miniconda/install/

2. Create a conda enviorement with following command:
    ```
    conda create --name <env_name> python=3.11
    ```
    For example:
    ```
    conda create --name bg python=3.11
    ```
3. Activate the enviorerment: 
    ```
    conda activate <env_name>
    ```
4. Install PyPi dependencies: 
    ```
    pip install -r requirements.txt
    ```
5. If you have completed all the steps successfully you should be able to run a benchmark:
    ```
    python3 atari_dqn.py --task "PongNoFrameskip-v4" --batch-size 64
    ```

### CLuster (plgrid - athena)
TBC.

