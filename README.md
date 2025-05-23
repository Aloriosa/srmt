# Shared Recurrent Memory Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Aloriosa/srmt/blob/main/LICENSE)

This work was done in collaboration of AIRI, DeepPavlov.ai, and London Institute for Mathematical Sciences.

## Installation

Create a conda environment from the export file:
```bash
conda env create -f srmt_env_export.yml
```
## Training

**SRMT** training is done with the `train.py` script.
To modify the default training configuration values, one can use the command line arguments corresponding to the names of variables in `srmt/training_config.py`. 

For example, SRMT with reward functions from the [paper](https://arxiv.org/abs/2501.13200) is trained as follows.

Sparse reward function:
```bash
python train.py --experiment=<name of the folder to store checkpoints> --attn_core=true --use_rnn=false --core_memory=true --const_reward=true --intrinsic_target_reward=0 --seed=<random seed>
```
Dense reward function:
```bash
python3 train.py --experiment=<name of the folder to store checkpoints> --attn_core=true --use_rnn=false --core_memory=true --const_reward=true --seed=<random seed>
```
Moving Negative reward function:
```bash
python3 train.py --experiment=<name of the folder to store checkpoints> --attn_core=true --use_rnn=false --core_memory=true --any_move_reward=true --seed=<random seed>
```
Directional reward function:
```bash
python3 train.py --experiment=<name of the folder to store checkpoints> --attn_core=true --use_rnn=false --core_memory=true --target_reward=true --positive_reward=true --intrinsic_target_reward=0.005 --seed=<random seed>
```
Directional Negative reward function:
```bash
python3 train.py --experiment=<name of the folder to store checkpoints> --attn_core=true --use_rnn=false --core_memory=true --target_reward=true --reversed_reward=true --seed=<random seed>
```


### Evaluation 
To evaluate the trained model on the test set of environments, use:
```bash
python eval.py
```

## Episode Visualization

To run a single episode with the trained **SRMT** agents and produce an animation:

```bash
python3 example.py
```

The animation will be stored in the folder containing the experiment checnkpoints.

To avoid performance issues, it is recommended to set the following environment variables restricting Numpy CPU threads to 1:

```bash
export OMP_NUM_THREADS="1" 
export MKL_NUM_THREADS="1" 
export OPENBLAS_NUM_THREADS="1"
```

## Citation

```
@misc{sagirova2025srmtsharedmemorymultiagent,
      title={SRMT: Shared Memory for Multi-agent Lifelong Pathfinding}, 
      author={Alsu Sagirova and Yuri Kuratov and Mikhail Burtsev},
      year={2025},
      eprint={2501.13200},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.13200}, 
}
```

## Acknowledgements

The repository is inspired by the [Learn to Follow](https://github.com/AIRI-Institute/learn-to-follow) repository.
