# Shared Recurrent Memory Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Aloriosa/srmt/blob/main/LICENSE)

This work was done in collaboration of AIRI, DeepPavlov.ai, and London Institute for Mathematical Sciences.

## Installation

Create a conda environment from the export file:
```bash
conda env create -f srmt_env_export.yml
```
## Training

To train **SRMT** from scratch, run:

```bash
python train.py
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


## Acknowledgements

The repository is inspired by the [Learn to Follow](https://github.com/AIRI-Institute/learn-to-follow) repository.
