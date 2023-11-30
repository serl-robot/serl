# SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning

## Installation
- Conda Environment:
    - create an environment with `conda create -n serl_dev python=3.10`

## Launcher
- The launcher module is in charge of lauching learner and actor on seperate processes and even on separate machines.
- `cd serl_launcher` and `pip install -e .`

## RL library
- For GPU:
```
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

- For TPU
```
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
- See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

- Then `cd` back to the root serl path, then run `pip install -e .`

## Robot Infra
- please follow README.md under robot_infra folder

## Examples

[This folder contains example usages of serl as in the paper.](serl_examples/)

