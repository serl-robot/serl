# SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning

## Installation

- Recommended:

    - Assume the machines have the lastest Nvdia drivers and CUDA Versions (either 12.1 or 11.x)
    - Run
        ```bash
        pip install --upgrade pip

        pip install -e .
        ```

        ```
        # CUDA 12 installation
        # Note: wheels only available on linux.
        pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

        # CUDA 11 installation
        # Note: wheels only available on linux.
        pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```

- Check [here](https://github.com/google/jax#installation) for JAX installation with local CUDA and CUDNN installations,
    - This way can be more complicated.
 
- For running experiments from vision, please also `git clone` and `pip install -e .` this library https://github.com/Leo428/efficientnet-jax. It is forked from https://github.com/rwightman/efficientnet-jax to support learning with pre-trained visual encoders (EfficientNet and MobileNets) in JAX and Flax.

## Examples

[This folder contains example usages of serl as in the paper.](serl_examples/)

