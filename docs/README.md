# jaxrl5

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/ikostrikov/jaxrl5/tree/main.svg?style=svg&circle-token=668374ebe0f27c7ee70edbdfbbd1dd928725c01a)](https://dl.circleci.com/status-badge/redirect/gh/ikostrikov/jaxrl5/tree/main) [![codecov](https://codecov.io/gh/ikostrikov/jaxrl5/branch/main/graph/badge.svg?token=Q5QMIDZNZ3)](https://codecov.io/gh/ikostrikov/jaxrl5)

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
 
- For running Franka experiments from vision, please also `git clone` and `pip install -e .` this library https://github.com/Leo428/efficientnet-jax. It enables high UTD learning from pixels by using pre-trained encoders.

## Examples

[Here.](examples/)

## Tests

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES= pytest tests
```
