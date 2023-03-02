# pyjuice

[![CUDA CI Tests](https://github.com/Juice-jl/pyjuice/actions/workflows/ci_tests.yml/badge.svg?branch=main)](https://github.com/Juice-jl/pyjuice/actions/workflows/ci_tests.yml)
[![codecov](https://codecov.io/gh/Juice-jl/pyjuice/branch/main/graph/badge.svg?token=XpgPLYa2RQ)](https://codecov.io/gh/Juice-jl/pyjuice)

Probabilistic Circuits Package

## Installation

For now need to manually install pytorch's nightly build (cuda).

0. (Optional) Make a new conda environment

    ```bash
    conda create -n pyjuice python=3
    conda activate pyjuice
    ```

1. Clone this repository and `cd` into it.

2. Install the `pyjuice` package in developement mode, run the following:

    ```bash
    pip install --editable .
    ```

3. Install Pytorch Nightly Build

    ```bash
    pip3 install -I torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
    ```

## Testing

- Install `pytest`:

    ```bash
    pip install pytest
    ```

- To run the tests, simply call:

    ```bash
    pytest
    ```
