# pyjuice

[![CUDA CI Tests](https://github.com/Juice-jl/pyjuice/actions/workflows/ci_tests.yml/badge.svg?branch=main)](https://github.com/Juice-jl/pyjuice/actions/workflows/ci_tests.yml)
[![codecov](https://codecov.io/gh/Juice-jl/pyjuice/branch/main/graph/badge.svg?token=XpgPLYa2RQ)](https://codecov.io/gh/Juice-jl/pyjuice)

Probabilistic Circuits Package

## Installation

0. (Optional) Make a new conda environment

    ```bash
    conda create -n pyjuice python=3.8
    conda activate pyjuice
    ```

1. Clone this repository and `cd` into it.

2. Install the `pyjuice` package in developement mode, run the following:

    ```bash
    pip install --editable .
    ```

3. Install Pytorch `2.0`.  See [pytorch installation guide](https://pytorch.org/get-started/locally/) for more details. Make sure to install version `>=2.0`.

    ```bash
    pip3 install torch torchvision
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
