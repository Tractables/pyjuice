# pyjuice

Probabilistic Circuits Package

## Installation

For now need to manually install pytorch's nightly build (cuda).

0. Clone this repository and `cd` into it.

1. Install the `pyjuice` package in developement mode, run the following:

    ```bash
    pip install --editable .
    ```

2. Install Pytorch Nightly Build

    ```bash
    pip3 install -I torch --index-url https://download.pytorch.org/whl/nightly/cu118
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
