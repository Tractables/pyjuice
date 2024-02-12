<img align="right" width="180px" src="https://avatars.githubusercontent.com/u/58918144?s=200&v=4">

# PyJuice

[![CUDA CI Tests](https://github.com/Juice-jl/pyjuice/actions/workflows/ci_tests.yml/badge.svg?branch=main)](https://github.com/Juice-jl/pyjuice/actions/workflows/ci_tests.yml)
[![codecov](https://codecov.io/gh/Juice-jl/pyjuice/branch/main/graph/badge.svg?token=XpgPLYa2RQ)](https://codecov.io/gh/Juice-jl/pyjuice)

PyJuice is a library for [Probabilistic Circuits](https://starai.cs.ucla.edu/papers/ProbCirc20.pdf) (PCs) written in [PyTorch](https://github.com/pytorch/pytorch). It has code for inference (e.g., marginals, sampling) and learning (e.g., EM, pruning) in PCs, which can be either defined by hand or generated directly from pre-specified structures (e.g., [PD](https://arxiv.org/pdf/1202.3732.pdf), [RAT-SPN](https://proceedings.mlr.press/v115/peharz20a/peharz20a.pdf), [HCLT](https://proceedings.neurips.cc/paper_files/paper/2021/file/1d0832c4969f6a4cc8e8a8fffe083efb-Paper.pdf)).

## Why PyJuice?

The biggest advantage of PyJuice is its speed.

<table style="width:80%">
  <tr>
    <td></td>
    <td colspan="5", align="center"><b>PD</b></td>
    <td colspan="5", align="center"><b>RAT-SPN</b></td>
  </tr>
  <tr>
    <td># nodes</td>
    <td>172K</td>
    <td>344K</td>
    <td>688K</td>
    <td>1.38M</td>
    <td>2.06M</td>
    <td>58K</td>
    <td>116K</td>
    <td>232K</td>
    <td>465K</td>
    <td>930K</td>
  </tr>
  <tr>
    <td># edges</td>
    <td>15.6M</td>
    <td>56.3M</td>
    <td>213M</td>
    <td>829M</td>
    <td>2.03B</td>
    <td>616K</td>
    <td>2.2M</td>
    <td>8.6M</td>
    <td>33.4M</td>
    <td>132M</td>
  </tr>
  <tr>
    <td><b><a href="https://github.com/SPFlow/SPFlow">SPFlow</a></b></td>
    <td>$\geq\!25000$</td>
    <td>$\geq\!25000$</td>
    <td>$\geq\!25000$</td>
    <td>$\geq\!25000$</td>
    <td>$\geq\!25000$</td>
  </tr>
  <tr>
    <td><b><a href="https://github.com/cambridge-mlg/EinsumNetworks">EiNet</a></b></td>
    <td>$34.2_{\pm0.0}$</td>
    <td>$88.7_{\pm0.2}$</td>
    <td>$456.1_{\pm2.3}$</td>
    <td>$1534.7_{\pm0.5}$</td>
    <td>OOM</td>
  </tr>
</table>

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

3. Install GPU Enabled Pytorch `2.0`.  See [pytorch installation guide](https://pytorch.org/get-started/locally/) for more details. Make sure to install version `>=2.0`.

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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
