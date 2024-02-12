<img align="right" width="180px" src="https://avatars.githubusercontent.com/u/58918144?s=200&v=4">

# PyJuice

[![CUDA CI Tests](https://github.com/Juice-jl/pyjuice/actions/workflows/ci_tests.yml/badge.svg?branch=main)](https://github.com/Juice-jl/pyjuice/actions/workflows/ci_tests.yml)
[![codecov](https://codecov.io/gh/Juice-jl/pyjuice/branch/main/graph/badge.svg?token=XpgPLYa2RQ)](https://codecov.io/gh/Juice-jl/pyjuice)

PyJuice is a library for [Probabilistic Circuits](https://starai.cs.ucla.edu/papers/ProbCirc20.pdf) (PCs) written in [PyTorch](https://github.com/pytorch/pytorch). It has code for inference (e.g., marginals, sampling) and learning (e.g., EM, pruning) in PCs, which can be either [defined by hand](https://github.com/Juice-jl/pyjuice#example-usage-define-your-own-pc) or generated directly [from pre-specified structures](https://github.com/Juice-jl/pyjuice#example-usage-pre-specified-structures) (e.g., [PD](https://arxiv.org/pdf/1202.3732.pdf), [RAT-SPN](https://proceedings.mlr.press/v115/peharz20a/peharz20a.pdf), [HCLT](https://proceedings.neurips.cc/paper_files/paper/2021/file/1d0832c4969f6a4cc8e8a8fffe083efb-Paper.pdf)).

## Why PyJuice?

The biggest advantage of PyJuice is its speed and scalability. We benchmark PyJuice against prior PC packages [SPFlow](https://github.com/SPFlow/SPFlow), [EiNet](https://github.com/cambridge-mlg/EinsumNetworks), and [Juice.jl](https://github.com/Juice-jl/ProbabilisticCircuits.jl) on the [PD](https://arxiv.org/pdf/1202.3732.pdf) and [HCLT](https://proceedings.neurips.cc/paper_files/paper/2021/file/1d0832c4969f6a4cc8e8a8fffe083efb-Paper.pdf) structures with various sizes by variating their width. We report the average ($\pm$ standard deviation of 5 runs) runtime (in seconds) per training epoch of 60K samples. All experiments were carried out on an RTX 4090 GPU with 24GB memory. To maximize parallelism, we always use the maximum possible batch size. "OOM" denotes out-of-memory with batch size 2.

<table>
  <tr>
    <td></td>
    <td colspan="5", align="center"><b><a href="https://arxiv.org/pdf/1202.3732.pdf">PD</a></b></td>
  </tr>
  <tr>
    <td># nodes</td>
    <td>172K</td>
    <td>344K</td>
    <td>688K</td>
    <td>1.38M</td>
    <td>2.06M</td>
  </tr>
  <tr>
    <td># edges</td>
    <td>15.6M</td>
    <td>56.3M</td>
    <td>213M</td>
    <td>829M</td>
    <td>2.03B</td>
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
  <tr>
    <td><b><a href="https://github.com/Juice-jl/ProbabilisticCircuits.jl">Juice.jl</a></b></td>
    <td>$12.6_{\pm0.5}$</td>
    <td>$37.0_{\pm1.7}$</td>
    <td>$141.7_{\pm6.9}$</td>
    <td>OOM</td>
    <td>OOM</td>
  </tr>
  <tr>
    <td><b>PyJuice</b></td>
    <td>$2.0_{\pm0.0}$</td>
    <td>$5.3_{\pm0.0}$</td>
    <td>$15.4_{\pm0.0}$</td>
    <td>$57.1_{\pm0.2}$</td>
    <td>$203.7_{\pm0.1}$</td>
  </tr>
</table>

<table>
  <tr>
    <td></td>
    <td colspan="5", align="center"><b><a href="https://proceedings.neurips.cc/paper_files/paper/2021/file/1d0832c4969f6a4cc8e8a8fffe083efb-Paper.pdf">HCLT</a></b></td>
  </tr>
  <tr>
    <td># nodes</td>
    <td>89K</td>
    <td>178K</td>
    <td>355K</td>
    <td>710K</td>
    <td>1.42M</td>
  </tr>
  <tr>
    <td># edges</td>
    <td>2.56M</td>
    <td>10.1M</td>
    <td>39.9M</td>
    <td>159M</td>
    <td>633M</td>
  </tr>
  <tr>
    <td><b><a href="https://github.com/SPFlow/SPFlow">SPFlow</a></b></td>
    <td>$22955.6_{\pm18.4}$</td>
    <td>$\geq\!25000$</td>
    <td>$\geq\!25000$</td>
    <td>$\geq\!25000$</td>
    <td>$\geq\!25000$</td>
  </tr>
  <tr>
    <td><b><a href="https://github.com/cambridge-mlg/EinsumNetworks">EiNet</a></b></td>
    <td>$52.5_{\pm0.3}$</td>
    <td>$77.4_{\pm0.4}$</td>
    <td>$233.5_{\pm2.8}$</td>
    <td>$1170.7_{\pm8.9}$</td>
    <td>$5654.3_{\pm17.4}$</td>
  </tr>
  <tr>
    <td><b><a href="https://github.com/Juice-jl/ProbabilisticCircuits.jl">Juice.jl</a></b></td>
    <td>$4.7_{\pm0.2}$</td>
    <td>$6.4_{\pm0.5}$</td>
    <td>$12.4_{\pm1.3}$</td>
    <td>$41.1_{\pm0.1}$</td>
    <td>$143.2_{\pm5.1}$</td>
  </tr>
  <tr>
    <td><b>PyJuice</b></td>
    <td>$0.8_{\pm0.0}$</td>
    <td>$1.3_{\pm0.0}$</td>
    <td>$2.6_{\pm0.0}$</td>
    <td>$8.8_{\pm0.0}$</td>
    <td>$24.9_{\pm0.1}$</td>
  </tr>
</table>

As indicated by the tables, PyJuice is not only much faster than existing implementations, but it is also more scalable -- it can train much larger PCs in a reasonable amount of time without suffering from OOM issues.

## Installation

Since PyJuice is in active development, we recommend installing it from the latest development branch:

```
git clone git@github.com:Juice-jl/pyjuice.git
cd pyjuice
pip install -e .
```

## Example Usage (pre-specified structures)



## Example Usage (define your own PC)


