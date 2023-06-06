import pyjuice as juice
import pyjuice.nodes.distributions as dists
import multiprocessing as mp

import pytest


def compilation_speed_test():
    num_latents = 1024
    num_cats = 512
    num_vars = 32

    curr_zs = juice.inputs(0, num_latents, dists.Categorical(num_cats = num_cats))

    for var in range(1, num_vars):
        curr_xs = juice.inputs(var, num_latents, dists.Categorical(num_cats = num_cats))
        ns = juice.summate(curr_zs, num_nodes = num_latents)
        curr_zs = juice.multiply(curr_xs, ns)

    ns = juice.summate(curr_zs, num_nodes = 1)

    print("> Start compilation...")
    pc = juice.TensorCircuit(ns)
    print("> Done")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    compilation_speed_test()