import pyjuice as juice
import torch
import time
from pyjuice.model.backend import em_par_update, eval_top_down_probs, compute_cum_par_flows

import pytest


@pytest.mark.slow
def hmm_optim_speed_test():

    seq_length = 128
    num_latents = 1024
    vocab_size = 50257

    device = torch.device("cuda:0")

    root_ns = juice.structures.HMM(
        seq_length = seq_length,
        num_latents = num_latents,
        num_emits = vocab_size,
        homogeneous = True
    )

    pc = juice.compile(root_ns)
    pc.to(device)

    data = torch.randint(0, 50257, (64, seq_length)).to(device)

    lls = pc(data)
    pc.backward(data, flows_memory = 1.0, allow_modify_flows = False,
                propagation_alg = "LL", logspace_flows = True)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(40):
        lls = pc(data)
        pc.backward(data, flows_memory = 1.0, allow_modify_flows = False,
                    propagation_alg = "LL", logspace_flows = True)
    torch.cuda.synchronize()
    t1 = time.time()

    tdp_ms = (t1 - t0) / 40 * 1000

    print(f"Running forward + backward on average takes {tdp_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 94.181ms.")
    print("--------------------------------------------------------------")

    cum_flow_val = pc._cum_flow

    pc._cum_flow = cum_flow_val
    pc.mini_batch_em(step_size = 0.1, pseudocount = 1e-6, step_size_rescaling = True)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        pc._cum_flow = cum_flow_val
        pc.mini_batch_em(step_size = 0.1, pseudocount = 1e-6, step_size_rescaling = True)
    torch.cuda.synchronize()
    t1 = time.time()

    tdp_ms = (t1 - t0) / 100 * 1000

    print(f"Running Anemone on average takes {tdp_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 55.378ms.")
    print("--------------------------------------------------------------")

    # Update input layers
    for layer in pc.input_layer_group:
        layer.mini_batch_em(step_size = 0.1, pseudocount = 1e-6)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        for layer in pc.input_layer_group:
            layer.mini_batch_em(step_size = 0.1, pseudocount = 1e-6)
    torch.cuda.synchronize()
    t1 = time.time()

    tdp_ms = (t1 - t0) / 100 * 1000

    print(f"Running input EM on average takes {tdp_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 5.253ms.")
    print("--------------------------------------------------------------")

    # Normalize and update parameters
    em_par_update(pc.params, pc.param_flows, pc.par_update_kwargs, 
                  step_size = 0.1, pseudocount = 1e-6,
                  keep_zero_params = False)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        em_par_update(pc.params, pc.param_flows, pc.par_update_kwargs, 
                      step_size = 0.1, pseudocount = 1e-6,
                      keep_zero_params = False)
    torch.cuda.synchronize()
    t1 = time.time()

    tdp_ms = (t1 - t0) / 100 * 1000

    print(f"Normalize + update parameter accumulation on average takes {tdp_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 0.245ms.")
    print("--------------------------------------------------------------")

    # Evaluate TDP
    eval_top_down_probs(pc, update_pflow = True, scale = (1.0 - 0.1))

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        eval_top_down_probs(pc, update_pflow = True, scale = (1.0 - 0.1))
    torch.cuda.synchronize()
    t1 = time.time()

    tdp_ms = (t1 - t0) / 100 * 1000

    print(f"Evaluate TDP on average takes {tdp_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 55.511ms.")
    print("--------------------------------------------------------------")

    pc.init_param_flows(flows_memory = 0.9)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        pc.init_param_flows(flows_memory = 0.9)
    torch.cuda.synchronize()
    t1 = time.time()

    tdp_ms = (t1 - t0) / 100 * 1000

    print(f"Initialize parameter flows on average takes {tdp_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 7.164ms.")
    print("--------------------------------------------------------------")

    def process():
        pc.init_param_flows(flows_memory = 0.9)

        eval_top_down_probs(pc, update_pflow = True, scale = (1.0 - 0.1), use_cudagraph = True)

        # Update input layers
        for layer in pc.input_layer_group:
            layer.mini_batch_em(step_size = 0.1, pseudocount = 1e-6)

        # Accumulate parameter flows of tied nodes
        compute_cum_par_flows(pc.param_flows, pc.parflow_fusing_kwargs)

        # Normalize and update parameters
        em_par_update(pc.params, pc.param_flows, pc.par_update_kwargs, 
                      step_size = 0.1, pseudocount = 1e-6,
                      keep_zero_params = False)

    process()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        process()
    torch.cuda.synchronize()
    t1 = time.time()

    tdp_ms = (t1 - t0) / 100 * 1000

    print(f"Running custom `process` on average takes {tdp_ms:.3f}ms.")
    print("Reference computation time on RTX 4090: 68.185ms.")
    print("--------------------------------------------------------------")


if __name__ == "__main__":
    hmm_optim_speed_test()
