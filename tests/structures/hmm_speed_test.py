import pyjuice as juice
import pyjuice.nodes.distributions as dists
import torch
import time


def hmm_speed_test():

    device = torch.device("cuda:0")

    seq_length = 32
    vocab_size = 33279

    fw_4090_ref_times = {64: 0.230, 512: 0.956, 2048: 9.136}
    bk_4090_ref_times = {64: 0.929, 512: 3.503, 2048: 21.262}

    for num_latents in [64, 512, 2048]:

        group_size = min(juice.utils.util.max_cdf_power_of_2(num_latents), 1024)
        num_node_groups = num_latents // group_size
        
        with juice.set_group_size(group_size = group_size):
            ns_input = juice.inputs(seq_length - 1, num_node_groups = num_node_groups,
                                    dist = dists.Categorical(num_cats = vocab_size))
            
            ns_sum = None
            curr_zs = ns_input
            for var in range(seq_length - 2, -1, -1):
                curr_xs = ns_input.duplicate(var, tie_params = True)
                
                if ns_sum is None:
                    ns = juice.summate(
                        curr_zs, num_node_groups = num_node_groups)
                    ns_sum = ns
                else:
                    ns = ns_sum.duplicate(curr_zs, tie_params=True)

                curr_zs = juice.multiply(curr_xs, ns)
                
            ns = juice.summate(curr_zs, num_node_groups = 1, group_size = 1)
        
        ns.init_parameters()

        pc = juice.TensorCircuit(ns, max_tied_ns_per_parflow_group = 2)
        pc.print_statistics()

        pc.to(device)

        data = torch.randint(0, 33279, (512, seq_length)).to(device)

        print("==============================================================")
        print(f"- num_latents={num_latents}, seq_length={seq_length}")

        lls = pc(data, record_cudagraph = num_latents <= 1024)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            lls = pc(data)
        torch.cuda.synchronize()
        t1 = time.time()
        aveg_fw_ms = (t1 - t0) * 1000 / 100

        print(f"Forward pass on average takes {aveg_fw_ms:.3f}ms.")
        print(f"Reference computation time on RTX 4090: {fw_4090_ref_times[num_latents]:.3f}ms.")
        print("--------------------------------------------------------------")

        data_trans = data.permute(1, 0).contiguous()
        pc.backward(data_trans, allow_modify_flows = True, record_cudagraph = num_latents <= 1024)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            pc.backward(data_trans, allow_modify_flows = True)
        torch.cuda.synchronize()
        t1 = time.time()
        aveg_bk_ms = (t1 - t0) * 1000 / 100

        print(f"Backward pass on average takes {aveg_bk_ms:.3f}ms.")
        print(f"Reference computation time on RTX 4090: {bk_4090_ref_times[num_latents]:.3f}ms.")
        print("--------------------------------------------------------------")

        print("==============================================================")


if __name__ == "__main__":
    hmm_speed_test()

