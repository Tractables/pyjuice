import pyjuice as juice
import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
from pyjuice.nodes.methods import foldup_aggregate
import pyjuice.nodes.distributions as dists

import pytest


def augment_input_ns_with_external_distributions(root_ns):

    num_vars = root_ns.num_vars

    old2new = dict()

    def aggr_fn(ns, ch_outputs):
        if ns.is_input():
            var = ns.scope.to_list()[0]
            ns_exi = juice.deepcopy(ns)
            ns_ext = juice.inputs(var + num_vars, num_node_blocks = ns.num_node_blocks, block_size = ns.block_size, dist = dists.External())

            new_ns = juice.multiply(ns_exi, ns_ext)
            old2new[ns] = new_ns
        
        elif ns.is_prod():
            new_chs = []
            for nc in ch_outputs:
                if nc.is_prod():
                    new_chs.extend(nc.chs)
                else:
                    new_chs.append(nc)

            new_ns = juice.multiply(*new_chs)
            old2new[ns] = new_ns

        else:
            assert ns.is_sum()

            new_ns = juice.summate(*ch_outputs, edge_ids = ns.edge_ids, num_node_blocks = ns.num_node_blocks, block_size = ns.block_size, params = ns.get_params())
            old2new[ns] = new_ns

        return new_ns

    new_root_ns = foldup_aggregate(aggr_fn, root_ns)

    for ns in root_ns:
        if ns.is_tied():
            source_ns = ns.get_source_ns()
            old2new[ns].set_source_ns(old2new[source_ns])

    return new_root_ns


@pytest.mark.slow
def test_hmm_extern():

    num_latents = 128

    device = torch.device("cuda:0")

    train_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = True, download = True)
    test_dataset = torchvision.datasets.MNIST(root = "./examples/data", train = False, download = True)

    train_data = train_dataset.data.reshape(60000, 28*28)
    test_data = test_dataset.data.reshape(10000, 28*28)

    num_features = train_data.size(1)

    train_loader = DataLoader(
        dataset = TensorDataset(train_data),
        batch_size = 512,
        shuffle = True,
        drop_last = True
    )
    test_loader = DataLoader(
        dataset = TensorDataset(test_data),
        batch_size = 512,
        shuffle = False,
        drop_last = True
    )

    root_ns = juice.structures.HCLT(
        train_data.float().to(device), 
        num_bins = 32, 
        sigma = 0.5 / 32, 
        num_latents = num_latents, 
        chunk_size = 32
    )
    root_ns.init_parameters(perturbation = 2.0)
    root_ns = augment_input_ns_with_external_distributions(root_ns) # Step 1
    pc = juice.TensorCircuit(root_ns)

    pc.to(device)

    # Simple full-batch EM
    for epoch in range(10):

        pc.zero_param_flows()

        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].long().to(device)

            # Assume this is the NN output
            nn_logprobs = torch.rand([x.size(0), x.size(1), 256], device = device) * 0 # Step 2

            # external_soft_evi[b,v] should be the logprob of the v-th variable in the b-th sample
            external_soft_evi = nn_logprobs.gather(2, x[:,:,None])

            # Keep this to keep it compatible with num_latents > 1
            external_soft_evi = external_soft_evi.repeat(1, 1, num_latents)

            unnorm_lls = pc(x.repeat(1, 2), external_soft_evi = external_soft_evi) # Step 3: include `.repeat(1, 2)` and the `external_soft_evi` argument
            pc.backward(x.repeat(1, 2), allow_modify_flows = False, logspace_flows = True)

            train_ll += unnorm_lls.mean().detach().cpu().numpy().item()

        pc.mini_batch_em(step_size = 1.0, pseudocount = 0.0001)

        print(f"Epoch {epoch}: Train LL = {train_ll / len(train_loader)}")


if __name__ == "__main__":
    test_hmm_extern()
