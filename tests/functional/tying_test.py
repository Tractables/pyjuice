import pyjuice as juice
import torch
from pyjuice.functional import tie_param_flows


def tying_test():

    device = torch.device("cuda:0")

    N = 20

    param_flows = torch.rand([1000]).cuda()
    tied_param_ids = torch.arange(20).cuda()
    tied_param_group_ids = torch.arange(10).unsqueeze(1).repeat(1, 2).reshape(-1).cuda()

    tied_flows = param_flows[:20].reshape(10, 2).sum(dim = 1)

    tie_param_flows(
        param_flows = param_flows, 
        num_tied_params = N, 
        tied_param_ids = tied_param_ids, 
        tied_param_group_ids = tied_param_group_ids
    )

    assert torch.max(torch.abs(tied_flows.unsqueeze(1).repeat(1, 2).reshape(-1) - param_flows[:20])) < 1e-6


if __name__ == "__main__":
    tying_test()