import torch
import time

torch.set_float32_matmul_precision('high')


@torch.compile(mode = "reduce-overhead", fullgraph = True)
def run_layer(node_mars, element_mars, params, nids, cids, pids):
    ch_mars = element_mars[cids]
    maxval = ch_mars.max(dim = 1, keepdim = True).values
    node_mars[nids] = (((ch_mars - maxval).exp() * params[pids]).sum(
        dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)

    return None


def main():
    N = 10000
    M = 20000
    E = 200
    B = 512

    device = torch.device("cuda:0")

    node_mars = (torch.rand([N, B]) * -2.0).to(device)
    element_mars = (torch.rand([M, B]) * -2.0).to(device)
    params = torch.rand([N, 1]).to(device)

    nids = torch.arange(N).to(device)
    cids = torch.randint(0, M, [N, E]).to(device)
    pids = torch.randint(0, M, [N, E]).to(device)

    run_layer(node_mars, element_mars, params, nids, cids, pids)

    s = time.time()
    for _ in range(1000):
        run_layer(node_mars, element_mars, params, nids, cids, pids)
    torch.cuda.synchronize()
    e = time.time()
    print((e - s) / 1000)


if __name__ == "__main__":
    main()