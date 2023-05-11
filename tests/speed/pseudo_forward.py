import numpy as np
import torch
import time

import triton
import triton.language as tl

torch.set_float32_matmul_precision('high')


@triton.jit
def _simple_kernel(node_mars_ptr, nids_ptr, nid_size, batch_size, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < nid_size * batch_size

    nid_offsets = offsets // batch_size
    batch_offsets = offsets % batch_size

    n_offsets = tl.load(nids_ptr + nid_offsets, mask = mask, other = 0)
    n_offsets = n_offsets * batch_size + batch_offsets

    tl.store(node_mars_ptr + n_offsets, 0.1232, mask = mask)

def run_simple(node_mars, nids):
    nids = nids.reshape(-1)
    batch_size = node_mars.size(1)

    grid = lambda meta: (triton.cdiv(nids.size(0) * batch_size, meta['BLOCK_SIZE']),)
    _simple_kernel[grid](node_mars_ptr = node_mars, nids_ptr = nids, nid_size = nids.size(0), batch_size = batch_size, BLOCK_SIZE = 1024)


# @torch.compile(mode = "default")
def run_layer(node_mars, element_mars, params, nids, cids, pids, bbb):
    # ch_mars = element_mars[cids]
    # ch_mars = torch.index_select(element_mars, 0, cids.reshape(-1)).reshape(cids.size(0), cids.size(1), -1)
    # maxval = ch_mars.max(dim = 1, keepdim = True).values
    
    # node_mars[nids] = (((ch_mars - maxval).exp() * params[pids]).sum(
    #     dim = 1).clamp(min = 1e-10)).log() + maxval.squeeze(1)
    
    # node_mars[nids] = ((ch_mars - maxval).sum(dim = 1))
    
    # node_mars[nids] = 1.22378
    # bbb[:,:] = 1.22378
    # aaa = torch.index_select(node_mars, 0, nids.reshape(-1))
    # aaa[:,:] = 1.22378

    run_simple(node_mars, nids)

    return None


def main():

    device = torch.device("cuda:0")

    data = np.load("fsss.npz")

    node_mars = torch.from_numpy(data["node_mars"]).to(device)
    element_mars = torch.from_numpy(data["element_mars"]).to(device)
    params = torch.from_numpy(data["params"]).to(device)

    nids = torch.from_numpy(data["nids"]).to(device)
    cids = torch.from_numpy(data["cids"]).to(device)
    pids = torch.from_numpy(data["pids"]).to(device)

    bbb = torch.empty([nids.size(0), node_mars.size(1)]).to(device)

    with torch.no_grad():
        run_layer(node_mars, element_mars, params.unsqueeze(1), nids, cids, pids, bbb)

        s = time.time()
        for _ in range(1000):
            run_layer(node_mars, element_mars, params.unsqueeze(1), nids, cids, pids, bbb)
        torch.cuda.synchronize()
        e = time.time()
        print((e - s) / 1000)


if __name__ == "__main__":
    main()