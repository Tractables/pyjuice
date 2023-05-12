from __future__ import annotations

import torch
import functools

from pyjuice.utils.context_manager import _DecoratorContextManager
from pyjuice.utils.kmeans import faiss_kmeans
from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes
from .lvd_backend.counting import get_pairwise_count


def lvd_callback_fn(ns: CircuitNodes, lvdistiller: LVDistiller, lv_dataset: Optional[torch.Tensor] = None, **kwargs):

    if lv_dataset is None:
        # Do nothing if no LV dataset is provided
        return

    # Preprocess LV dataset
    if lv_dataset.dtype == torch.long and lv_dataset.dim() > 1:
        raise ValueError(f"Expecting `lv_dataset` with dtype `torch.long` to be a 1-D tensor, but got a tensor of shape {lv_dataset.size()}.")

    elif lv_dataset.dtype != torch.long:
        if lv_dataset.dim() != 2:
            raise ValueError(f"Expecting `lv_dataset` with dtype `{lv_dataset.dtype}` to be a 2-D tensor, but received a {lv_dataset.dim()}-D tensor.")

        # Discretize `lv_dataset`
        if lvdistiller.discretizer == "KMeans":
            lv_dataset = faiss_kmeans(lv_dataset, n_clusters = ns.num_nodes)
        else:
            raise NotImplementedError(f"Unknown discretizer {lvdistiller.discretizer}.")

    assert lv_dataset.max() < ns.num_nodes, f"Got more latent clusters ({lv_dataset.max().item()}) than nodes ({ns.num_nodes})."

    dataset_id = len(lvdistiller.datasets)
    lvdistiller.datasets.append(lv_dataset)

    lvdistiller.ns2dataset_id[ns] = dataset_id

    # Perform LVD by the specified backend
    if lvdistiller.backend == "counting":
        if isinstance(ns, ProdNodes):
            for cs in ns.chs:
                if cs not in lvdistiller.ns2dataset_id:
                    lvdistiller.ns2dataset_id[cs] = dataset_id

        # Get candidate LVD nodes
        nodes_for_lvd = []
        if isinstance(ns, SumNodes):
            if any([cs in lvdistiller.ns2dataset_id for cs in ns.chs]):
                nodes_for_lvd.append(ns)
        elif isinstance(ns, ProdNodes):
            for cs in ns.chs:
                if any([ccs in lvdistiller.ns2dataset_id for ccs in cs.chs]):
                    nodes_for_lvd.append(cs)

        # Run LVD by counting
        for ns in nodes_for_lvd:
            ns_dataset = lvdistiller.datasets[lvdistiller.ns2dataset_id[ns]]
            num_ch_nodes = sum([cs.num_nodes for cs in ns.chs])
            edge_params = torch.empty([ns.num_nodes, num_ch_nodes], dtype = torch.float32, device = ns_dataset.device)
            sid = 0
            for cs in ns.chs:
                eid = sid + cs.num_nodes
                if cs in lvdistiller.ns2dataset_id:
                    cs_dataset = lvdistiller.datasets[lvdistiller.ns2dataset_id[cs]]
                    pairwise_count = get_pairwise_count(ns_dataset, cs_dataset, ns.num_nodes, cs.num_nodes)
                else:
                    # Randomly initialize parameters
                    pairwise_count = torch.exp(-torch.rand([ns.num_nodes, cs.num_nodes]) * 2.0)

                edge_params[:,sid:eid] = pairwise_count / pairwise_count.sum(dim = 1, keepdim = True)
                sid = eid

            ns.set_params(edge_params)
        
    else:
        raise NotImplementedError(f"Unknown backend {lvdistiller.backend}.")


class LVDistiller(_DecoratorContextManager):
    def __init__(self, discretizer = "KMeans", pseudocount = 0.1, 
                 backend = "counting", verbose = False):

        self.discretizer = discretizer
        self.pseudocount = pseudocount
        self.backend = backend
        self.verbose = verbose

        self.datasets = []

        self.ns2dataset_id = dict()

        self.callback = functools.partial(lvd_callback_fn, lvdistiller = self)

    def __enter__(self) -> None:
        CircuitNodes.INIT_CALLBACKS.append(self.callback)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        CircuitNodes.INIT_CALLBACKS.remove(self.callback)