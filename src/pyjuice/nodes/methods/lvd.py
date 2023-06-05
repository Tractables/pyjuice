from __future__ import annotations

import torch
import functools

from pyjuice.utils.context_manager import _DecoratorContextManager
from pyjuice.utils.kmeans import faiss_kmeans
from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes
from .lvd_backend.counting import lvd_by_counting


class LVDistiller(_DecoratorContextManager):
    def __init__(self, discretizer = "KMeans", pseudocount = 0.1, 
                 backend = "counting", verbose = False, **kwargs):

        self.discretizer = discretizer
        self.pseudocount = pseudocount
        self.backend = backend
        self.verbose = verbose
        self.kwargs = kwargs

        self.lv_datasets = []
        self.obs_datasets = []

        self.ns2lv_dataset_id = dict()
        self.ns2obs_dataset_id = dict()

        self.callback = functools.partial(lvd_callback_fn, lvdistiller = self)

    def __enter__(self) -> None:
        CircuitNodes.INIT_CALLBACKS.append(self.callback)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        CircuitNodes.INIT_CALLBACKS.remove(self.callback)


def lvd_callback_fn(ns: CircuitNodes, lvdistiller: LVDistiller, lv_dataset: Optional[torch.Tensor] = None, 
                    obs_dataset: Optional[torch.Tensor] = None, **kwargs):

    if obs_dataset is not None:
        obs_dataset_id = len(lvdistiller.obs_datasets)
        lvdistiller.obs_datasets.append(obs_dataset)
        lvdistiller.ns2obs_dataset_id[ns] = obs_dataset_id

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

    lv_dataset_id = len(lvdistiller.lv_datasets)
    lvdistiller.lv_datasets.append(lv_dataset)

    lvdistiller.ns2lv_dataset_id[ns] = lv_dataset_id

    # Perform LVD by the specified backend
    if lvdistiller.backend == "counting":
        lvd_by_counting(lvdistiller, ns)
        
    else:
        raise NotImplementedError(f"Unknown LVD backend {lvdistiller.backend}.")