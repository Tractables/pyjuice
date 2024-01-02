from __future__ import annotations

import torch

from typing import Optional, Any


class Distribution():
    def __init__(self):
        pass

    def get_signature(self):
        raise NotImplementedError()

    def get_metadata(self):
        return [] # no metadata

    def normalize_params(self, params: torch.Tensor):
        return params

    def num_parameters(self):
        """
        The number of parameters per node.
        """
        raise NotImplementedError()

    def num_param_flows(self):
        """
        The number of parameter flows per node.
        """
        raise NotImplementedError()

    def init_parameters(self, num_nodes: int, perturbation: float = 2.0, params: Optional[Any] = None, **kwargs):
        """
        Initialize parameters for `num_nodes` nodes.
        Returned parameters should be flattened into a vector.
        """
        raise NotImplementedError()

    @property
    def need_external_params(self):
        """
        A flag indicating whether users need to pass in `params` to the 
        constructor of InputNodes.
        """
        return False

    @staticmethod
    def fw_mar_fn(*args, **kwargs):
        """
        Forward evaluation for log-probabilities.
        Args:
        `local_offsets`: [BLOCK_SIZE] the local indices of the to-be-processed input nodes
        `data`: [BLOCK_SIZE, num_vars_per_node] data of the corresponding nodes
        `params_ptr`: pointer to the parameter vector
        `s_pids`: [BLOCK_SIZE] start parameter index (offset) for all input nodes
        `metadata_ptr`: pointer to metadata
        `s_mids_ptr`: pointer to the start metadata index (offset)
        `mask`: [BLOCK_SIZE] indicate whether each node should be processed
        `num_vars_per_node`: numbers of variables per input node/distribution
        `BLOCK_SIZE`: CUDA block size
        """
        raise NotImplementedError()

    @staticmethod
    def bk_flow_fn(*args, **kwargs):
        """
        Accumulate statistics and compute input parameter flows.
        Args:
        `local_offsets`: [BLOCK_SIZE] the local indices of the to-be-processed input nodes
        `ns_offsets`: [BLOCK_SIZE] the global offsets used to load from `node_mars_ptr`
        `data`: [BLOCK_SIZE, num_vars_per_node] data of the corresponding nodes
        `flows`: [BLOCK_SIZE] node flows
        `node_mars_ptr`: pointer to the forward values
        `params_ptr`: pointer to the parameter vector
        `param_flows_ptr`: pointer to the parameter flow vector
        `s_pids`: [BLOCK_SIZE] start parameter index (offset) for all input nodes
        `s_pfids`: [BLOCK_SIZE] start parameter flow index (offset) for all input nodes
        `metadata_ptr`: pointer to metadata
        `s_mids_ptr`: pointer to the start metadata index (offset)
        `mask`: [BLOCK_SIZE] indicate whether each node should be processed
        `num_vars_per_node`: numbers of variables per input node/distribution
        `BLOCK_SIZE`: CUDA block size
        """
        raise NotImplementedError()

    @staticmethod
    def sample_fn(*args, **kwargs):
        """
        Sample from the distribution.
        Args:
        `samples_ptr`: pointer to store the resultant samples 
        `local_offsets`: [BLOCK_SIZE] the local indices of the to-be-processed input nodes
        `batch_offsets`: [BLOCK_SIZE] batch id corresponding to every node
        `vids`: [BLOCK_SIZE] variable ids (only univariate distributions are supported)
        `s_pids`: [BLOCK_SIZE] start parameter index (offset) for all input nodes 
        `params_ptr`: pointer to the parameter vector
        `metadata_ptr`: pointer to metadata
        `s_mids_ptr`: pointer to the start metadata index (offset)
        `mask`: [BLOCK_SIZE] indicate whether each node should be processed
        `batch_size`: batch size
        `BLOCK_SIZE`: CUDA block size
        `seed`: random seed
        """
        raise NotImplementedError()

    @staticmethod
    def em_fn(*args, **kwargs):
        """
        Parameter update with EM
        Args:
        `local_offsets`: [BLOCK_SIZE] the local indices of the to-be-processed input nodes
        `params_ptr`: pointer to the parameter vector
        `param_flows_ptr`: pointer to the parameter flow vector
        `s_pids`: [BLOCK_SIZE] start parameter index (offset) for all input nodes
        `s_pfids`: [BLOCK_SIZE] start parameter flow index (offset) for all input nodes
        `metadata_ptr`: pointer to metadata
        `s_mids_ptr`: pointer to the start metadata index (offset)
        `mask`: [BLOCK_SIZE] indicate whether each node should be processed
        `step_size`: EM step size (0, 1]
        `pseudocount`: pseudocount 
        `BLOCK_SIZE`: CUDA block size
        """
        raise NotImplementedError()
