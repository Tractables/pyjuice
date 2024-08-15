from __future__ import annotations

import torch

from typing import Optional, Any


class Distribution():
    def __init__(self):
        pass

    def get_signature(self):
        """
        Get the signature of the current distribution.
        """
        raise NotImplementedError()

    def get_metadata(self):
        """
        Get the metadata of the current distribution.
        """
        return [] # no metadata

    def normalize_parameters(self, params: torch.Tensor, **kwargs):
        """
        Normalize node parameters.
        """
        return params

    def set_meta_parameters(self, **kwargs):
        """
        Assign meta-parameters to `self._params`.
        
        :note: the actual parameters are not initialized after this function call.
        """
        raise NotImplementedError()

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
        Randomly initialize node parameters.

        :param num_nodes: number of nodes
        :type num_nodes: int

        :param perturbation: "amount of perturbation" added to the parameters (should be greater than 0)
        :type perturbation: float
        """
        raise NotImplementedError()

    def init_meta_parameters(self, num_nodes: int, params: Any, **kwargs):
        """
        Initialize meta-parameters for `num_nodes` nodes.
        Return shape should be the same with `init_parameters`.
        """
        raise NotImplementedError()

    @property
    def need_meta_parameters(self):
        """
        A flag indicating whether users need to pass in meta-parameters to the 
        constructor of InputNodes.
        """
        return False

    def get_data_dtype(self):
        """
        Get the data dtype for the distribution.
        """
        return torch.float32

    @staticmethod
    def fw_mar_fn(*args, **kwargs):
        """
        Forward evaluation for log-probabilities.

        :param local_offsets: [BLOCK_SIZE] the local indices of the to-be-processed input nodes
        :param data: [BLOCK_SIZE, num_vars_per_node] data of the corresponding nodes
        :param params_ptr: pointer to the parameter vector
        :param s_pids: [BLOCK_SIZE] start parameter index (offset) for all input nodes
        :param metadata_ptr: pointer to metadata
        :param s_mids_ptr: pointer to the start metadata index (offset)
        :param mask: [BLOCK_SIZE] indicate whether each node should be processed
        :param num_vars_per_node: numbers of variables per input node/distribution
        :param BLOCK_SIZE: CUDA block size
        """
        raise NotImplementedError()

    @staticmethod
    def bk_flow_fn(*args, **kwargs):
        """
        Accumulate statistics and compute input parameter flows.

        :param local_offsets: [BLOCK_SIZE] the local indices of the to-be-processed input nodes
        :param ns_offsets: [BLOCK_SIZE] the global offsets used to load from `node_mars_ptr`
        :param data: [BLOCK_SIZE, num_vars_per_node] data of the corresponding nodes
        :param flows: [BLOCK_SIZE] node flows
        :param node_mars_ptr: pointer to the forward values
        :param params_ptr: pointer to the parameter vector
        :param param_flows_ptr: pointer to the parameter flow vector
        :param s_pids: [BLOCK_SIZE] start parameter index (offset) for all input nodes
        :param s_pfids: [BLOCK_SIZE] start parameter flow index (offset) for all input nodes
        :param metadata_ptr: pointer to metadata
        :param s_mids_ptr: pointer to the start metadata index (offset)
        :param mask: [BLOCK_SIZE] indicate whether each node should be processed
        :param num_vars_per_node: numbers of variables per input node/distribution
        :param BLOCK_SIZE: CUDA block size
        """
        raise NotImplementedError()

    @staticmethod
    def sample_fn(*args, **kwargs):
        """
        Sample from the distribution.

        :param samples_ptr: pointer to store the resultant samples 
        :param local_offsets: [BLOCK_SIZE] the local indices of the to-be-processed input nodes
        :param batch_offsets: [BLOCK_SIZE] batch id corresponding to every node
        :param vids: [BLOCK_SIZE] variable ids (only univariate distributions are supported)
        :param s_pids: [BLOCK_SIZE] start parameter index (offset) for all input nodes 
        :param params_ptr: pointer to the parameter vector
        :param metadata_ptr: pointer to metadata
        :param s_mids_ptr: pointer to the start metadata index (offset)
        :param mask: [BLOCK_SIZE] indicate whether each node should be processed
        :param batch_size: batch size
        :param BLOCK_SIZE: CUDA block size
        :param seed: random seed
        """
        raise NotImplementedError()

    @staticmethod
    def em_fn(*args, **kwargs):
        """
        Parameter update with EM

        :param local_offsets: [BLOCK_SIZE] the local indices of the to-be-processed input nodes
        :param params_ptr: pointer to the parameter vector
        :param param_flows_ptr: pointer to the parameter flow vector
        :param s_pids: [BLOCK_SIZE] start parameter index (offset) for all input nodes
        :param s_pfids: [BLOCK_SIZE] start parameter flow index (offset) for all input nodes
        :param metadata_ptr: pointer to metadata
        :param s_mids_ptr: pointer to the start metadata index (offset)
        :param mask: [BLOCK_SIZE] indicate whether each node should be processed
        :param step_size: EM step size (0, 1]
        :param pseudocount: pseudocount 
        :param BLOCK_SIZE: CUDA block size
        """
        raise NotImplementedError()

    @staticmethod
    def partition_fn(local_offsets, params_ptr, s_pids, metadata_ptr, s_mids_ptr, mask, BLOCK_SIZE, TILE_SIZE_K):
        """
        Evaluate partition function

        :param local_offsets: [BLOCK_SIZE] the local indices of the to-be-processed input nodes
        :param params_ptr: pointer to the parameter vector
        :param s_pids: [BLOCK_SIZE] start parameter index (offset) for all input nodes
        :param metadata_ptr: pointer to metadata
        :param s_mids_ptr: pointer to the start metadata index (offset)
        :param mask: [BLOCK_SIZE] indicate whether each node should be processed
        :param BLOCK_SIZE: CUDA block size
        :param TILE_SIZE_K: tile size for processing each individual input node
        """
        raise NotImplementedError()

    def _get_constructor(self):
        raise NotImplementedError()

    def _need_2nd_kernel_dim(self):
        return False
