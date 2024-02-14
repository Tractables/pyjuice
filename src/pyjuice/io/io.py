from __future__ import annotations

import torch
import pickle
from functools import partial
from typing import Sequence

from pyjuice.model import TensorCircuit
from pyjuice.nodes import CircuitNodes, InputNodes, ProdNodes, SumNodes
from .serialization import serialize_nodes, deserialize_nodes


def save(fname: str, model: Union[CircuitNodes,TensorCircuit]):
    """
    Save a PC to file. The file should end with ".jpc".

    :param fname: file name
    :type fname: str

    :param model: an uncompiled (`CircuitNodes`) or compiled (`TensorCircuit`) PC
    :type model: Union[CircuitNodes,TensorCircuit]
    """
    if isinstance(model, TensorCircuit):
        model.update_parameters()
        root_ns = model.root_ns
    else:
        root_ns = model

    if fname.endswith(".jpc"):
        sel_nodes = serialize_nodes(root_ns)
        with open(fname, "wb") as f:
            pickle.dump(sel_nodes, f)
    else:
        raise ValueError(f"Unknown file type `.{fname.split('.')[-1]}`.")


def load(fname: str) -> CircuitNodes:
    """
    Load a PC from file. The file should end with ".jpc".

    :param fname: file name
    :type fname: str

    :returns: a PC node
    """
    if fname.endswith(".jpc"):
        with open(fname, "rb") as f:
            sel_nodes = pickle.load(f)
        root_ns = deserialize_nodes(sel_nodes)
    else:
        raise ValueError(f"Unknown file type `.{fname.split('.')[-1]}`.")

    return root_ns