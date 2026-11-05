"""
What a layout's transitions are read as: the enumerated tokens and the transition tables, whichever side built them (:mod:`.tables`, :mod:`.host`).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch


@dataclass(frozen = True)
class EnumeratedTokens:
    """
    The enumerated classes' tokens that are live at every boundary (lead somewhere from some active column), and
    every class's CODE at every boundary: its stored column, or ``-1 - e`` for the ``e``-th of the boundary's live
    enumerated tokens (a class dead at the boundary, enumerated: -1 - the count, matching no entry).

    :ivar ptr: [n + 1] int32, the boundary's tokens are ``tok[ptr[t] .. ptr[t + 1]]``; ``ptr_host`` the same on the host
    :ivar tok: [num + 1] int32, the tokens, then a 0 (kernels read a boundary's first token unmasked for a stored
        entry: past the last boundary's, this one)
    :ivar index: [num + 1] int32, every token's place among the automaton's enumerated tokens
        (``TokenClasses.enumerated_tokens``, in class order) -- its column of the enumerated tokens' probabilities
        (:meth:`~..state.State.enumerated_probs`) --, then a 0
    :ivar code: [n, C] int32
    :ivar num: the tokens over all boundaries (0: none is live at this length, or nothing is enumerated)
    :ivar on: whether any class is enumerated, which is what the codes depend on: without, every code is the class itself;
        with, a stored class's is its stored column, past the classes enumerated before it, even if ``num`` is 0
    """

    ptr: torch.Tensor
    tok: torch.Tensor
    index: torch.Tensor
    code: torch.Tensor
    num: int
    ptr_host: np.ndarray
    on: bool


@dataclass(frozen = True)
class TransitionTables:
    """
    Every boundary's transitions as entries (column, successor, mass), the successor of a sequence end being its one
    column. Ungrouped, one entry per (column, class) that leads somewhere, its mass the class's mass -- read by the
    class's code (:class:`EnumeratedTokens`). Grouped, one per (column, successor) pair, its mass the pair's
    transition mass: the mass of every class that leads there (:func:`transition_masses`) -- a missing token then
    costs a gather per successor, not per class.

    :ivar grouped: which of the two
    :ivar entry_ptr: [n, W + 1] int32, the entries of column ``q`` at ``t``: ``entry_ptr[t, q] .. entry_ptr[t, q + 1]``
        (global numbers; ``entry_ptr[t, 0]``: the boundary's first)
    :ivar entry_succ, entry_x: [entries] int32, every entry's successor column and where its mass is: its class's
        code, or its pair's number within the boundary
    :ivar pred_ptr, pred_col, pred_x: the entries leading to column ``j`` at ``t + 1``, as a CSR (``pred_ptr``
        [n, W + 1]) into their columns and their mass indices
    :ivar num_pairs: grouped only, pairs per boundary (Python ints)
    :ivar max_pairs: grouped only, the most pairs of any column
    :ivar enum_ptr, enum_e, enum_pair: grouped only, the enumerated classes' live entries of every column, as a CSR
        (``enum_ptr`` [n, W + 1] int32, global numbers): the token's number ``e`` at its boundary and the pair it leads
        to; ``max_enum`` the most of any column
    :ivar cls_ptr, cls_col, cls_succ: the same (column, class) entries BY CLASS: the columns where class ``c`` leads
        somewhere at ``t``, ascending, ``cls_col[cls_ptr[t, c] .. cls_ptr[t, c + 1]]`` (``cls_ptr`` [n, C + 1] int32,
        global numbers), and their successors (a sequence end's: its one column) -- the class weights' walk over a
        class's columns (:func:`~.score._class_weight_kernel`); ``max_cls`` the most of any (boundary, class);
        ``cls_order`` [n, C] int32, every boundary's classes by their columns, most first (the walk's lanes take
        classes in this order, so that a program's classes have lists of a length)
    :ivar entry_class, pred_class: ungrouped only, [entries] int32: every entry's class, in the entry and the
        predecessor order -- by which a session's per-sample inputs rule entries out (the ``input @ block`` entry loops'
        :class:`~.live.SuccessorLists`, the CUDA top-down walk's terms)
    :ivar boundary_base, num_entries: boundaries alike share their entries (see :func:`transition_tables`), so a
        structure with a value PER BOUNDARY AND ENTRY -- a sample's successor lists -- counts the boundaries' entries
        one after the other: ``boundary_base`` int64 [n + 1] on the device, boundary ``t``'s entries are
        ``boundary_base[t] .. boundary_base[t + 1]`` there (its entry ``e`` at ``boundary_base[t] + e - entry_ptr[t, 0]``),
        ``num_entries`` their total, ``boundary_base[n]``
    """

    grouped: bool
    entry_ptr: torch.Tensor
    entry_succ: torch.Tensor
    entry_x: torch.Tensor
    pred_ptr: torch.Tensor
    pred_col: torch.Tensor
    pred_x: torch.Tensor
    num_pairs: list
    max_pairs: int
    enum_ptr: Optional[torch.Tensor] = None
    enum_e: Optional[torch.Tensor] = None
    enum_pair: Optional[torch.Tensor] = None
    max_enum: int = 0
    cls_ptr: Optional[torch.Tensor] = None
    cls_col: Optional[torch.Tensor] = None
    cls_succ: Optional[torch.Tensor] = None
    max_cls: int = 0
    cls_order: Optional[torch.Tensor] = None
    entry_class: Optional[torch.Tensor] = None
    pred_class: Optional[torch.Tensor] = None
    boundary_base: Optional[torch.Tensor] = None
    num_entries: int = 0
