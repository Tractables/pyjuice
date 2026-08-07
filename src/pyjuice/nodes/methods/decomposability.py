from __future__ import annotations

from pyjuice.nodes import CircuitNodes


def is_structured_decomposable(root_ns) -> bool:
    """
    Whether every product node of the PC decomposes its scope the same way, i.e. whether the circuit
    respects a single vtree.

    Formally: for any two product nodes with the same scope, the partition of that scope into their
    children's scopes is identical. That is exactly the condition under which a vtree exists, since
    a scope's decomposition is then a function of the scope alone.

    Two structures make the difference concrete. An `HCLT` follows one Chow-Liu tree, so every
    product over a given scope splits it identically -- structured decomposable. A `PD` circuit
    deliberately sums over SEVERAL split points of the same region, so one scope has more than one
    decomposition -- not structured decomposable, and that is the point of the structure rather than
    a defect.

    :note: what this buys the sampler. During a top-down pass the frontier is expanded by taking a
           product node's children, so the SHAPE of the frontier after each layer is determined by
           the scope decompositions encountered. Under structured decomposability those are a
           function of the scope alone -- never of which node the sampler happened to draw -- so the
           whole index plan (which frontier slot each layer owns, and where its children are written)
           repeats identically on every call and can be computed once. Without it, a sum node may
           choose between products of different arity and the plan genuinely changes call to call.
           MEASURED: HMM / HCLT / RAT-SPN / a ragged hand-built circuit are invariant, PD is not,
           and the flag separates the two exactly.

    :param root_ns: the root of the PC, or a compiled `TensorCircuit`
    :type root_ns: Union[CircuitNodes,TensorCircuit]

    :returns: whether the PC is structured decomposable
    :rtype: bool
    """
    if not isinstance(root_ns, CircuitNodes):
        root_ns = root_ns.root_ns

    def _partition(ns):
        # Sorted, so that the same decomposition written with its children in another order compares
        # equal -- `chs` order is an artifact of how the circuit was built, not of the decomposition
        return tuple(sorted(tuple(sorted(ch.scope)) for ch in ns.chs))

    # Keyed by a canonical form of the scope rather than by `BitSet` itself: `BitSet.__eq__` ignores
    # trailing zero bytes but `__hash__` is taken over the raw bytes, so two equal scopes of
    # different byte length would land in different buckets and every check would silently pass.
    partitions = dict()

    for ns in root_ns:
        if ns.is_prod():
            # A ONE-CHILD product splits nothing, so it is not a decomposition and must not be
            # compared against real ones. Skipping it is not cosmetic: pyjuice caps a PC with
            # `summate(multiply(ns), ...)`, so every circuit has a unary product over the full scope,
            # and counting it made even an HMM -- which follows an obvious linear vtree -- come out
            # non-decomposable.
            if len(ns.chs) < 2:
                continue

            scope = frozenset(ns.scope)
            partition = _partition(ns)
            if partitions.setdefault(scope, partition) != partition:
                return False

        elif ns.is_sum():
            # The one way a unary product can still create ambiguity: a sum node that mixes one with
            # a real split, i.e. a mixture of a "flat" component and a decomposed one over the same
            # scope. The loop above deliberately ignores the unary member, so this is what stops that
            # slipping through -- and it is a local check, which the global one is not.
            child_partitions = {_partition(cs) for cs in ns.chs if cs.is_prod()}
            if len(child_partitions) > 1:
                return False

    return True
