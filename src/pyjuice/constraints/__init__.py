"""
Constraints on a PC's variables, for exact (and controlled approximate) constrained inference.

A constraint is a formal language over token ids; see :mod:`pyjuice.constraints.language.base`.
The package is conventionally imported as ``jc``::

    import pyjuice as juice
    import pyjuice.constraints as jc

    pc = juice.compile(juice.structures.HMM(seq_length = 12, num_latents = 64, num_emits = V))

    # the tokens 3, 7 appear in a row, and token 5 never appears
    c = jc.DFA.contains([[3, 7]], vocab_size = V) & ~jc.DFA.contains([[5]], vocab_size = V)
    # or on the text the tokens spell, read from a tokenizer: the text is a date (`pip install pyjuice[constraints]`)
    #   c = jc.Regex(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", tokenizer)
    # or in temporal logic over sets of token ids: never `good` right after `neg`, and `end` eventually
    #   c = jc.LTLf("G(neg -> X !good) & F end", {"neg": [...], "good": [...], "end": [...]}, vocab_size = V)
    cc = jc.compile(c, pc)      # a ConstrainedCircuit: pc under c, valid across parameter updates

    # decoding x ([batch, 12] token ids): the circuit's queries in a session that reuses work between them (with a
    # language model, or masked diffusion: see ConstrainedCircuit.decoder)
    with cc.decoder() as decoder:
        for t in range(12):
            logp = decoder.conditional(x[:, :t], targets = [t])    # log p(x_t = v | x_<t, C), [batch, V]
"""

from .language import (Constraint, Matcher, And, Or, Not, Concat, OPTIONAL_CAPABILITIES, DFA, DFAMatcher, Vocabulary,
                       Regex, JSONSchema, LTLf)
from .compiler import compile
from .errors import ConstraintCompileError
from .constrained_circuit import ConstrainedCircuit
