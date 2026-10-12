"""
The examples in :meth:`ConstrainedCircuit.decoder`'s docstring, run as written: Hugging Face ``generate`` (a tiny random
GPT-2; beam search reorders the rows) with the Ctrl-G weighting, with and without candidates, and a masked-diffusion
loop (a random denoiser). Every sequence they produce satisfies the constraint, and is the one the same code produces
with the circuit's own queries in place of the decoders'.
"""
import contextlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

import pyjuice as juice
import pyjuice.constraints as jc

V, N = 3, 6


def examples():
    """The docstring's code blocks (after a line ending in ``::``), dedented, in order."""
    lines = inspect.cleandoc(jc.ConstrainedCircuit.decoder.__doc__).splitlines()
    depth = lambda line: len(line) - len(line.lstrip())
    blocks, i = [], 0
    while i < len(lines):
        if not lines[i].rstrip().endswith("::"):
            i += 1
            continue
        j = i + 1
        while not lines[j].strip():
            j += 1
        k = j
        while k < len(lines) and (not lines[k].strip() or depth(lines[k]) >= depth(lines[j])):
            k += 1
        blocks.append(textwrap.dedent("\n".join(lines[j:k])).strip("\n"))
        i = k
    assert len(blocks) == 4, "the loop, the processor, its candidates, masked diffusion"
    return blocks


@pytest.fixture(name = "cc")
def cc_fixture(build_pc):
    return jc.compile(jc.DFA.contains([[1, 2]], V), build_pc("hmm", N, V))


def test_the_loop(cc):
    data = torch.randint(0, V, (4, N), generator = torch.Generator().manual_seed(0))
    scope = dict(cc = cc, data = data, n = N)
    exec(examples()[0], scope)
    with juice.fast_inference():
        assert torch.equal(scope["logp"], cc.conditional(data[:, :N - 1], targets = [N - 1]))


@pytest.mark.parametrize("candidates", [False, True])
def test_generation_with_a_language_model(cc, candidates):
    transformers = pytest.importorskip("transformers")
    _, processor, top, _ = examples()
    if candidates:                                                  # the four lines from `c = ...` replaced
        body = textwrap.indent(top, " " * 8)
        start = processor.index("        c = self.decoder")
        end = processor.index("        return scores")
        processor = processor[:start] + body + "\n" + processor[end:]
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size = 70, n_positions = 32, n_embd = 16, n_layer = 1, n_head = 2,
                                     bos_token_id = 69, eos_token_id = 69, pad_token_id = 68)
    model = transformers.GPT2LMHeadModel(config).cuda().eval()       # the PC's 3 tokens and 67 more (EOS among them)
    prompt = torch.randint(0, 70, (2, 3), generator = torch.Generator().manual_seed(1)).cuda()
    scope = dict(torch = torch, jc = jc, cc = cc, pc = cc.pc, V = V, model = model, prompt = prompt)
    with torch.no_grad():
        exec(processor, scope)
        free = jc.compile(jc.DFA.anything(V), cc.pc)               # the circuits' own queries, no decoder
        plain = scope["ConstrainedProcessor"](cc, free, prompt.size(1))
        want = model.generate(prompt, max_new_tokens = cc.n, num_beams = 4, logits_processor = [plain])
    out = scope["out"]
    assert out.shape == (2, 3 + N) and torch.equal(out, want)
    assert all(cc.constraint.accepts(row) for row in out[:, 3:].tolist())


class Plain:
    """A circuit whose decoder answers with the circuit's own queries (inside a fast_inference scope)."""

    def __init__(self, cc):
        self.cc, self.n = cc, cc.n

    @contextlib.contextmanager
    def decoder(self, **defaults):
        with juice.fast_inference():
            yield SimpleNamespace(conditional = lambda *a, **k: self.cc.conditional(*a, **{**defaults, **k}))


def test_masked_diffusion(cc):
    *_, diffusion = examples()
    B, MASK = 8, 7

    def run(circuit):
        g = torch.Generator(device = "cuda").manual_seed(0)
        denoiser = lambda x: torch.randn(x.size(0), x.size(1), V, generator = g, device = "cuda")
        scope = dict(torch = torch, cc = circuit, V = V, B = B, MASK = MASK, denoiser = denoiser)
        exec(diffusion, scope)
        return scope["x"]

    x = run(cc)
    assert torch.equal(x, run(Plain(cc)))
    assert not (x == MASK).any() and all(cc.constraint.accepts(row) for row in x.tolist())
