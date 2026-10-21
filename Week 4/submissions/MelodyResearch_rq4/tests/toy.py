"""Small synthetic fixtures: a tiny causal torch stack with real blocks/hooks and a fake tokenizer.
No weights, no downloads, no inference on any real model."""

from __future__ import annotations

import re
from types import SimpleNamespace

import torch
from torch import nn

from rq4 import config as C
from rq4.hooks import CaptureHook, InterventionHook

D, V, NL = 16, 4096, 24


class Block(nn.Module):
    def __init__(self, seed, tuple_out=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = nn.Parameter(torch.randn(D, D, generator=g) * 0.05, requires_grad=False)
        self.tuple_out = tuple_out

    def forward(self, h):
        cm = torch.cumsum(h, 1) / torch.arange(1, h.shape[1] + 1, device=h.device).view(1, -1, 1)  # causal mixing
        out = h + torch.tanh(cm @ self.w.to(h.dtype))
        return (out, None) if self.tuple_out else out


class ToyLM(nn.Module):
    def __init__(self, tuple_out=False, dtype=torch.float32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Parameter(torch.randn(V, D, generator=g), requires_grad=False)
        self.layers = nn.ModuleList([Block(100 + i, tuple_out) for i in range(NL)])
        self.dtype = dtype

    def forward(self, ids):
        h = self.emb[ids].to(self.dtype)
        for blk in self.layers:
            o = blk(h)
            h = o[0] if isinstance(o, tuple) else o
        return h @ self.emb.to(self.dtype).T


class ToyTokenizer:
    """Word/punctuation tokenizer; ids assigned on first sight. Optional merge rule for boundary tests."""

    def __init__(self, merge_colon_space=False):
        self.vocab = {"<unk>": 0}
        self.inv = {0: "<unk>"}
        self.merge = merge_colon_space

    def _id(self, tok):
        if tok not in self.vocab:
            self.vocab[tok] = len(self.vocab)
            self.inv[self.vocab[tok]] = tok
        return self.vocab[tok]

    def _split(self, text):
        toks = re.findall(r"\w+|[^\w\s]", text)
        if self.merge:                      # simulate ':' swallowing the next '(' -> ':('
            out, i = [], 0
            while i < len(toks):
                if toks[i] == ":" and i + 1 < len(toks) and toks[i + 1] == "(":
                    out.append(":("); i += 2
                else:
                    out.append(toks[i]); i += 1
            return out
        return toks

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[self._id(t) for t in self._split(text)])

    def decode(self, ids, **kw):
        return " ".join(self.inv.get(int(i), f"<{int(i)}>") for i in ids)


class ToyBackend:
    d_model, n_layers = D, NL

    def __init__(self, model=None, tokenizer=None, hidden_scale=1.0, seed=0):
        self.model = model or ToyLM(seed=seed)
        self.tok = tokenizer or ToyTokenizer()
        g = torch.Generator().manual_seed(7)
        self.U = torch.randn(V, D, generator=g)
        self.J = {l: torch.randn(D, D, generator=torch.Generator().manual_seed(50 + l)) for l in range(23)}
        self.hidden_scale = hidden_scale
        self.mask = torch.ones(V, dtype=torch.bool)

    def tokenize(self, text):
        return self.tok(text).input_ids

    def decode_token(self, i):
        return self.tok.inv[int(i)]

    def unembedding(self):
        return self.U

    def jacobian(self, layer):
        return self.J[layer]

    def rank_fn(self, logits, ids):
        from rq4.scoring import local_ranks
        return local_ranks(logits, ids)

    def _ids(self, ids):
        return torch.tensor([list(ids)])

    def capture(self, ids, layers, position):
        caps = {l: CaptureHook(self.model.layers[l], position) for l in layers}
        for c in caps.values():
            c.__enter__()
        try:
            with torch.no_grad():
                self.model(self._ids(ids))
        finally:
            for c in caps.values():
                c.__exit__(None, None, None)
        return {l: c.value * self.hidden_scale for l, c in caps.items()}

    def _with_edit(self, edit, fn):
        if edit is None:
            return fn(), None
        hook = InterventionHook(self.model.layers[edit.layer], edit)
        with hook:
            out = fn()
        return out, hook.record

    def logits(self, ids, edit):
        def run():
            with torch.no_grad():
                return self.model(self._ids(ids))[0]
        return self._with_edit(edit, run)

    def readouts(self, prompt, ids, edit):
        pos = len(ids) - 1

        def run():
            caps = {l: CaptureHook(self.model.layers[l], pos) for l in C.READOUT_LAYERS}
            for c in caps.values():
                c.__enter__()
            try:
                with torch.no_grad():
                    full = self.model(self._ids(ids))[0]
            finally:
                for c in caps.values():
                    c.__exit__(None, None, None)
            out = {l: (caps[l].value @ self.J[l]) @ self.U.T for l in C.READOUT_LAYERS}
            out[C.MODEL_OUTPUT_READOUT] = full[pos]
            return out, full[pos]

        (r, ml), rec = self._with_edit(edit, run)
        return r, ml, rec

    def generate(self, ids, edit, max_new):
        def run():
            cur = list(ids)
            with torch.no_grad():
                for _ in range(max_new):
                    cur.append(int(self.model(self._ids(cur))[0, -1].argmax()))
            return self.tok.decode(cur[len(ids):])
        return self._with_edit(edit, run)
