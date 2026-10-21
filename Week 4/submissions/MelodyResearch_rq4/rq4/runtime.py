"""Real backend: HF Qwen3.5-0.8B + released Jacobian Lens on a remote Linux CUDA GPU. Heavy imports happen only here.

REMOTE-ONLY: never instantiated on macOS / without CUDA (the constructor refuses). Not exercised in Stage 0. The jlens calls mirror Week 3's
run_week3_experiment.py (from_hf, lens.apply, lens.jacobians, vis._meaningful_token_mask/_ranks_of)
and csinatra's interventions.py (model.layers[l], model._hf_model); they must be validated by the
Stage 1 preflight (`check_lens_block_alignment` etc.) before any scientific inference.
"""

from __future__ import annotations

import sys
from contextlib import ExitStack
from importlib import metadata

from . import config as C
from .hooks import CaptureHook, InterventionHook


def choose_dtype(torch_mod) -> str:
    """bf16 where the GPU natively supports it, else fp16 (T4-class). Emulated bf16 does not count."""
    try:
        bf16 = torch_mod.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:                       # older torch without the keyword
        bf16 = torch_mod.cuda.is_bf16_supported()
    return "bfloat16" if bf16 else "float16"


def _jlens_version() -> str:
    try:
        dist = metadata.distribution("jlens")
        raw = dist.read_text("direct_url.json")
        if raw:
            import json
            commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
            if commit:
                return f"{dist.version}@{commit}"
        return dist.version
    except Exception:  # noqa: BLE001
        return "unknown"


class QwenBackend:
    def __init__(self):
        if sys.platform != "linux":
            raise RuntimeError("QwenBackend is remote-only (Linux CUDA); refusing on " + sys.platform)
        import torch
        import transformers

        import jlens
        from jlens import vis

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU visible; refusing to load the model (no CPU/MPS fallback)")
        self.torch, self.vis = torch, vis
        jlens.configure_logging()
        self.dtype_name = choose_dtype(torch)
        print(f"Loading {C.MODEL_NAME} on CUDA ({self.dtype_name}) ...", file=sys.stderr)
        hf = transformers.AutoModelForCausalLM.from_pretrained(C.MODEL_NAME, revision=C.MODEL_REVISION, dtype=getattr(torch, self.dtype_name)).to("cuda")
        hf.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(C.MODEL_NAME, revision=C.MODEL_REVISION)
        self.model = jlens.from_hf(hf, self.tokenizer)
        self.hf = hf
        self.lens = jlens.JacobianLens.from_pretrained(C.LENS_REPO, filename=C.LENS_FILE, revision=C.LENS_REVISION)
        self.d_model, self.n_layers = self.model.d_model, self.model.n_layers
        self.device = self.model.input_device
        if getattr(self.device, 'type', str(self.device).split(':')[0]) != 'cuda':
            raise RuntimeError(f'model input device is {self.device}, expected cuda (no CPU/MPS fallback)')
        self._mask = None
        self._U = None

    def provenance(self) -> dict:
        t = self.torch
        props = t.cuda.get_device_properties(0)
        return {"platform": sys.platform, "dtype": self.dtype_name, "device": str(self.device),
                "gpu_name": props.name, "gpu_capability": list(t.cuda.get_device_capability(0)),
                "gpu_total_mb": props.total_memory / 1e6, "cuda": t.version.cuda, "torch": t.__version__,
                "model_revision": C.MODEL_REVISION, "lens_revision": C.LENS_REVISION,
                "transformers": metadata.version("transformers"), "jlens": _jlens_version()}

    # ---- static checks (called by preflight) ----
    def lens_info(self) -> dict:
        return {"source_layers": list(self.lens.source_layers), "d_model": self.lens.d_model,
                "n_prompts": self.lens.n_prompts, "model_d_model": self.d_model, "model_n_layers": self.n_layers}

    def meaningful_mask(self):
        if self._mask is None:
            vocab = self.unembedding().shape[0]
            self._mask = self.vis._meaningful_token_mask(self.tokenizer, vocab, self.device).cpu()
        return self._mask

    # ---- tokenisation ----
    def tokenize(self, text):
        # Match Week 3 and lens.apply exactly, including any model-specific
        # special-token policy set by jlens.from_hf.
        return [int(i) for i in self.model.encode(text)[0].tolist()]

    def phrase_tokenize(self, text):
        """Target/sham phrase ids: the HF tokenizer with NO special tokens (frozen protocol)."""
        return [int(i) for i in self.tokenizer(text, add_special_tokens=False).input_ids]

    def decode_token(self, i):
        return self.tokenizer.decode([int(i)], clean_up_tokenization_spaces=False)

    def unembedding(self):
        """CPU copy (cached): directions are built on CPU so U and J never sit on different devices."""
        if self._U is None:
            self._U = self.hf.get_output_embeddings().weight.detach().cpu()
        return self._U

    def jacobian(self, layer):
        return self.jacobians_cpu(layer)

    def rank_fn(self, logits_1d, ids):
        t = self.torch
        ids_t = t.as_tensor(list(ids), dtype=t.long)
        return [int(r) for r in self.vis._ranks_of(logits_1d.unsqueeze(0), ids_t.unsqueeze(0))[0].tolist()]

    def _ids(self, ids):
        return self.torch.tensor([list(ids)], dtype=self.torch.long, device=self.device)

    # ---- forward helpers ----
    @staticmethod
    def _module(model, layer):
        return model.layers[layer]

    def capture(self, ids, layers, position):
        t = self.torch
        with ExitStack() as stack:      # a registration failure part-way unwinds every hook already entered
            caps = {l: stack.enter_context(CaptureHook(self._module(self.model, l), position)) for l in layers}
            with t.inference_mode():
                self.hf(input_ids=self._ids(ids), use_cache=False)
        return {l: c.value.cpu() for l, c in caps.items()}

    def _with_edit(self, edit, fn):
        if edit is None:
            return fn(), None
        hook = InterventionHook(self._module(self.model, edit.layer), edit)
        with hook:
            out = fn()
        return out, hook.record

    def logits(self, ids, edit):
        t = self.torch

        def run():
            with t.inference_mode():
                return self.hf(input_ids=self._ids(ids), use_cache=False).logits[0].float().cpu()

        return self._with_edit(edit, run)

    def readouts(self, prompt, ids, edit):
        pos = len(ids) - 1

        def run():
            with self.torch.inference_mode():
                lens_logits, model_logits, _ = self.lens.apply(self.model, prompt, layers=list(C.READOUT_LAYERS), positions=[pos])
            out = {l: lens_logits[l][0].float().cpu() for l in C.READOUT_LAYERS}
            out[C.MODEL_OUTPUT_READOUT] = model_logits[0].float().cpu()
            return out, model_logits[0].float().cpu()

        (readouts, model_logits), rec = self._with_edit(edit, run)
        return readouts, model_logits, rec

    def generate(self, ids, edit, max_new):
        t = self.torch
        inp = self._ids(ids)

        def run():
            with t.inference_mode():
                out = self.hf.generate(inp, max_new_tokens=max_new, do_sample=False,
                                       pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(out[0, inp.shape[1]:], skip_special_tokens=False)

        return self._with_edit(edit, run)

    def lens_formulas(self):
        """Candidate reconstructions of the released-lens readout from a block output h [d]."""
        t = self.torch
        U = self.unembedding().float()
        J = lambda l: self.jacobians_cpu(l)
        return {
            "h@J @ U.T": lambda l, h: (h.float() @ J(l)) @ U.T,
            "h@J.T @ U.T": lambda l, h: (h.float() @ J(l).T) @ U.T,
            "unembed(h@J)": lambda l, h: self.model.unembed((h.float() @ J(l)).to(self.device)).float().cpu().flatten(),
            "unembed(h@J.T)": lambda l, h: self.model.unembed((h.float() @ J(l).T).to(self.device)).float().cpu().flatten(),
        }

    def jacobians_cpu(self, layer):
        cache = self.__dict__.setdefault("_jcache", {})
        if layer not in cache:
            cache[layer] = self.lens.jacobians[layer].float().cpu()
        return cache[layer]

    def release(self):
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()
