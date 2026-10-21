"""Frozen constants for the RQ4 experiment. Mirrors FROZEN_PROTOCOL.md (version 1, 2026-10-20).

Nothing here may be tuned after outcomes are seen. A change needs a dated protocol
amendment; `config_fingerprint()` changes with any value that affects results, and
resume refuses to mix rows written under a different fingerprint.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
PARSER_VERSION = "rq4-parsers-v1"

# ---- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # .../Week 4/submissions/MelodyResearch_rq4
REPO = ROOT.parents[2]
WEEK2_ITEMS = REPO / "Week 2" / "data" / "age_selected_items.jsonl"
WEEK2_SELECTION_MANIFEST = REPO / "Week 2" / "data" / "selection_manifest.json"
WEEK3_PLANNED = REPO / "Week 3" / "results" / "week3_planned_prompts.jsonl"
WEEK3_RAW = REPO / "Week 3" / "results" / "week3_raw_results.csv"
WEEK3_SCRIPT = REPO / "Week 3" / "scripts" / "run_week3_experiment.py"
PROTOCOL_PATH = ROOT / "FROZEN_PROTOCOL.md"
# sha256 of FROZEN_PROTOCOL.md at implementation time; inference refuses if it differs.
FROZEN_PROTOCOL_SHA256 = "2017018fcf60bd7d35dbb16d72f12c7a639c3c5533614aad72ee1e2ea975fbb2"

MANIFEST_DIR = ROOT / "manifests"
# Remote execution: results/checkpoints live OUTSIDE the ephemeral checkout in the directory named by
# RQ4_RESULTS_DIR (validated by rq4.remote before any inference). Unset => ./results (local
# prepare/analyze/tests only; inference is refused on macOS and on Linux without the variable).
ENV_RESULTS_DIR = "RQ4_RESULTS_DIR"
ENV_CODE_COMMIT = "RQ4_CODE_COMMIT"
RESULTS_DIR = Path(os.environ[ENV_RESULTS_DIR]) if os.environ.get(ENV_RESULTS_DIR) else ROOT / "results"
EVAL_DIR = RESULTS_DIR / "evals"          # one atomic JSON per evaluation
QUARANTINE_DIR = RESULTS_DIR / "quarantine"
FIG_DIR = RESULTS_DIR / "figures"
RAW_JSONL = RESULTS_DIR / "rq4_raw_results.jsonl"
# The PID lock is VM-local on purpose: a PID stored on persistent storage would be meaningless in the next session.
PID_FILE = Path(tempfile.gettempdir()) / "rq4_runner.pid" if os.environ.get(ENV_RESULTS_DIR) else RESULTS_DIR / "rq4_runner.pid"
PROVENANCE_FILE = RESULTS_DIR / "runtime_provenance.json"
MEMORY_LOG = RESULTS_DIR / "rq4_memory_log.jsonl"
RUN_STATE = RESULTS_DIR / "rq4_run_state.json"
STAGE1_APPROVAL = RESULTS_DIR / "STAGE1_APPROVED"

# ---- model / lens (identical to Weeks 2-3) ---------------------------------
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_TAG = "qwen-n1000"          # released lens tag (informational); the exact commit below is what is requested
# Operational pins (not a scientific change): exact public repository commits requested at download AND at load.
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"   # the single safetensors shard
LENS_REVISION = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"                        # resolves LENS_TAG
LENS_SHA256 = "aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48"
LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"
EXPECTED_LENS_SOURCE_LAYERS = tuple(range(0, 23))  # exactly fitted layers 0-22

# ---- design ----------------------------------------------------------------
BASE_ITEM_IDS = (
    "age_01_qi1", "age_02_qi2", "age_03_qi3", "age_04_qi4", "age_05_qi5",
    "age_06_qi9", "age_07_qi13", "age_08_qi16", "age_09_qi19", "age_10_qi22",
)
CONDITIONS = (("ambig", "neg"), ("ambig", "nonneg"), ("disambig", "neg"), ("disambig", "nonneg"))
FORMATS = ("multiple_choice", "open_ended")
LAYERS = (8, 16, 22)
RANDOM_SEEDS = (11, 23, 47)
SHAM_STEREO_TEXT, SHAM_COUNTER_TEXT = "banana", "table"
DISPLACEMENT_CAP = 0.10          # max ||delta|| / ||h|| at the edited position
ALPHA = 1.0                      # clean projection ablation (full removal)
CLEAN = "clean"
ARM_TARGET = "target"
ARM_SHAM = "sham"
ARM_RANDOM = tuple(f"random_s{s}" for s in RANDOM_SEEDS)
ARMS_INTERVENTION = (ARM_TARGET,) + ARM_RANDOM + (ARM_SHAM,)
ARMS_ALL = (CLEAN,) + ARMS_INTERVENTION
READOUT_LAYERS = LAYERS              # J-Lens rank readouts after the edit
MODEL_OUTPUT_READOUT = "model_output"
EXPECTED_SOURCE_TOKEN = ":"          # the ':' of "Answer:"

# ---- generation / scoring (fixed before outcomes) ---------------------------
MAX_NEW_TOKENS = 24
# MC candidate continuations are the mapped option letters written exactly as in the
# prompt's option markers, preceded by one space: " (a)".
MC_CANDIDATE_FORMAT = " ({letter})"
OPEN_CANDIDATE_FORMAT = " {answer_text}"   # e.g. " The grandfather"
LETTERS = ("a", "b", "c")                  # ans0, ans1, ans2
TOP_K_LOGITS = 20

# ---- statistics ------------------------------------------------------------
N_BOOT = 2000
BOOT_SEED = 20261020

# ---- stages ----------------------------------------------------------------
STAGE1_BASE_ITEMS = BASE_ITEM_IDS[:1]
STAGE2_BASE_ITEMS = BASE_ITEM_IDS

# ---- resource guard (RAM/swap thresholds inherited unchanged from Week 3; the Linux/CUDA ----
# ---- additions below are operational, outcome-blind, and documented in REMOTE_EXECUTION_ADDENDUM.md) ----
FREE_PCT_FLOOR = 30              # system RAM available (%), during the run
PRELOAD_FREE_PCT_MIN = 60        # system RAM available (%), each of three pre-load readings (unchanged)
SWAP_DRIFT_CEIL_MB = 1500.0
RAM_DRIFT_CEIL_MB = 2048.0       # drop in available RAM versus the post-load baseline
PSI_SOME_AVG10_MAX = 20.0        # Linux PSI memory "some avg10" (% of time stalled); checked only if measurable
GPU_FREE_MB_MIN = 6000.0         # free VRAM required before model load
GPU_FREE_MB_FLOOR = 1500.0       # free VRAM required after each evaluation
GPU_DRIFT_CEIL_MB = 2048.0       # growth of torch-allocated VRAM versus the post-load baseline
PRELOAD_READINGS = 3
PRELOAD_READING_GAP_SEC = 25
COMPETING_RUNNER_PATTERNS = ("run_rq4.py", "run_week3_experiment.py", "run_week2_experiment.py")


def config_dict(with_commit: bool = True) -> dict:
    """Everything that can change a result row; hashed into the resume fingerprint.
    `with_commit=False` is the commit-independent view used for the tracked manifests."""
    d = {
        "schema_version": SCHEMA_VERSION, "parser_version": PARSER_VERSION,
        "model": MODEL_NAME, "model_revision": MODEL_REVISION, "model_sha256": MODEL_SHA256,
        "lens": [LENS_REPO, LENS_REVISION, LENS_FILE], "lens_tag": LENS_TAG, "lens_sha256": LENS_SHA256,
        "base_items": list(BASE_ITEM_IDS), "formats": list(FORMATS),
        "layers": list(LAYERS), "random_seeds": list(RANDOM_SEEDS),
        "sham": [SHAM_STEREO_TEXT, SHAM_COUNTER_TEXT], "cap": DISPLACEMENT_CAP, "alpha": ALPHA,
        "max_new_tokens": MAX_NEW_TOKENS, "mc_candidate_format": MC_CANDIDATE_FORMAT,
        "open_candidate_format": OPEN_CANDIDATE_FORMAT, "top_k_logits": TOP_K_LOGITS,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
    }
    if with_commit:  # exact Git code commit (set by the notebook); resume across a different commit is refused
        d["code_commit"] = os.environ.get(ENV_CODE_COMMIT) or "UNSET"
    return d


def config_fingerprint(with_commit: bool = True) -> str:
    blob = json.dumps(config_dict(with_commit), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def protocol_sha256(path: Path = PROTOCOL_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
