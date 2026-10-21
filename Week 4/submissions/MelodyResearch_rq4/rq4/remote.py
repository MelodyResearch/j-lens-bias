"""Remote-only execution guards. No torch/transformers import; everything here is unit-testable on any OS.

* inference is refused on anything but Linux (macOS in particular), before any model code is imported;
* results/checkpoints must live in an explicit absolute RQ4_RESULTS_DIR on persistent storage;
* the exact Git code commit (RQ4_CODE_COMMIT) must be a full SHA that matches the checkout's HEAD;
* runtime provenance (dtype, versions) is pinned on first use and must match on every resume.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import config as C

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
PLACEHOLDER = "__RQ4_CODE_COMMIT__"
EPHEMERAL_PREFIXES = ("/tmp", "/var/tmp", "/dev/shm")
PERSISTENT_FSTYPE_PREFIXES = ("fuse",)      # Google Drive in Colab is a FUSE mount at /content/drive


class RemoteGuardError(RuntimeError):
    pass


# ---- platform -----------------------------------------------------------------------------------------
def refuse_inference_on_this_platform(platform: str | None = None) -> None:
    """Called first thing for ANY --run-inference request, whatever other flags were given."""
    platform = sys.platform if platform is None else platform
    if platform == "darwin":
        raise SystemExit("REFUSING: inference is remote-only (Linux CUDA, e.g. Google Colab). It is never run on macOS. "
                         "Prepare/analyze/tests remain available locally.")
    if not platform.startswith("linux"):
        raise SystemExit(f"REFUSING: inference needs a Linux CUDA VM; platform is {platform!r}.")


# ---- results directory --------------------------------------------------------------------------------
def parse_proc_mounts(text: str) -> list[tuple[str, str]]:
    """[(mountpoint, fstype)] from /proc/mounts text."""
    out = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out.append((parts[1].replace("\\040", " "), parts[2]))
    return out


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def read_proc_mounts() -> str:
    try:
        return Path("/proc/mounts").read_text()
    except OSError:
        return ""


def validate_results_dir(raw: str | None, *, mounts_text: str | None = None, realpath=os.path.realpath,
                         code_root: Path = C.ROOT) -> Path:
    """Fail closed. Returns the resolved directory path (not yet created)."""
    if not raw or not raw.strip():
        raise RemoteGuardError(f"{C.ENV_RESULTS_DIR} is unset: refusing remote inference (results must be persistent)")
    if not os.path.isabs(raw):
        raise RemoteGuardError(f"{C.ENV_RESULTS_DIR}={raw!r} must be an absolute path")
    if ".." in Path(raw).parts:
        raise RemoteGuardError(f"{C.ENV_RESULTS_DIR}={raw!r} must not contain '..'")
    real = realpath(raw)
    if _within(real, str(realpath(str(code_root)))):
        raise RemoteGuardError(f"results dir {real} is inside the (ephemeral) code checkout")
    for pref in EPHEMERAL_PREFIXES:
        if _within(real, pref):
            raise RemoteGuardError(f"results dir {real} is under ephemeral {pref}")
    if _within(real, "/content"):
        mounts = parse_proc_mounts(read_proc_mounts() if mounts_text is None else mounts_text)
        ok = [mp for mp, fs in mounts
              if mp not in ("/", "/content") and _within(mp, "/content") and fs.startswith(PERSISTENT_FSTYPE_PREFIXES)
              and _within(real, mp) and real != mp.rstrip("/")]
        if not ok:
            raise RemoteGuardError(f"results dir {real} is under /content without a persistent (Drive/FUSE) mount; "
                                   f"mount Google Drive first and use a folder inside it")
    return Path(real)


def ensure_writable(path: Path) -> None:
    """Create the directory and prove the atomic write -> fsync -> rename pattern works there."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RemoteGuardError(f"cannot create results dir {path}: {exc}") from exc
    probe = path / f".rq4_write_probe_{os.getpid()}"
    tmp = probe.with_name(probe.name + ".tmp")
    try:
        with open(tmp, "w") as f:
            f.write("ok")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, probe)
        if probe.read_text() != "ok":
            raise RemoteGuardError(f"{path}: write probe read back wrongly")
    except OSError as exc:
        raise RemoteGuardError(f"{path} is not writable with atomic rename: {exc}") from exc
    finally:
        for q in (tmp, probe):
            try:
                q.unlink()
            except OSError:
                pass


# ---- code commit --------------------------------------------------------------------------------------
def git_head(root: Path = C.ROOT) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def validate_code_commit(env_value: str | None, head: str | None) -> str:
    if not env_value or env_value == PLACEHOLDER:
        raise RemoteGuardError(f"{C.ENV_CODE_COMMIT} is unset or still the placeholder: refusing remote inference")
    if not SHA_RE.match(env_value):
        raise RemoteGuardError(f"{C.ENV_CODE_COMMIT} must be a full lowercase 40-hex SHA, got {env_value!r}")
    if head is None:
        raise RemoteGuardError("cannot read git HEAD of the checkout (fail closed)")
    if head != env_value:
        raise RemoteGuardError(f"checkout HEAD {head} != {C.ENV_CODE_COMMIT} {env_value}")
    return env_value


def require_remote_config(env=None, *, platform: str | None = None, mounts_text: str | None = None, head=None) -> dict:
    """Everything that must hold before any model code is touched. Returns the validated settings."""
    env = os.environ if env is None else env
    refuse_inference_on_this_platform(platform)
    raw = env.get(C.ENV_RESULTS_DIR)
    results = validate_results_dir(raw, mounts_text=mounts_text)
    if Path(raw) != C.RESULTS_DIR:
        raise RemoteGuardError("config.RESULTS_DIR does not match RQ4_RESULTS_DIR (set the variable before starting Python)")
    commit = validate_code_commit(env.get(C.ENV_CODE_COMMIT), git_head() if head is None else head)
    ensure_writable(results)
    return {"results_dir": str(results), "code_commit": commit}


# ---- runtime provenance -------------------------------------------------------------------------------
PINNED_KEYS = ("code_commit", "config_fingerprint", "dtype", "model_revision", "lens_revision", "torch", "transformers", "jlens")


def merge_provenance(existing: dict | None, current: dict) -> dict:
    """First session pins {code_commit, fingerprint, dtype, library versions}; later sessions must match them
    (GPU model may differ; dtype may not). Returns the updated document."""
    pinned = {k: current.get(k) for k in PINNED_KEYS if k in current}
    if existing is None:
        return {"pinned": pinned, "sessions": [current]}
    diffs = {k: (existing["pinned"].get(k), pinned.get(k)) for k in pinned if existing["pinned"].get(k) != pinned.get(k)}
    if diffs:
        raise RemoteGuardError(f"runtime provenance differs from the first session (refusing to mix): {diffs}")
    return {"pinned": existing["pinned"], "sessions": existing["sessions"] + [current]}


def record_provenance(current: dict, path: Path = C.PROVENANCE_FILE) -> dict:
    from .store import write_atomic
    existing = json.loads(path.read_text()) if path.exists() else None
    doc = merge_provenance(existing, {**current, "recorded_at": time.time()})
    write_atomic(path, doc)
    return doc
