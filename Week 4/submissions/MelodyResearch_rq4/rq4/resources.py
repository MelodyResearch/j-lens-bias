"""Resource guard for remote Linux/CUDA: RAM, swap, PSI memory pressure, GPU VRAM, PID lock, duplicate jobs.

Parsers and the decision rules are pure functions so they are unit-tested without a GPU. RAM/swap
thresholds are inherited unchanged from Week 3 (config.py). Unreadable telemetry fails closed, except
Linux PSI memory pressure, which is "if measurable" (containers often hide /proc/pressure).
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


def _kb(text: str, key: str) -> float | None:
    m = re.search(rf"^{key}:\s+(\d+)\s*kB", text, re.M)
    return float(m.group(1)) if m else None


def parse_meminfo(text: str) -> dict:
    """{total_mb, available_mb, swap_total_mb, swap_used_mb} from /proc/meminfo text; None where absent."""
    tot, av = _kb(text, "MemTotal"), _kb(text, "MemAvailable")
    st, sf = _kb(text, "SwapTotal"), _kb(text, "SwapFree")
    mb = lambda v: None if v is None else v / 1024.0
    return {"total_mb": mb(tot), "available_mb": mb(av), "swap_total_mb": mb(st),
            "swap_used_mb": None if st is None or sf is None else (st - sf) / 1024.0}


def parse_cgroup_memory(limit_text: str | None, usage_text: str | None) -> tuple[float, float] | None:
    """(limit_mb, usage_mb) from cgroup memory files, or None if unlimited/unreadable."""
    try:
        lim = (limit_text or "").strip()
        if not lim or lim == "max" or int(lim) >= 1 << 60:
            return None
        return int(lim) / 1048576.0, int((usage_text or "").strip()) / 1048576.0
    except ValueError:
        return None


def parse_psi_memory(text: str) -> dict | None:
    """{some_avg10, full_avg10} from /proc/pressure/memory, else None."""
    out = {}
    for kind in ("some", "full"):
        m = re.search(rf"^{kind}\s+avg10=([\d.]+)", text, re.M)
        if m:
            out[f"{kind}_avg10"] = float(m.group(1))
    return out if "some_avg10" in out else None


def parse_nvidia_smi(text: str) -> dict | None:
    """First GPU of `nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits`."""
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*$", line)
        if m:
            return {"gpu_total_mb": float(m.group(1)), "gpu_used_mb": float(m.group(2)), "gpu_free_mb": float(m.group(3))}
    return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return ""


def snapshot(torch_mod=None, *, read=_read, run=_run) -> dict:
    """One telemetry reading. `free_pct` = available RAM as % of total (cgroup-limit aware)."""
    mi = parse_meminfo(read("/proc/meminfo") or "")
    total, avail = mi["total_mb"], mi["available_mb"]
    cg = parse_cgroup_memory(read("/sys/fs/cgroup/memory.max"), read("/sys/fs/cgroup/memory.current")) or \
        parse_cgroup_memory(read("/sys/fs/cgroup/memory/memory.limit_in_bytes"), read("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    if cg and total is not None and avail is not None and cg[0] < total:
        total, avail = cg[0], min(avail, cg[0] - cg[1])
    psi = parse_psi_memory(read("/proc/pressure/memory") or "")
    gpu = parse_nvidia_smi(run(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                                "--format=csv,noheader,nounits"])) or {}
    snap = {
        "ts": time.time(),
        "free_pct": None if not total or avail is None else round(100.0 * avail / total, 2),
        "available_mb": avail, "total_mb": total, "swap_used_mb": mi["swap_used_mb"],
        "psi_some_avg10": psi["some_avg10"] if psi else None, "psi_full_avg10": psi.get("full_avg10") if psi else None,
        "gpu_total_mb": gpu.get("gpu_total_mb"), "gpu_free_mb": gpu.get("gpu_free_mb"),
        "gpu_alloc_mb": None, "gpu_reserved_mb": None,
    }
    if torch_mod is not None and torch_mod.cuda.is_available():
        try:
            snap["gpu_alloc_mb"] = torch_mod.cuda.memory_allocated() / 1e6
            snap["gpu_reserved_mb"] = torch_mod.cuda.memory_reserved() / 1e6
        except Exception:  # noqa: BLE001
            pass
    return snap


def post_load_baseline(pre: dict, post: dict) -> dict:
    """Baseline for is_safe after model load: RAM/GPU allocation from `post` (the pre snapshot is taken without
    torch, so its gpu_alloc_mb is None), swap from `pre` so swap growth during the load itself is still caught."""
    return {**post, "swap_used_mb": pre.get("swap_used_mb")}


def is_safe(snap: dict, baseline: dict) -> tuple[bool, str]:
    """Stop conditions during the run: unreadable RAM/swap/GPU telemetry, low RAM, swap or RAM growth versus
    the post-load baseline, high memory pressure (if measurable), low free VRAM, torch VRAM growth."""
    if snap.get("free_pct") is None:
        return False, "free_pct unreadable (fail closed)"
    if snap["free_pct"] < C.FREE_PCT_FLOOR:
        return False, f"free_pct {snap['free_pct']}% < {C.FREE_PCT_FLOOR}%"
    if snap.get("swap_used_mb") is None:
        return False, "swap unreadable (fail closed)"
    if snap.get("psi_some_avg10") is not None and snap["psi_some_avg10"] > C.PSI_SOME_AVG10_MAX:
        return False, f"memory pressure some avg10 {snap['psi_some_avg10']} > {C.PSI_SOME_AVG10_MAX}"
    b = baseline.get("swap_used_mb")
    if b is not None and snap["swap_used_mb"] - b >= C.SWAP_DRIFT_CEIL_MB:
        return False, f"swap grew {snap['swap_used_mb'] - b:.0f}MB (ceiling {C.SWAP_DRIFT_CEIL_MB})"
    ba, sa = baseline.get("available_mb"), snap.get("available_mb")
    if ba is not None and sa is not None and ba - sa >= C.RAM_DRIFT_CEIL_MB:
        return False, f"available RAM fell {ba - sa:.0f}MB (ceiling {C.RAM_DRIFT_CEIL_MB})"
    if snap.get("gpu_free_mb") is None:
        return False, "GPU free memory unreadable (fail closed)"
    if snap["gpu_free_mb"] < C.GPU_FREE_MB_FLOOR:
        return False, f"free VRAM {snap['gpu_free_mb']:.0f}MB < {C.GPU_FREE_MB_FLOOR:.0f}MB"
    ga, gb = snap.get("gpu_alloc_mb"), baseline.get("gpu_alloc_mb")
    if ga is None or gb is None:
        return False, "torch GPU allocation unreadable (fail closed)"
    if ga - gb > C.GPU_DRIFT_CEIL_MB:
        return False, f"GPU allocation grew {ga - gb:.0f}MB (ceiling {C.GPU_DRIFT_CEIL_MB})"
    return True, "ok"


def preload_gate(snapshot_fn=snapshot, sleep_fn=time.sleep, log_fn=None) -> tuple[bool, str]:
    """Three readings spaced PRELOAD_READING_GAP_SEC apart; every one must pass (>=60% available RAM, readable
    swap, low pressure if measurable, enough free VRAM). `log_fn(tag, snap)` records each reading."""
    for i in range(1, C.PRELOAD_READINGS + 1):
        if i > 1:
            sleep_fn(C.PRELOAD_READING_GAP_SEC)
        s = snapshot_fn()
        if log_fn:
            log_fn(f"preload_reading_{i}", s)
        if s.get("free_pct") is None or s.get("swap_used_mb") is None or s.get("gpu_free_mb") is None:
            return False, f"reading {i}: telemetry unreadable (fail closed)"
        if s["free_pct"] < C.PRELOAD_FREE_PCT_MIN:
            return False, f"reading {i}: free_pct {s['free_pct']}% < {C.PRELOAD_FREE_PCT_MIN}%"
        if s.get("psi_some_avg10") is not None and s["psi_some_avg10"] > C.PSI_SOME_AVG10_MAX:
            return False, f"reading {i}: memory pressure some avg10 {s['psi_some_avg10']} > {C.PSI_SOME_AVG10_MAX}"
        if s["gpu_free_mb"] < C.GPU_FREE_MB_MIN:
            return False, f"reading {i}: free VRAM {s['gpu_free_mb']:.0f}MB < {C.GPU_FREE_MB_MIN:.0f}MB"
    return True, "ok"


def log_snapshot(tag: str, snap: dict, path: Path = C.MEMORY_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"tag": tag, **snap}) + "\n")
    print(f"[memory:{tag}] {snap}", file=sys.stderr)


def competing_processes(ps_output: str, my_pid: int, my_ppid: int, patterns=C.COMPETING_RUNNER_PATTERNS) -> list[str]:
    """Lines of `ps -eo pid,command` that look like another model job (python running a runner)."""
    hits = []
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, cmd = int(parts[0]), parts[1]
        if pid in (my_pid, my_ppid):
            continue
        first = cmd.split()[0] if cmd.split() else ""
        if Path(first).name.startswith("python") and any(p in cmd for p in patterns):
            hits.append(line.strip())
    return hits


def refuse_if_duplicate() -> None:
    hits = competing_processes(_run(["ps", "-eo", "pid,args"]), os.getpid(), os.getppid())
    if hits:
        raise RuntimeError("duplicate model job detected: " + " | ".join(hits))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        # A process we cannot signal is still present; fail closed on the lock.
        return True
    except ProcessLookupError:
        return False
    return True


def acquire_lock(path: Path = C.PID_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            pid = int(path.read_text().strip())
        except ValueError:
            pid = -1
        if pid > 0 and _alive(pid) and pid != os.getpid():
            raise RuntimeError(f"another RQ4 runner is active (pid {pid})")
    path.write_text(str(os.getpid()))


def release_lock(path: Path = C.PID_FILE) -> None:
    try:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass
