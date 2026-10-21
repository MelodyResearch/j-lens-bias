import importlib.util
import json
import os

import pytest

from rq4 import config as C
from rq4 import data, resources, store


# ---- prompt matching --------------------------------------------------------------------------------
def _week3_module():
    spec = importlib.util.spec_from_file_location("w3", C.WEEK3_SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)            # stdlib-only imports at module level; no torch/jlens
    return m


def test_prompts_identical_to_week3_builder_and_saved_plan():
    w3 = _week3_module()
    rows = data.load_items()
    saved = {(r["pair_id"], r["prompt_format"]): r for r in map(json.loads, C.WEEK3_PLANNED.read_text().splitlines())}
    plan = data.plan_prompts(rows)
    assert len(plan) == 80 == len(saved)
    for p, row in ((p, next(r for r in rows if data.pair_id_for(r) == p["pair_id"])) for p in plan):
        assert p["prompt"] == w3.build_prompt(p["prompt_format"], row) == saved[(p["pair_id"], p["prompt_format"])]["prompt"]
        assert p["prompt"].endswith("Answer:")
    assert w3.PROMPT_TEMPLATE_MC == data.PROMPT_TEMPLATE_MC and w3.PROMPT_TEMPLATE_OPEN == data.PROMPT_TEMPLATE_OPEN
    assert w3.pair_id_for(rows[0]) == data.pair_id_for(rows[0])


def test_matched_pairs_differ_only_by_options():
    plan = data.plan_prompts(data.load_items())
    by = {}
    for p in plan:
        by.setdefault(p["pair_id"], {})[p["prompt_format"]] = p
    assert len(by) == 40
    for pair in by.values():
        mc, oe = pair["multiple_choice"], pair["open_ended"]
        assert mc["prompt"].startswith(oe["prompt"][: -len("\nAnswer:")]) and mc["prompt_hash"] != oe["prompt_hash"]
        assert mc["stereotype_target_text"] == oe["stereotype_target_text"] and mc["correct_role"] == oe["correct_role"]


def test_frozen_items_and_week3_positions():
    rows = data.load_items()
    assert tuple(dict.fromkeys(r["base_item_id"] for r in rows)) == C.BASE_ITEM_IDS and len(C.BASE_ITEM_IDS) == 10
    ref = data.week3_reference_positions()
    assert len(ref) == 80 and all(v["source_token"].strip() == ":" and v["analysis_position"] == v["seq_len"] - 1 for v in ref.values())


def test_protocol_hash_pinned_and_lens_layers():
    assert C.protocol_sha256() == C.FROZEN_PROTOCOL_SHA256, "FROZEN_PROTOCOL.md changed; needs dated amendment"
    assert C.EXPECTED_LENS_SOURCE_LAYERS == tuple(range(23)) and C.LAYERS == (8, 16, 22)
    assert C.DISPLACEMENT_CAP == 0.10 and C.N_BOOT == 2000 and C.BOOT_SEED == 20261020


def test_planned_keys_unique_and_counts():
    keys = [store.eval_key(p["prompt_id"], a, l) for p, a, l in store.planned_keys(data.plan_prompts(data.load_items()))]
    assert len(keys) == len(set(keys)) == 80 * 16


# ---- store --------------------------------------------------------------------------------------------
def row(prompt_id="p1", arm="target", layer=8, fp="f", **kw):
    r = {"schema_version": C.SCHEMA_VERSION, "key": store.eval_key(prompt_id, arm, layer), "config_fingerprint": fp,
         "stage": "1", "base_item_id": "b", "pair_id": "pp", "prompt_id": prompt_id, "context_condition": "ambig",
         "question_polarity": "neg", "prompt_format": "open_ended", "prompt_hash": "h", "arm": arm, "layer": layer, "status": "ok"}
    r.update(kw)
    return r


def test_atomic_write_unique_and_duplicate_refused(tmp_path):
    p = store.save_row(tmp_path, row())
    assert p.exists() and not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        store.save_row(tmp_path, row())
    with pytest.raises(ValueError):
        store.save_row(tmp_path, row(layer=None))                 # non-clean arm without a layer
    with pytest.raises(ValueError):
        store.save_row(tmp_path, {**row(arm="clean", layer=None), "key": "wrong"})
    with pytest.raises(ValueError):
        store.write_atomic(tmp_path / "x.json", {"v": float("nan")})   # NaN never reaches disk
    assert not (tmp_path / "x.json").exists()


def test_resume_quarantines_corrupt_and_stale_and_rejects_foreign_fingerprint(tmp_path):
    q = tmp_path / "q"
    store.save_row(tmp_path, row("p1"))
    store.save_row(tmp_path, row("p2"))
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "p3__target__L8.json.tmp").write_text("{}")
    (tmp_path / "p9__target__L8.json").write_text(json.dumps({"key": "p9__target__L8"}))    # schema-invalid
    got = store.load_valid_rows(tmp_path, "f", q)
    assert sorted(got) == ["p1__target__L8", "p2__target__L8"] and len(list(q.iterdir())) == 3
    with pytest.raises(store.FingerprintMismatch):
        store.load_valid_rows(tmp_path, "other", q)


def test_combine_unique(tmp_path):
    for i in range(3):
        store.save_row(tmp_path / "e", row(f"p{i}"))
    out = tmp_path / "all.jsonl"
    assert store.combine(tmp_path / "e", out, "f") == 3
    keys = [json.loads(l)["key"] for l in out.read_text().splitlines()]
    assert keys == sorted(set(keys))


# ---- resources ----------------------------------------------------------------------------------------
MEMINFO = "MemTotal:       13000000 kB\nMemFree: 1000 kB\nMemAvailable:   10400000 kB\nSwapTotal: 2048000 kB\nSwapFree: 2000000 kB\n"
SMI = "15360, 1000, 14360\n"


def good_snap(**kw):
    s = {"free_pct": 80.0, "available_mb": 10000.0, "swap_used_mb": 10.0, "psi_some_avg10": 0.5,
         "gpu_free_mb": 14000.0, "gpu_alloc_mb": 1700.0}
    s.update(kw)
    return s


def test_linux_telemetry_parsers():
    mi = resources.parse_meminfo(MEMINFO)
    assert mi["total_mb"] == pytest.approx(13000000 / 1024) and mi["available_mb"] == pytest.approx(10400000 / 1024)
    assert mi["swap_used_mb"] == pytest.approx(48000 / 1024)
    assert resources.parse_meminfo("garbage")["available_mb"] is None
    assert resources.parse_psi_memory("some avg10=1.50 avg60=0.5 avg300=0.1 total=5\nfull avg10=0.25 avg60=0 avg300=0 total=1\n") == {"some_avg10": 1.5, "full_avg10": 0.25}
    assert resources.parse_psi_memory("") is None
    assert resources.parse_nvidia_smi("header\n" + SMI) == {"gpu_total_mb": 15360.0, "gpu_used_mb": 1000.0, "gpu_free_mb": 14360.0}
    assert resources.parse_nvidia_smi("No devices were found") is None
    assert resources.parse_cgroup_memory("max\n", "123") is None
    assert resources.parse_cgroup_memory(str(8 * 2**30), str(2 * 2**30)) == (8192.0, 2048.0)


def test_snapshot_from_fake_proc_is_cgroup_aware_and_fail_closed():
    files = {"/proc/meminfo": MEMINFO, "/sys/fs/cgroup/memory.max": str(6 * 2**30), "/sys/fs/cgroup/memory.current": str(4 * 2**30)}
    s = resources.snapshot(read=files.get, run=lambda c: SMI)
    assert s["free_pct"] == pytest.approx(100 * 2048 / 6144, abs=0.01) and s["gpu_free_mb"] == 14360.0
    assert s["psi_some_avg10"] is None                                  # PSI absent: "if measurable"
    empty = resources.snapshot(read=lambda p: None, run=lambda c: "")   # e.g. run on a Mac: everything unreadable
    assert empty["free_pct"] is None and empty["gpu_free_mb"] is None and empty["swap_used_mb"] is None
    assert not resources.is_safe(empty, good_snap())[0]
    assert not resources.preload_gate(lambda: empty, lambda s: None)[0]


def test_safety_rule_ram_swap_psi_gpu():
    base = good_snap()
    assert resources.is_safe(good_snap(), base)[0]
    assert resources.is_safe(good_snap(psi_some_avg10=None), base)[0]            # pressure unmeasurable is allowed
    assert not resources.is_safe(good_snap(free_pct=None), base)[0]              # fail closed
    assert not resources.is_safe(good_snap(free_pct=10), base)[0]
    assert not resources.is_safe(good_snap(swap_used_mb=None), base)[0]
    assert not resources.is_safe(good_snap(swap_used_mb=10 + C.SWAP_DRIFT_CEIL_MB), base)[0]
    assert not resources.is_safe(good_snap(available_mb=10000 - C.RAM_DRIFT_CEIL_MB), base)[0]
    assert not resources.is_safe(good_snap(psi_some_avg10=C.PSI_SOME_AVG10_MAX + 1), base)[0]
    assert not resources.is_safe(good_snap(gpu_free_mb=None), base)[0]
    assert not resources.is_safe(good_snap(gpu_free_mb=C.GPU_FREE_MB_FLOOR - 1), base)[0]
    assert not resources.is_safe(good_snap(gpu_alloc_mb=1700 + C.GPU_DRIFT_CEIL_MB + 1), base)[0]
    assert not resources.is_safe(good_snap(gpu_alloc_mb=None), base)[0]


def test_preload_gate_three_spaced_readings_and_unchanged_60pct():
    assert C.PRELOAD_FREE_PCT_MIN == 60 and C.PRELOAD_READINGS == 3 and C.PRELOAD_READING_GAP_SEC == 25
    sleeps, logged = [], []
    assert resources.preload_gate(good_snap, sleeps.append, lambda t, s: logged.append(t))[0]
    assert sleeps == [25, 25] and logged == ["preload_reading_1", "preload_reading_2", "preload_reading_3"]
    assert not resources.preload_gate(lambda: good_snap(free_pct=59.9), lambda s: None)[0]
    assert not resources.preload_gate(lambda: good_snap(gpu_free_mb=C.GPU_FREE_MB_MIN - 1), lambda s: None)[0]
    assert not resources.preload_gate(lambda: good_snap(psi_some_avg10=50.0), lambda s: None)[0]
    assert not resources.preload_gate(lambda: good_snap(swap_used_mb=None), lambda s: None)[0]
    seq = iter([good_snap(), good_snap(), good_snap(free_pct=40)])             # a late dip still stops the load
    assert not resources.preload_gate(lambda: next(seq), lambda s: None)[0]


def test_duplicate_job_detection_ignores_self_and_non_python():
    ps = "\n".join(["100 /usr/bin/python3 run_rq4.py --run-inference", "200 python3 run_rq4.py", "300 vim run_rq4.py",
                    "400 /opt/py/python3.13 'Week 3/scripts/run_week3_experiment.py'", "500 python other.py"])
    hits = resources.competing_processes(ps, my_pid=200, my_ppid=1)
    assert len(hits) == 2 and any(h.startswith("100") for h in hits) and any(h.startswith("400") for h in hits)


def test_lock(tmp_path):
    lock = tmp_path / "pid"
    resources.acquire_lock(lock)
    resources.acquire_lock(lock)                       # same pid: fine
    lock.write_text(str(os.getppid()))                 # a different, live process
    if os.getppid() > 1:
        with pytest.raises(RuntimeError):
            resources.acquire_lock(lock)
    lock.write_text("99999999")                        # stale
    resources.acquire_lock(lock)
    resources.release_lock(lock)
    assert not lock.exists()


def test_post_load_baseline_uses_post_gpu_alloc_and_pre_swap():
    pre = good_snap(swap_used_mb=10, gpu_alloc_mb=None)      # pre-load snapshot is taken without torch
    post = good_snap(swap_used_mb=12, gpu_alloc_mb=15000.0)
    assert not resources.is_safe(post, {"swap_used_mb": pre["swap_used_mb"], "gpu_alloc_mb": pre["gpu_alloc_mb"]})[0]  # old behaviour
    base = resources.post_load_baseline(pre, post)
    assert base["gpu_alloc_mb"] == 15000.0 and base["swap_used_mb"] == 10
    assert resources.is_safe(post, base)[0]                                               # healthy load accepted
    assert not resources.is_safe(good_snap(swap_used_mb=10 + C.SWAP_DRIFT_CEIL_MB, gpu_alloc_mb=15000.0), base)[0]
    assert not resources.is_safe(good_snap(gpu_alloc_mb=15000.0 + C.GPU_DRIFT_CEIL_MB + 1), base)[0]
    assert not resources.is_safe(good_snap(gpu_free_mb=1.0, gpu_alloc_mb=15000.0), base)[0]   # VRAM floor kept
    assert not resources.is_safe(good_snap(free_pct=5, gpu_alloc_mb=15000.0), base)[0]        # RAM floor kept
    # swap grew during the load itself -> rejected immediately after load
    assert not resources.is_safe(good_snap(swap_used_mb=10 + C.SWAP_DRIFT_CEIL_MB, gpu_alloc_mb=15000.0),
                                 resources.post_load_baseline(pre, post))[0]
