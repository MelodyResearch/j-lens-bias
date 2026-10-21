import subprocess
import sys

import numpy as np
import pytest

from rq4 import analysis as A
from rq4 import config as C
from rq4 import figures
from synth import synth_rows, write_jsonl


def test_bootstrap_is_deterministic_and_brackets_mean():
    v = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    a, b = A.bootstrap_mean_ci(v), A.bootstrap_mean_ci(v)
    assert a == b and a["mean"] == 5.5 and a["lo"] < 5.5 < a["hi"] and a["n"] == 10
    w = np.sqrt(np.arange(1.0, 11.0))
    assert A.bootstrap_mean_ci(w, seed=1) != A.bootstrap_mean_ci(w) and A.bootstrap_mean_ci(w) == A.bootstrap_mean_ci(w)
    one = A.bootstrap_mean_ci(np.array([3.0]))
    assert one["lo"] == one["hi"] == 3.0
    assert A.bootstrap_mean_ci(np.array([]))["n"] == 0
    c = A.bootstrap_mean_ci(np.full(10, 2.0))
    assert c["lo"] == c["hi"] == 2.0


def test_primary_estimand_recovers_known_effects(tmp_path):
    p = tmp_path / "r.jsonl"
    write_jsonl(p, synth_rows())
    df = A.load_results(p)
    pairs = A.per_pair_effects(df)
    assert len(pairs) == 40 * 3 and pairs.diff_excess_abs.notna().all()
    # random effects per seed: rand_effect*(1+s%3) -> mean over seeds of |d| for seeds 11,23,47 = 0.1*(1+(11%3)... computed directly
    rnd_mean = np.mean([0.1 * (1 + int(a[-2:]) % 3) for a in C.ARM_RANDOM])
    assert pairs.mc_excess_abs.iloc[0] == pytest.approx(2.0 - rnd_mean)
    assert pairs.oe_excess_abs.iloc[0] == pytest.approx(0.5 - rnd_mean)
    assert (pairs.diff_excess_abs - 1.5).abs().max() < 1e-9
    items = A.per_item(pairs)
    assert len(items) == 30 and (items.n_conditions == 4).all()
    s = A.summarize(items)
    d = s[(s.metric == "diff_excess_abs") & (s.layer == 8)].iloc[0]
    assert d["mean"] == pytest.approx(1.5) and d.n == 10
    # behaviour: MC target flips category (excess 1 - 0), open does not -> MC - open = 1
    assert (pairs.diff_excess_changed == 1.0).all() and (pairs.mc_excess_changed == 1.0).all() and (pairs.oe_excess_changed == 0.0).all()
    assert pairs.diff_excess_abs_sham.notna().all()
    sp = A.seed_spread(pairs)
    assert set(sp.format) == {"mc", "oe"} and sp.seed_sd.notna().all()


def test_negative_result_is_kept_not_dropped(tmp_path):
    p = tmp_path / "r.jsonl"
    write_jsonl(p, synth_rows(mc_effect=0.0, oe_effect=0.0, rand_effect=0.3, flip_target=False))
    items = A.per_item(A.per_pair_effects(A.load_results(p)))
    s = A.summarize(items)
    row = s[(s.metric == "diff_excess_abs") & (s.layer == 16)].iloc[0]
    assert row["n"] == 10 and abs(row["mean"]) < 1e-9          # a null/negative effect is reported, with n


def test_missing_arm_excludes_pair_from_matched_comparison_only(tmp_path):
    drop = {("age_01_qi1__ambig__neg__open_ended", "random_s23", 8)}
    p = tmp_path / "r.jsonl"
    from synth import synth_rows as sr
    rows = sr(drop=drop)
    write_jsonl(p, rows)
    pairs = A.per_pair_effects(A.load_results(p))
    bad = pairs[(pairs.pair_id == "age_01_qi1__ambig__neg") & (pairs.layer == 8)].iloc[0]
    assert np.isnan(bad.get("diff_excess_abs", np.nan)) or "diff_excess_abs" not in bad
    good = pairs[(pairs.pair_id == "age_01_qi1__ambig__neg") & (pairs.layer == 16)].iloc[0]
    assert good.diff_excess_abs == pytest.approx(1.5, abs=0.5)


def test_duplicate_keys_rejected(tmp_path):
    rows = synth_rows(n_items=1)
    p = tmp_path / "r.jsonl"
    write_jsonl(p, rows + rows[:1])
    with pytest.raises(ValueError, match="duplicate"):
        A.load_results(p)


def test_completion_status_labels(tmp_path):
    p = tmp_path / "r.jsonl"
    write_jsonl(p, synth_rows(n_items=1))
    assert A.completion_status(A.load_results(p))["status"] == "PILOT_ONLY"
    write_jsonl(p, synth_rows())
    full = A.completion_status(A.load_results(p))
    assert full["status"] == "COMPLETE_GATE_MET", full
    write_jsonl(p, synth_rows(drop={("age_02_qi2__ambig__neg__open_ended", "target", 8)}))
    assert A.completion_status(A.load_results(p))["status"] == "INCONCLUSIVE"
    assert A.completion_status(A.load_results(p), machine_safe=False)["status"] == "BLOCKED"


def test_figures_render_from_saved_results_only(tmp_path):
    p = tmp_path / "r.jsonl"
    write_jsonl(p, synth_rows())
    paths = figures.make_figures(p, tmp_path / "figs")
    assert len(paths) == 3 and all(x.exists() and x.stat().st_size > 5000 for x in paths)
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError):
        figures.make_figures(empty, tmp_path / "f2")


def _run_cli(args, tmp):
    code = ("import sys; sys.argv=['run_rq4.py']+%r; sys.path.insert(0, %r); import run_rq4; run_rq4.main(); "
            "bad=[m for m in ('torch','transformers','jlens') if m in sys.modules]; print('HEAVY:', bad)") % (args, str(C.ROOT))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp, timeout=120)


@pytest.mark.parametrize("args", [[], ["--prepare-only"], ["--i-confirm-no-heavy-experiment"]])
def test_default_and_partial_flags_stay_prepare_only_without_heavy_imports(args, tmp_path):
    r = _run_cli(args, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "PREPARE-ONLY" in r.stdout and "HEAVY: []" in r.stdout
    assert "1280 planned evaluations" in r.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific refusal path")
@pytest.mark.parametrize("args", [["--run-inference"], ["--run-inference", "--i-confirm-no-heavy-experiment"],
                                  ["--run-inference", "--i-confirm-no-heavy-experiment", "--stage", "2"]])
def test_run_inference_refused_on_macos_before_any_heavy_import(args, tmp_path):
    r = _run_cli(args, tmp_path)
    assert r.returncode != 0 and "remote-only" in (r.stdout + r.stderr)
    assert "HEAVY" not in r.stdout and "PREPARE-ONLY" not in r.stdout
