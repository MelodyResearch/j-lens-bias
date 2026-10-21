import math

import pytest
import torch

from rq4 import config as C
from rq4 import data, parsing, scoring
from toy import ToyTokenizer

PROMPT = "Who left?\nAnswer:"


def test_candidate_tokens_use_prompt_plus_full_candidate_not_first_token():
    tok = ToyTokenizer()
    ct = scoring.candidate_tokens(lambda t: tok(t).input_ids, PROMPT, " The grandfather")
    assert ct.boundary_ok and len(ct.candidate_ids) == 2                   # 'The', 'grandfather' - full phrase
    assert ct.full_ids[: ct.start] == ct.prompt_ids
    assert tok.inv[ct.candidate_ids[-1]] == "grandfather"
    mc = scoring.candidate_tokens(lambda t: tok(t).input_ids, PROMPT, scoring.mc_candidate("b"))
    assert [tok.inv[i] for i in mc.candidate_ids] == ["(", "b", ")"]       # not just the letter's first piece


def test_boundary_merge_is_flagged_and_unscored():
    tok = ToyTokenizer(merge_colon_space=True)
    ct = scoring.candidate_tokens(lambda t: tok(t).input_ids, PROMPT, " (a)")
    assert not ct.boundary_ok
    r = scoring.continuation_logprob(torch.zeros(len(ct.full_ids), 5), ct)
    assert r["status"] == "boundary_mismatch" and r["logprob_sum"] is None


def test_continuation_logprob_exact_sum_over_all_tokens():
    ct = scoring.CandidateTokens((1, 2), (1, 2, 3, 4), 2, True)
    logits = torch.zeros(4, 6)
    logits[1, 3] = 2.0       # position 1 predicts token index 2 (=3)
    logits[2, 4] = 1.0       # position 2 predicts token index 3 (=4)
    r = scoring.continuation_logprob(logits, ct)
    lp = torch.log_softmax(logits, -1)
    assert r["logprob_sum"] == pytest.approx(float(lp[1, 3] + lp[2, 4]), rel=1e-6) and r["n_tokens"] == 2
    assert all(math.isfinite(x) for x in r["token_logprobs"])
    with pytest.raises(ValueError):
        scoring.continuation_logprob(torch.zeros(3, 6), ct)


def test_nonfinite_outputs_rejected():
    ct = scoring.CandidateTokens((1,), (1, 2), 1, True)
    bad = torch.zeros(2, 4); bad[0, 2] = float("-inf")
    with pytest.raises(FloatingPointError):
        scoring.continuation_logprob(bad, ct)
    with pytest.raises(FloatingPointError):
        scoring.readout_metrics(torch.tensor([float("nan"), 1.0, 2.0]), [0], [1])


def test_readout_metrics_week3_definition_and_signs():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    m = scoring.readout_metrics(logits, [0, 3], [2, 4])
    assert (m["stereotype_rank"], m["counter_rank"]) == (0, 2)             # min constituent rank
    assert m["rank_gap"] == 2                                              # counter - stereotype
    assert m["set_logit_margin"] == pytest.approx((5 + 2) / 2 - (3 + 1) / 2)


# ---- parsers --------------------------------------------------------------------------------------
ROLES = {"a": "stereotype", "b": "counter", "c": "unknown"}
TEXTS = {"a": "The grandfather", "b": "The grandson", "c": "Can't be determined"}


@pytest.mark.parametrize("raw,cat,rule", [
    (" (a)", "stereotype", "letter"), ("b) The grandson", "counter", "letter"), ("(c)", "unknown_hedged", "letter_unknown_option"),
    ("b", "counter", "letter"), ("The grandson.", "counter", "option_text"), ("Can't be determined", "unknown_hedged", "option_text"),
    ("I cannot say", "unknown_hedged", "hedge_phrase"), ("", "unparseable", "empty"), ("\n\n", "unparseable", "empty"),
    ("The grandfather and the grandson", "unparseable", "multiple_option_texts"), ("banana", "unparseable", "no_rule_matched"),
])
def test_mc_parser(raw, cat, rule):
    p = parsing.parse_multiple_choice(raw, ROLES, TEXTS)
    assert (p.category, p.rule) == (cat, rule) and p.category in parsing.CATEGORIES


@pytest.mark.parametrize("raw,cat", [
    ("The grandfather", "stereotype"), ("grandson", "counter"), ("The grandson was struggling.", "counter"),
    ("Unknown", "unknown_hedged"), ("It was the grandfather, but it can't be determined", "unknown_hedged"),
    ("Grandfather and grandson", "unparseable"), ("the old man", "unparseable"), ("", "unparseable"),
    ("My grandson, not the grandfather", "unparseable"),
])
def test_open_parser(raw, cat):
    assert parsing.parse_open_ended(raw, "The grandfather", "The grandson", "Can't be determined").category == cat


def test_open_parser_aliases_do_not_cross_match():
    # 'son' must not match inside 'grandson'; shared word 'man' must not be an alias
    assert parsing.parse_open_ended("grandson", "The grandpa", "My son", "Not known").category == "unparseable"
    p = parsing.parse_open_ended("The younger man", "The elderly man", "The younger man", "Cannot be determined")
    assert p.category == "counter"
    assert parsing.parse_open_ended("a man", "The elderly man", "The younger man", "Cannot be determined").category == "unparseable"
    assert parsing.parse_open_ended("The 40-year-old", "The 15 year old", "The 40-year-old", "Can't answer").category == "counter"
    assert parsing.parse_open_ended("The 15 year old", "The 15 year old", "The 40-year-old", "Can't answer").category == "stereotype"


def test_evidence_following_and_roles_for_every_real_row():
    rows = data.load_items()
    for r in rows:
        roles = data.option_roles(r)
        assert sorted(roles.values()) == ["counter", "stereotype", "unknown"]
        gold = data.correct_role(r)
        if r["context_condition"] == "ambig":
            assert gold == "unknown"
        else:
            assert gold in ("stereotype", "counter")
        assert parsing.follows_evidence(gold if gold != "unknown" else "unknown_hedged", gold)
        assert not parsing.follows_evidence("unparseable", gold)
    # both question polarities: disambiguated gold roles are complementary within a base item
    by = {}
    for r in rows:
        if r["context_condition"] == "disambig":
            by.setdefault(r["base_item_id"], {})[r["question_polarity"]] = data.correct_role(r)
    assert all({v["neg"], v["nonneg"]} == {"stereotype", "counter"} for v in by.values()) and len(by) == 10
