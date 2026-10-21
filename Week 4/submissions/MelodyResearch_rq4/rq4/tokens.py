"""Target-phrase tokenization: complete phrase, leading space, meaningful mask, deduplicated."""

from __future__ import annotations

from dataclasses import dataclass


class IneligiblePair(Exception):
    """Raised when a target pair cannot define a contrast direction (pre-outcome exclusion)."""


def phrase_token_ids(tokenizer, text: str, mask) -> list[int]:
    """Token ids of ' ' + text as a continuation, kept only if `mask[id]` is True, first-seen order,
    duplicates removed. `mask` is the repository's meaningful-token mask (jlens.vis)."""
    ids = tokenizer(" " + text, add_special_tokens=False).input_ids
    seen: set[int] = set()
    out: list[int] = []
    for i in ids:
        i = int(i)
        if i in seen or not bool(mask[i]):
            continue
        seen.add(i)
        out.append(i)
    return out


@dataclass(frozen=True)
class TargetIds:
    stereotype: tuple[int, ...]
    counter: tuple[int, ...]
    raw_stereotype: tuple[int, ...]   # before mask/dedup, for the audit log
    raw_counter: tuple[int, ...]
    overlap: tuple[int, ...]

    def as_log(self) -> dict:
        return {"stereotype_ids": list(self.stereotype), "counter_ids": list(self.counter),
                "raw_stereotype_ids": list(self.raw_stereotype), "raw_counter_ids": list(self.raw_counter),
                "overlap_ids": list(self.overlap)}


def resolve_target_ids(tokenizer, stereotype_text: str, counter_text: str, mask) -> TargetIds:
    """Both phrases tokenized in full. Ineligible if either has no meaningful token or the
    two meaningful token SETS are identical."""
    raw_s = tuple(int(i) for i in tokenizer(" " + stereotype_text, add_special_tokens=False).input_ids)
    raw_c = tuple(int(i) for i in tokenizer(" " + counter_text, add_special_tokens=False).input_ids)
    s = tuple(phrase_token_ids(tokenizer, stereotype_text, mask))
    c = tuple(phrase_token_ids(tokenizer, counter_text, mask))
    if not s:
        raise IneligiblePair(f"no meaningful token in stereotype phrase {stereotype_text!r}")
    if not c:
        raise IneligiblePair(f"no meaningful token in counter phrase {counter_text!r}")
    if set(s) == set(c):
        raise IneligiblePair(f"identical token sets for {stereotype_text!r} and {counter_text!r}")
    return TargetIds(s, c, raw_s, raw_c, tuple(sorted(set(s) & set(c))))
