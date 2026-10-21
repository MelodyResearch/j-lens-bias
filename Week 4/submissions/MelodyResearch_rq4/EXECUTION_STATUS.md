# Execution status — 2026-10-20 (updated: remote-only execution)

**Zero real evaluation rows exist.** No generation, no intervention outcome, no Stage 1 approval; Stage 2, outcome analysis, results push/PR and the Google Doc update have not occurred. Status remains `BLOCKED` for local execution: the Mac stopped correctly (48 % free memory < 60 %) and inference is now **refused on macOS**. Execution moved to a remote Linux CUDA GPU (Google Colab) via `notebooks/rq4_colab.ipynb`, with persistent Drive checkpoints, a fail-closed Linux/GPU telemetry gate, a prespecified outcome-blind Stage 1 validator, and a review gate before Stage 2 (details: `REMOTE_EXECUTION_ADDENDUM.md`). The remote path is synthetic-tested only (167 tests) and has never run on a GPU; the notebook pins the audited code commit `28fdda141ce139018e9d5a290fa784378b760aa3`. The local-run resume instructions in the Stage 0 text below are superseded.

---

## Stage 0 record (local)

Stage 0 is complete. The independent Python 3.12 run passed all 97 tests, and the default runner prepared 80 prompts, 40 matched pairs, and 1,280 planned evaluation keys without loading a model. The tokenization-only audit of the cached Qwen tokenizer found zero Week 3 source-position mismatches, zero prompt/candidate boundary failures across all three answer roles and 80 prompts, and zero ineligible target pairs among 40 pairs. This audit did not run the model.

The exact public artifacts were cached locally during the earlier preparation; they are preserved but will not be used for inference. Colab downloads its own copies remotely. The model repository resolves to commit `2fc06364715b967f1860aea9cf38778875588b17`, with model-weight SHA-256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`. The released lens revision `qwen-n1000` resolves to commit `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`, with lens-file SHA-256 `aa26b68ed73cf903280dbd8d1806f4ed8580aad205f396a5c997ee19259c9b48`. The local runtime has Python 3.12.14, torch 2.14.1, transformers 5.20.2, and `jlens` at source commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`.

Stage 1 was attempted through the frozen resource gate and stopped **before model load**: its first reading was 48% free memory, below the protocol's 60% pre-load minimum. The later read-only check showed 47% free memory, 15,912 MB of swap in use, and pressure level 2. There are zero real evaluation rows, no generation, no intervention outcome, and no Stage 1 approval. The status is `BLOCKED` by local machine safety. Stage 2, the statistical outcome analysis, GitHub push/PR, and Google Doc RQ4 update have not occurred.

Local resume instructions removed: inference is remote-only (see the addendum). The pilot's raw generations, `runtime_preflight.json`, `rq4_memory_log.jsonl` and the validator report must be inspected before recording Stage 1 approval. Do not interpret or publish a pilot as the full ten-item result.
