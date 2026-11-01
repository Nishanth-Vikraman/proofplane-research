# Implementation status

## Delivered

- Strict Pydantic v2 schemas with explicit visible/hidden separation and hidden-key validation.
- SQLite corpus, causal-only primary index export, stable LLM cache keys, controlled-open vocabulary proposals/approval/merge, and compact NumPy embedding serialization.
- Deterministic Python context bundling with fixed-to-buggy diffs, AST symbols, bounded tests/tree, hashes, and hidden-test exclusion.
- Synthetic and BugsInPy adapters. The BugsInPy adapter reads the documented upstream metadata layout and preserves `bugsinpy-checkout -v 0|1` semantics.
- Descriptor, code-change, sequence, metadata, hybrid, random, and hybrid-MMR retrieval. Oracle retrieval is gated from the ordinary path.
- Deterministic offline and OpenAI structured generation providers; explicit direct, brainstorm, random-mechanism, retrieved-context, retrieved-mechanism, MMR, and oracle contracts; arm-specific prompts and causal evidence sanitization; Docker-first execution; and JSONL/CSV/Markdown reports.
- Typer command groups, YAML/environment configuration, spend guard, prompts, bootstrap script, documentation, and leakage tests.
- Three-case offline smoke fixture covering stale derived state, equality boundary behavior, and partial mutation after an exception.

## Verification performed

On 2026-08-07 in this workspace:

```text
python -m pytest                         31 passed
python -m ruff check .                  All checks passed
python -m mypy proofplane_v0            Success: no issues
python -m proofplane_v0.cli --help      Exit 0
python -m proofplane_v0.cli smoke       Exit 0; three synthetic cases reported
python -m proofplane_v0.cli api-smoke   Exit 0; real structured API response
```

Real integration verification:

- Loaded all 501 local BugsInPy metadata records.
- Constructed exact FastAPI 1, YouTube-DL 1, and PySnooper 1 counterfactual workspaces.
- Verified FastAPI 1 visible tests: 11 pass on candidate and reference.
- Verified FastAPI 1 hidden trigger: fails on candidate and passes on reference.
- Validated YouTube-DL 1 as the project-disjoint training case: 79 visible tests pass on both versions and the exact hidden trigger fails only on the candidate.
- Excluded PySnooper 1 because its recorded hidden test requires an undeclared dependency and cannot reproduce in the recorded environment.
- Ran real visible/hidden OpenAI structured extraction for the validated train/evaluation records.
- Built a causal index containing no hidden payload fields.
- Ran `brainstorm` and `retrieved_mechanisms_mmr` at equal K=1 through Docker using only validated records. Both generated tests were valid/runnable but exhibited reference-only failures, honestly scoring zero successful regression-revealing tests.
- Generated the final real report under `data/artifacts/20260807T070305Z-experiment/`.
- Corrected the generation layer after that pilot: prompts now identify base/reference as authoritative, every design-spec arm has a distinct contract, historical mechanism arms receive portable summary plus abstract trigger sequence and test strategy, and each run persists the exact prompt and sanitized generation input for audit.
- Added resumable full-catalog preparation, batch validation/status, validated-case enrichment, `--limit 0` full-eval execution, case-level continuation, unique generated filenames, and rank-correct success@K accounting.
- Replaced the unbounded all-tests context sweep with changed-module/symbol relevance ranking and a hard serialized context ceiling covering diff, selected test snippets, repository tree, and metadata.
- Added bounded generation schemas/output, retry-and-shorten recovery for truncated structured JSON, and pre-call prompt/error artifacts for failed generations.

## Remaining limitations

- BugsInPy contains historical dependencies; additional cases may be excluded when their pinned environment cannot build from current package indexes.
- API token usage is tracked for calls made after the persistent reservation guard was introduced; earlier verification calls in the existing SQLite database show zero tracked tokens.
- The host provides Python 3.13.5; the package declares and targets Python 3.12+.
