# proofplane-v0

`proofplane-v0` is a deterministic Python research prototype for testing whether observable code-change structure can retrieve transferable historical bug mechanisms that improve proactive LLM test generation on held-out real bugs. It is not a debugger or bug localizer: runtime input contains a base repository, candidate change, and pre-existing tests before any failure is known.

## Causal benchmark design

Every corpus record separates a visible `CausalRetrievalDescriptor` from a hidden `HiddenBugPayload`. Only the visible descriptor enters the primary index. Historical portable mechanism summaries are revealed only after causal retrieval; oracle-family retrieval is an explicitly gated diagnostic. Automated leakage tests audit this boundary.

For BugsInPy, the counterfactual reference is production source at `fixed_commit_id` plus verified pre-fix tests; the candidate is production source at `buggy_commit_id` with exactly those same test inputs. The visible change is therefore the reverse of `bug_patch.txt`. Fixed source is the reference because it supplies the intended production behavior while removing the developer regression test introduced with the fix.

## Quick start

Requires Python 3.12+ and Docker for real generated-test execution.

```bash
python -m pip install -e ".[dev]"
python -m proofplane_v0.cli --help
python -m proofplane_v0.cli smoke
python -m pytest
python -m ruff check .
python -m mypy proofplane_v0
```

The offline smoke uses deterministic structured extraction and generation, creates three synthetic cases (stale derived state, equality boundary, partial mutation after exception), and writes JSONL, CSV, and Markdown under `data/artifacts/synthetic-smoke/`. It explicitly uses unsafe local execution only for these trusted fixture strings. All normal execution is Docker-first: no network, one CPU, 512 MB memory, 128 processes, read-only source, tmpfs, timeout, and captured output. `--unsafe-local-execution` is a conspicuous development fallback.

`max_context_chars` is a global serialized target-context ceiling, not merely a diff limit. Nearby tests are ranked by references to changed modules and symbols, capped by `max_nearby_tests`, and share a bounded character allocation. Repository-tree entries are separately capped by `max_repository_tree_entries`; changed-file and changed-symbol metadata is retained unless the global ceiling would be exceeded. Defaults are in `configs/v0.yaml` and `configs/full.yaml`.

Generation output is bounded by `generation_max_output_tokens`; individual rationale, expected-behavior, path, mechanism-list, and test-code fields also have schema limits. If structured output is truncated or invalid, generation retries up to twice with an explicit shorter-response repair instruction. The original prompt, sanitized evidence, retrieval result, and a machine-readable `generation_error.json` are retained even when all retries fail.

## OpenAI setup

Copy `.env.example` values into your environment. Model names are centralized in `configs/v0.yaml` and overridden by `OPENAI_REASONING_MODEL`, `OPENAI_FAST_MODEL`, and `OPENAI_EMBEDDING_MODEL`. Before every uncached call, SQLite reserves `api_call_reserve_usd` against the hard `api_budget_usd` guard; exceeding it requires `--allow-budget-overrun`. `python -m proofplane_v0.cli api-smoke` performs one real structured visible-only extraction.

## KITE-faithful focal-method synthetic-bug feedback

This is a separate strict experiment path. Benchmark construction inspects each offline real patch once to resolve the changed production method, then exports only its file, qualified name, reference location, and reference source hash. The real diff, buggy source, mechanism, and developer test never enter the exported manifest. Profiling, retrieval, injection, generation, feedback, and repair subsequently use that focal identity plus the fixed/reference checkout only. The held-out buggy checkout is opened only by final scoring.

Build the benchmark first. The default reproduces KITE's single-focal-method task. Increase the knob only for an explicit multi-focal ablation:

```powershell
python -m proofplane_v0.cli kite benchmark-build --adapter bugsinpy --root BugsInPy --split eval --max-focal-methods 1 --config configs/full.yaml
# Optional broader benchmark:
python -m proofplane_v0.cli kite benchmark-build --adapter bugsinpy --root BugsInPy --split eval --max-focal-methods 2 --config configs/full.yaml --output data/full/artifacts/kite-benchmark/manifest-max2.json
```

The default manifest is `data/full/artifacts/kite-benchmark/manifest.json`. `kite run` requires it and profiles exactly its focal methods; it no longer searches arbitrary repository functions. Reference-only target profiling produces WHAT/WHERE/HOW summaries, API-grounded related-test examples, and local anchor-relation AST queries. Retrieval embeds WHAT/WHERE/HOW separately and IDF-downweights ubiquitous AST shapes so structural scores do not saturate. Injection plans select a program-issued opaque anchor and must explicitly map historical WHAT/WHERE/HOW evidence to the proposed target edit. Every accepted injection must pass semantic-alignment and duplicate-patch filtering, deterministic locality/anchor checks, full-focal-unit syntax validation, reference and synthetic compilation, module-import probes, and an executable differential pytest probe that passes on reference and fails on the synthetic variant. Failed plans are rejected within the bounded injection budget.

Run the paper-conformance and real Docker gates before an experiment. Both write machine-readable results under `data/full/artifacts/kite-verification/` and make no OpenAI calls:

```powershell
python -m proofplane_v0.cli kite verify --config configs/full.yaml
python scripts/verify_kite_docker_gate.py --config configs/full.yaml --root BugsInPy
python -m proofplane_v0.cli kite verify --config configs/full.yaml
```

The reviewed-knowledge experiment compares four arms at K=3. The repaired direct arm is
the primary matched control for synthetic feedback: the pipeline constructs one reference-filtered,
bounded-repair suite, persists its content hash, and passes that exact suite byte-for-byte to both
the control and SBF arms before SBF enhancement:

| Arm | Target context | Historical mechanism | Executable synthetic feedback |
|---|---|---|---|
| `direct_reference` | Same reference-only target unit | No | No |
| `direct_reference_repaired` | Same selected target unit, correct-side filtered/repaired | No | No |
| `retrieved_mechanism_text` | Same reference-only target unit | Yes | No |
| `retrieved_mechanism_sbf` | Same reference-only target unit | Yes | Yes |

Build the historical mechanism/AST cache without any new enrichment API calls, then run one eligible focal case synchronously as a smoke test:

```powershell
python -m proofplane_v0.cli kite knowledge-build --config configs/full.yaml
python -m proofplane_v0.cli kite run --config configs/full.yaml --adapter bugsinpy --root BugsInPy --limit 1 --k 1 --run-id kite-smoke-1
```

After the smoke succeeds, launch up to 15 eligible focal cases at K=3 independently of the invoking shell:

```powershell
$runId = "$(Get-Date -Format yyyyMMddTHHmmssZ)-kite-lite"
powershell -ExecutionPolicy Bypass -File scripts/launch_kite_lite_detached.ps1 -RunId $runId -Limit 15 -K 3 -ReviewKnowledge
python -m proofplane_v0.cli kite status $runId --config configs/full.yaml
```

The launcher creates independent worker and heartbeat processes using `Win32_Process.Create`. It verifies the Docker Linux engine and local base image before every attempt, retries the same content-cached run up to three times, and writes:

- pipeline stdout/stderr and worker/monitor state under `data/full/artifacts/pipelines/`;
- per-case reference profiles, retrieval scores, injection plans/gates, generated tests, synthetic feedback, and final status-only scores under `data/full/artifacts/<run-id>/`;
- the exact sanitized `benchmark-manifest.json`, plus `heartbeat.json`, `results.jsonl`, `summary.csv`, and `REPORT.md` in the run directory.

`summary.csv` and `REPORT.md` distinguish cases with an accepted injection, unique injected patches, synthetic kills, enhancement-round calls, and promoted tests. If all injections for a case are rejected, the SBF arm still retains and evaluates its reference-passing base suite; this prevents injection-yield failures from being misreported as zero generated tests.

KITE API accounting has its own `kite_api_budget_usd` guard. Successful KITE calls replace their
reservation with a token-based estimate; failed calls retain their reservation conservatively.
`configs/full.yaml` currently sets a $25 cumulative KITE ceiling because prior cached experiments
already account for roughly $10.30 of conservative reservations. Embeddings and every
structured-output stage are content-cached.

The reviewed knowledge build follows the paper's offline separation: each historical TRAIN bug
unit is abstracted into WHAT/WHERE/HOW using its local correct/buggy focal evidence; real embeddings
then form deterministic complete-link semantic groups; a second structured call reviews each group
and synthesizes accepted mechanisms with expected buggy behavior, representative evidence, and
supporting bug-unit provenance. Eval targets never enter this stage. Every call and embedding is
content-cached, so retries and later runs reuse completed work. The paper does not publish its exact
grouping algorithm or threshold; this implementation records its explicit complete-link threshold
and group-size bound in configuration and manifests.

```powershell
python -m proofplane_v0.cli kite knowledge-build --explicit-llm --config configs/full.yaml
```

`random_mechanism_sbf` remains an optional control arm and is not included in the four-arm reviewed run.

## BugsInPy

```powershell
python -m proofplane_v0.cli dataset inspect --adapter bugsinpy --root BugsInPy --limit 25
python -m proofplane_v0.cli split create --adapter bugsinpy --root BugsInPy

# One verified train/evaluation pair. Case IDs are literal—do not type angle brackets.
python -m proofplane_v0.cli dataset prepare --adapter bugsinpy --root BugsInPy --case-id bugsinpy:youtube-dl:1
python -m proofplane_v0.cli dataset prepare --adapter bugsinpy --root BugsInPy --case-id bugsinpy:fastapi:1
python -m proofplane_v0.cli dataset validate bugsinpy:youtube-dl:1 --adapter bugsinpy --root BugsInPy
python -m proofplane_v0.cli dataset validate bugsinpy:fastapi:1 --adapter bugsinpy --root BugsInPy
python -m proofplane_v0.cli corpus enrich --adapter bugsinpy --root BugsInPy --split train --limit 1 --case-id bugsinpy:youtube-dl:1
python -m proofplane_v0.cli corpus enrich --adapter bugsinpy --root BugsInPy --split eval --limit 1 --case-id bugsinpy:fastapi:1
python -m proofplane_v0.cli index build
python -m proofplane_v0.cli experiment run --adapter bugsinpy --root BugsInPy --limit 1 --k 3 --arm direct --arm brainstorm --arm random_mechanisms --arm retrieved_raw_context --arm retrieved_mechanisms --arm retrieved_mechanisms_mmr
```

The adapter reads `projects/<project>/project.info`, `projects/<project>/bugs/<bug-id>/bug.info`, recorded commits, test paths, requirements, setup, and test selector. It clones the exact commits directly, stores the fixed triggering tests under `oracle_tests`, and restores buggy/pre-fix test content into both runtime workspaces. `dataset validate` builds isolated historical images, verifies visible tests pass on both versions, and verifies the recorded trigger fails only on the candidate. Invalid cases receive a persisted exclusion reason. The upstream shell helpers remain available under `BugsInPy/framework/bin`, but are not required by this Windows workflow.

To prepare several selected cases, repeat `--case-id`; `dataset prepare` refuses to clone all 501 cases accidentally. `--metadata-only` registers the catalog without cloning. A larger experiment follows the same sequence: prepare and validate project-disjoint train/eval cases, enrich them, then increase `--limit` and `--k`.

### Recommended staged experiment

Use this flow before paying for enrichment and generation across the complete valid dataset. It broadly prepares and validates the catalog, enriches a 100-case training corpus and 15 evaluation targets, then runs all six non-oracle arms at K=3. Set a deliberate `api_budget_usd` in `configs/full.yaml` before the API-backed steps.

Run from the repository root in PowerShell:

```powershell
# 1. Register the catalog, freeze project-isolated splits, prepare broadly, and validate in Docker.
python -m proofplane_v0.cli dataset prepare --adapter bugsinpy --root BugsInPy --metadata-only --config configs/full.yaml
python -m proofplane_v0.cli split create --adapter bugsinpy --root BugsInPy --config configs/full.yaml
python -m proofplane_v0.cli dataset prepare --adapter bugsinpy --root BugsInPy --all --continue-on-error --config configs/full.yaml
# Validate the pilot-relevant splits unattended. Keep at least 30 GB free for Docker builds.
python -m proofplane_v0.cli dataset validate-all --split train --adapter bugsinpy --root BugsInPy --resume --config configs/full.yaml
python -m proofplane_v0.cli dataset validate-all --split eval --adapter bugsinpy --root BugsInPy --resume --config configs/full.yaml
python -m proofplane_v0.cli dataset status --adapter bugsinpy --config configs/full.yaml

# 2. Pilot index: enrich up to 100 valid train cases and 15 valid eval cases.
python -m proofplane_v0.cli corpus enrich --split train --limit 100 --adapter bugsinpy --root BugsInPy --config configs/full.yaml
python -m proofplane_v0.cli corpus enrich --split eval --limit 15 --adapter bugsinpy --root BugsInPy --config configs/full.yaml
python -m proofplane_v0.cli index build --config configs/full.yaml

# 3. Pilot evaluation: all six non-oracle arms, K=3, on up to 15 enriched eval cases.
python -m proofplane_v0.cli experiment run --adapter bugsinpy --root BugsInPy --limit 15 --k 3 --continue-on-error --config configs/full.yaml
```

Inspect each arm's `retrieval.json`, `generation_input.json`, `generated.json`, and execution results under `data/full/artifacts/`. Check that retrieval uses several plausible, target-independent historical mechanisms rather than repeatedly selecting one weak analogy.

If the pilot is promising, expand the same database and index to every remaining valid training and evaluation case, then evaluate the complete enriched eval split:

```powershell
python -m proofplane_v0.cli corpus enrich-all --split train --adapter bugsinpy --root BugsInPy --config configs/full.yaml
python -m proofplane_v0.cli corpus enrich-all --split eval --adapter bugsinpy --root BugsInPy --config configs/full.yaml
python -m proofplane_v0.cli index build --config configs/full.yaml
python -m proofplane_v0.cli experiment run --adapter bugsinpy --root BugsInPy --limit 0 --k 3 --continue-on-error --config configs/full.yaml
```

`--limit 0` means every enriched eval case; 61 is the split's theoretical catalog size, while the actual run count depends on preparation, validation, and enrichment success. The commands are resumable through persisted workspaces, validation artifacts, API cache/ledger records, and enriched corpus records.

### Full 501-case workflow

Yes, the CLI can now attempt the complete BugsInPy catalog. This means cataloguing and attempting to prepare/validate all 501 cases. It does **not** mean all 501 will become experiment observations: historically unreproducible cases are persisted as exclusions, train cases form the retrieval corpus, dev cases are reserved for tuning, and generation is evaluated only on the project-isolated eval split.

The full workflow is intentionally resumable at the expensive preparation, validation, and enrichment stages. It also uses `configs/full.yaml`, which keeps the full corpus and API ledger separate from the pilot data.

Before starting, open `configs/full.yaml` and choose an explicit `api_budget_usd`. Its committed `$5` value is deliberately too low for an unattended full run. With the current `$0.25` conservative reservation per uncached API call, the theoretical ceiling is roughly `$305` if every one of the 332 train and 61 eval cases validates and all six non-oracle arms run at least once. Actual eligibility should be lower, but storage, Docker build time, and API use can still be substantial. Do not use `--allow-budget-overrun` as a substitute for setting a deliberate ceiling.

Run these commands from the repository root in PowerShell:

```powershell
# 1. Register all metadata and create project-isolated train/dev/eval splits.
python -m proofplane_v0.cli dataset prepare --adapter bugsinpy --root BugsInPy --metadata-only --config configs/full.yaml
python -m proofplane_v0.cli split create --adapter bugsinpy --root BugsInPy --config configs/full.yaml

# 2. Materialize every case. Failures are persisted and do not stop the batch.
python -m proofplane_v0.cli dataset prepare --adapter bugsinpy --root BugsInPy --all --continue-on-error --config configs/full.yaml

# 3. Validate the train/eval splits unattended. Per-case images are removed automatically.
python -m proofplane_v0.cli dataset validate-all --split train --adapter bugsinpy --root BugsInPy --resume --config configs/full.yaml
python -m proofplane_v0.cli dataset validate-all --split eval --adapter bugsinpy --root BugsInPy --resume --config configs/full.yaml
python -m proofplane_v0.cli dataset status --adapter bugsinpy --config configs/full.yaml

# 4. Enrich every validated train and eval case. Each command continues past case-level failures.
python -m proofplane_v0.cli corpus enrich-all --split train --adapter bugsinpy --root BugsInPy --config configs/full.yaml
python -m proofplane_v0.cli corpus enrich-all --split eval --adapter bugsinpy --root BugsInPy --config configs/full.yaml

# Optional: enrich development cases when tuning retrieval/prompt settings.
# python -m proofplane_v0.cli corpus enrich-all --split dev --adapter bugsinpy --root BugsInPy --config configs/full.yaml

# 5. Build the causal index from eligible records.
python -m proofplane_v0.cli index build --config configs/full.yaml

# 6. Run K=3 on every enriched eval case across all six non-oracle arms.
# Omitting --arm intentionally selects direct, brainstorm, random mechanisms,
# retrieved raw context, retrieved mechanisms, and retrieved mechanisms + MMR.
python -m proofplane_v0.cli experiment run --adapter bugsinpy --root BugsInPy --limit 0 --k 3 --continue-on-error --config configs/full.yaml
```

Preparation skips already materialized workspaces, validation reuses its per-case JSON artifacts with `--resume`, and `corpus enrich-all` skips already enriched records. If the spend guard stops enrichment, raise the explicit budget only if desired and rerun the same command. Batch summaries are written under `data/full/artifacts/`.

This workflow can consume substantial disk because each historical case has candidate/reference workspaces and Docker build cache. Per-case validation images are removed after use. Keep at least 30 GB free before unattended validation; Docker/WSL may become read-only or unstable when its host drive is exhausted.

Inspect an individual result or rebuild a report:

```powershell
python -m proofplane_v0.cli run case bugsinpy:fastapi:1 --adapter bugsinpy --root BugsInPy --arm retrieved_mechanisms_mmr --k 1
python -m proofplane_v0.cli report build 20260807T070305Z-experiment
```

Replace the report ID with the directory name printed by your own experiment command. A successful regression-revealing test must fail on the candidate and pass on the reference. Pass/pass, fail/fail, reference-only failure, syntax failure, setup/import failure, and timeout remain separate outcomes.

For a larger run, increase `case_limit`, `generation_k`, and the API budget in a copied YAML config. Each per-arm artifact directory contains `prompt.txt` (the exact LLM input), `generation_input.json` (the sanitized historical evidence supplied to that arm), `retrieval.json`, `generated.json`, and execution results. The combined experiment directory contains `results.jsonl`, `summary.csv`, and `REPORT.md`.

## Arms and analysis

Each arm now has an explicit retrieval and prompt contract:

| CLI arm | Historical selection | Information supplied to generator | Generation behavior |
|---|---|---|---|
| `direct` | none | none | Generate directly from visible target context. |
| `brainstorm` | none | none | Infer diverse visible-context hypotheses, then test them. |
| `random_mechanisms` | seeded random | Portable summary, abstract trigger sequence, and test strategy. | Random-mechanism control. |
| `retrieved_raw_context` | causal hybrid | Sanitized visible change summary only. | Tests whether retrieval context alone helps. |
| `retrieved_mechanisms` | causal hybrid | Portable summary, abstract trigger sequence, and test strategy. | Tests mechanism transfer after causal retrieval. |
| `retrieved_mechanisms_mmr` | causal hybrid + MMR | A diverse portfolio of the same portable mechanism fields. | Primary retrieval treatment. |
| `oracle` | hidden-family similarity | Portable fields from matching historical cases; no target hidden payload. | Explicit upper-bound diagnostic, excluded from primary conclusions. |

Running `experiment run` without any `--arm` flags executes all six non-oracle arms. `--limit 0` means every enriched eval case. Oracle must be selected explicitly with `--arm oracle`. All prompts state that base/reference is authoritative, candidate is suspected regressed, and a useful test passes on reference while failing on candidate.

Primary comparison: retrieved-mechanisms-MMR vs brainstorm. Secondary: retrieved mechanisms vs random mechanisms. Buckets are seen-family, compositional, novel-OOD, and unclassified; hidden labels are used only after splitting for analysis. Reports do not claim significance automatically and should add bootstrap intervals only when sample size permits.

## Known limitations

- Reverse-fix PRs may not resemble organic bug-introducing commits.
- Public bug datasets may overrepresent local bugs.
- Retrieval may add only prompt diversity.
- Generated-test infrastructure may dominate outcomes.
- Benchmark success may not transfer to private repositories.
- Old lockfiles can fail on current package indexes or unavailable historical base images; these are recorded as environment exclusions rather than benchmark failures.
