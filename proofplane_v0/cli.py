from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from proofplane_v0.config import Settings
from proofplane_v0.context.bundle import ContextBundle
from proofplane_v0.corpus.enrich import OpenAIEnricher
from proofplane_v0.corpus.store import CorpusStore
from proofplane_v0.corpus.vocabulary import VocabularyRegistry
from proofplane_v0.datasets.bugsinpy import BugsInPyAdapter
from proofplane_v0.datasets.synthetic import SyntheticAdapter
from proofplane_v0.evaluation.docker_runner import (
    EnvironmentBuildError,
    ValidationInfrastructureError,
)
from proofplane_v0.evaluation.workspace import VALIDATOR_VERSION, validate_counterfactual
from proofplane_v0.experiments.runner import (
    execute_case,
    run_experiment,
)
from proofplane_v0.experiments.runner import (
    smoke as run_smoke,
)
from proofplane_v0.kite.api import OpenAIService
from proofplane_v0.kite.benchmark import build_focal_benchmark
from proofplane_v0.kite.conformance import verify_conformance, write_conformance_report
from proofplane_v0.kite.knowledge import (
    build_mechanisms as build_kite_mechanisms,
)
from proofplane_v0.kite.knowledge import (
    build_reviewed_mechanisms,
    knowledge_manifest,
)
from proofplane_v0.kite.runner import run_kite_experiment
from proofplane_v0.workflow import (
    adapter_for,
    create_splits,
    enrich_split,
    prepare_cases,
    register_dataset,
)
from proofplane_v0.workflow import (
    build_index as build_primary_index,
)

app = typer.Typer(help="proofplane-v0 causal retrieval research prototype")
dataset_app = typer.Typer()
vocab_app = typer.Typer()
split_app = typer.Typer()
corpus_app = typer.Typer()
index_app = typer.Typer()
run_app = typer.Typer()
experiment_app = typer.Typer()
report_app = typer.Typer()
kite_app = typer.Typer(help="KITE-lite reference-only synthetic-bug-feedback experiment")
for name, sub in (
    ("dataset", dataset_app),
    ("vocab", vocab_app),
    ("split", split_app),
    ("corpus", corpus_app),
    ("index", index_app),
    ("run", run_app),
    ("experiment", experiment_app),
    ("report", report_app),
    ("kite", kite_app),
):
    app.add_typer(sub, name=name)
console = Console()


@kite_app.command("benchmark-build")
def kite_benchmark_build(
    config: Path = Path("configs/full.yaml"),
    adapter: str = "bugsinpy",
    root: Path = Path("BugsInPy"),
    split: str = "eval",
    max_focal_methods: int | None = typer.Option(
        None,
        "--max-focal-methods",
        min=1,
        help="Keep patches resolving to at most this many reference-side production methods.",
    ),
    limit: int = 0,
    output: Path | None = typer.Option(None, "--output"),  # noqa: B008
) -> None:
    """Build sanitized focal-method benchmark metadata from the offline real patch."""
    configured = Settings.load(config)
    target = output or configured.artifacts_dir / "kite-benchmark" / "manifest.json"
    manifest = build_focal_benchmark(
        adapter=adapter_for(adapter, root),
        store=CorpusStore(configured.database),
        data_dir=configured.data_dir,
        split=split,
        max_focal_methods=max_focal_methods or configured.kite_max_focal_methods,
        output=target,
        limit=limit,
    )
    eligible = sum(value.eligible for value in manifest.cases)
    console.print(
        f"Built KITE focal benchmark: {eligible}/{len(manifest.cases)} eligible; {target}"
    )


@kite_app.command("knowledge-build")
def kite_knowledge_build(
    config: Path = Path("configs/full.yaml"),
    explicit_llm: bool = typer.Option(
        False,
        "--explicit-llm",
        help=(
            "Run cached TRAIN-only bug-unit abstraction, embedding grouping, and "
            "group review/synthesis."
        ),
    ),
    allow_budget_overrun: bool = False,
) -> None:
    configured = Settings.load(config)
    store = CorpusStore(configured.database)
    mechanisms = build_kite_mechanisms(configured, store)
    if explicit_llm:
        if not os.getenv("OPENAI_API_KEY"):
            raise typer.BadParameter("OPENAI_API_KEY is required for --explicit-llm")
        service = OpenAIService(
            store, configured, allow_budget_overrun=allow_budget_overrun
        )
        mechanisms = build_reviewed_mechanisms(configured, store, service)
    output = configured.artifacts_dir / "kite-knowledge-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(knowledge_manifest(mechanisms), indent=2), encoding="utf-8")
    console.print(f"Built {len(mechanisms)} KITE train mechanisms: {output}")


@kite_app.command("run")
def kite_run(
    config: Path = Path("configs/full.yaml"),
    adapter: str = "bugsinpy",
    root: Path = Path("BugsInPy"),
    arm: list[str] = typer.Option([], "--arm"),  # noqa: B008
    limit: int = 15,
    k: int = 3,
    run_id: str | None = typer.Option(None, "--run-id"),
    benchmark_manifest: Path | None = typer.Option(None, "--benchmark-manifest"),  # noqa: B008
    allow_budget_overrun: bool = False,
    continue_on_error: bool = True,
) -> None:
    """Run strict reference-only KITE-lite; hidden buggy code is final-score-only."""
    if not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter("OPENAI_API_KEY is required")
    configured = Settings.load(config)
    selected = arm or [
        "direct_reference",
        "direct_reference_repaired",
        "retrieved_mechanism_text",
        "retrieved_mechanism_sbf",
    ]
    report = run_kite_experiment(
        settings=configured,
        adapter=adapter_for(adapter, root),
        arms=selected,
        limit=limit,
        k=k,
        run_id=run_id,
        benchmark_manifest=benchmark_manifest,
        allow_budget_overrun=allow_budget_overrun,
        continue_on_error=continue_on_error,
    )
    console.print(f"KITE-lite experiment complete: {report}")


@kite_app.command("status")
def kite_status(run_id: str, config: Path = Path("configs/full.yaml")) -> None:
    path = Settings.load(config).artifacts_dir / run_id / "heartbeat.json"
    if not path.exists():
        raise typer.BadParameter(f"No KITE heartbeat for {run_id}")
    console.print_json(path.read_text(encoding="utf-8"))


@kite_app.command("verify")
def kite_verify(
    config: Path = Path("configs/full.yaml"),
    benchmark_manifest: Path | None = typer.Option(None, "--benchmark-manifest"),  # noqa: B008
    output: Path | None = typer.Option(None, "--output"),  # noqa: B008
) -> None:
    """Run the machine-readable KITE paper conformance gate without API calls."""
    configured = Settings.load(config)
    mechanisms = build_kite_mechanisms(configured, CorpusStore(configured.database))
    manifest = (
        benchmark_manifest
        or configured.artifacts_dir / "kite-benchmark" / "manifest.json"
    )
    target = output or configured.artifacts_dir / "kite-verification"
    report = verify_conformance(
        settings=configured,
        mechanisms=mechanisms,
        benchmark_manifest=manifest,
    )
    json_path, markdown_path = write_conformance_report(report, target)
    console.print(
        f"KITE conformance: protocol_ready={report.protocol_ready}; "
        f"exact_paper_replication={report.exact_paper_replication}; "
        f"JSON={json_path}; report={markdown_path}"
    )


def settings(path: Path = Path("configs/v0.yaml")) -> Settings:
    return Settings.load(path)


@app.command()
def smoke(output: Path = typer.Option(Path("data/artifacts/synthetic-smoke"))) -> None:
    """Run the complete offline synthetic pipeline (explicit unsafe local fixture execution)."""
    report = run_smoke(settings(), output)
    console.print(f"[green]Smoke passed.[/] Report: {report}")


@app.command("api-smoke")
def api_smoke(config: Path = Path("configs/v0.yaml")) -> None:
    """Perform one real structured visible extraction (small API charge)."""
    if not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter("OPENAI_API_KEY is required")
    configured = Settings.load(config)
    store = CorpusStore(configured.database)
    bundle = ContextBundle(
        case_id="api-smoke",
        visible_diff="--- base/counter.py\n+++ candidate/counter.py\n@@\n- return x <= limit\n+ return x < limit",
        changed_files=["counter.py"],
        changed_symbols=["allowed"],
        nearby_tests={"test_counter.py": "def test_inside(): assert allowed(1, 2)"},
        repository_tree=["counter.py", "test_counter.py"],
        content_hash="api-smoke-v1",
    )
    descriptor = OpenAIEnricher(
        store,
        configured.fast_model,
        budget_usd=configured.api_budget_usd,
        reserve_usd=configured.api_call_reserve_usd,
    ).visible(bundle)
    console.print_json(descriptor.model_dump_json())


@dataset_app.command("inspect")
def dataset_inspect(adapter: str = "synthetic", root: Path | None = None, limit: int = 25) -> None:
    source = (
        SyntheticAdapter() if adapter == "synthetic" else BugsInPyAdapter(root or Path("BugsInPy"))
    )
    table = Table("Case", "Tests")
    cases = source.list_cases()
    for case in cases[:limit] if limit else cases:
        table.add_row(case.case_id, ", ".join(case.metadata.get("test_files", case.pre_fix_tests)))
    console.print(table)
    console.print(
        f"{len(cases)} total cases; showing {min(limit, len(cases)) if limit else len(cases)}"
    )


@dataset_app.command("prepare")
def dataset_prepare(
    adapter: str = "bugsinpy",
    root: Path = Path("BugsInPy"),
    case_id: list[str] = typer.Option([], "--case-id"),
    limit: int = 0,
    metadata_only: bool = False,
    all_cases: bool = typer.Option(False, "--all", help="Prepare every catalogued case."),
    continue_on_error: bool = typer.Option(
        False, help="Persist an exclusion and continue when a case cannot be prepared."
    ),
    config: Path = Path("configs/v0.yaml"),
) -> None:
    configured = Settings.load(config)
    source = adapter_for(adapter, root)
    store = CorpusStore(configured.database)
    cases = register_dataset(source, store)
    if metadata_only:
        console.print(f"Registered metadata for {len(cases)} cases")
        return
    if all_cases and (case_id or limit):
        raise typer.BadParameter("--all cannot be combined with --case-id or --limit")
    if not all_cases and not case_id and not limit:
        raise typer.BadParameter(
            "Refusing to clone every case. Supply --case-id, --limit, --all, or --metadata-only."
        )
    prepared = prepare_cases(
        source,
        store,
        configured,
        case_id,
        limit,
        continue_on_error=continue_on_error or all_cases,
    )
    failed = int(
        store.connection.execute(
            "SELECT COUNT(*) FROM dataset_cases WHERE dataset=? AND prepared=0 AND exclusion!=''",
            (adapter,),
        ).fetchone()[0]
    )
    console.print(f"Prepared {len(prepared)} cases under {configured.data_dir / 'workspaces'}")
    if failed:
        console.print(f"[yellow]{failed} case(s) currently carry preparation/validation exclusions.[/]")


@dataset_app.command("validate")
def dataset_validate(
    case_id: str,
    adapter: str = "bugsinpy",
    root: Path = Path("BugsInPy"),
    config: Path = Path("configs/v0.yaml"),
) -> None:
    configured = Settings.load(config)
    source = adapter_for(adapter, root)
    store = CorpusStore(configured.database)
    try:
        result = validate_counterfactual(source.get_case(case_id), configured)
    except (EnvironmentBuildError, ValidationInfrastructureError) as error:
        store.mark_prepared(case_id, False, "")
        console.print(f"[red]Validation environment/infrastructure failure:[/] {error}")
        raise typer.Exit(2) from error
    exclusion = "" if result.get("valid") else "counterfactual validation failed"
    store.mark_prepared(case_id, bool(result.get("valid")), exclusion)
    console.print_json(data=result)
    if not result.get("valid"):
        raise typer.Exit(1)


@dataset_app.command("validate-all")
def dataset_validate_all(
    adapter: str = "bugsinpy",
    root: Path = Path("BugsInPy"),
    split: str = typer.Option("", help="Optionally validate only train, dev, or eval."),
    case_id: list[str] = typer.Option([], "--case-id", help="Validate only selected case IDs."),
    case_file: Path | None = typer.Option(
        None, "--case-file", help="JSON list or object containing a case_ids list."
    ),
    limit: int = typer.Option(0, help="Maximum cases; 0 means all selected cases."),
    resume: bool = typer.Option(True, help="Reuse persisted validation artifacts."),
    config: Path = Path("configs/v0.yaml"),
) -> None:
    """Validate every prepared case independently and continue past exclusions."""
    configured = Settings.load(config)
    source = adapter_for(adapter, root)
    store = CorpusStore(configured.database)
    cases = register_dataset(source, store)
    selected_ids = set(case_id)
    if case_file:
        raw_selection = json.loads(case_file.read_text(encoding="utf-8-sig"))
        file_ids = raw_selection.get("case_ids", []) if isinstance(raw_selection, dict) else raw_selection
        if not isinstance(file_ids, list) or not all(isinstance(value, str) for value in file_ids):
            raise typer.BadParameter("--case-file must contain a JSON list of case IDs")
        selected_ids.update(file_ids)
    if selected_ids:
        cases = [case for case in cases if case.case_id in selected_ids]
        missing_ids = selected_ids.difference(case.case_id for case in cases)
        if missing_ids:
            raise typer.BadParameter(f"Unknown case IDs: {', '.join(sorted(missing_ids))}")
    if split:
        splits = store.splits() or create_splits(source, store, configured.split_seed)
        cases = [case for case in cases if splits.get(case.case_id) == split]
    summary: list[dict[str, object]] = []
    attempted = 0
    infrastructure_failure: str | None = None
    for case in cases:
        validation_path = (
            configured.artifacts_dir
            / "validations"
            / f"{case.case_id.replace(':', '-')}.json"
        )
        workspace = configured.data_dir / "workspaces" / case.case_id.replace(":", "-")
        if resume and validation_path.exists():
            result = json.loads(validation_path.read_text(encoding="utf-8"))
            if result.get("validator_version") == VALIDATOR_VERSION:
                valid = bool(result.get("valid"))
                store.mark_prepared(
                    case.case_id, valid, "" if valid else "counterfactual validation failed"
                )
                summary.append({"case_id": case.case_id, "status": "reused", "valid": valid})
                continue
        ready = (workspace / "candidate").exists() and (workspace / "reference").exists()
        if adapter == "bugsinpy":
            ready = ready and (workspace / "candidate" / ".proofplane-ready").exists()
            ready = ready and (workspace / "reference" / ".proofplane-ready").exists()
        if not ready:
            summary.append({"case_id": case.case_id, "status": "not_prepared", "valid": False})
            continue
        if limit and attempted >= limit:
            summary.append({"case_id": case.case_id, "status": "deferred", "valid": False})
            continue
        attempted += 1
        try:
            result = validate_counterfactual(case, configured)
            valid = bool(result.get("valid"))
            store.mark_prepared(
                case.case_id, valid, "" if valid else "counterfactual validation failed"
            )
            summary.append({"case_id": case.case_id, "status": "validated", "valid": valid})
        except ValidationInfrastructureError as error:
            message = f"{type(error).__name__}: {error}"
            store.mark_prepared(case.case_id, False, "")
            summary.append(
                {
                    "case_id": case.case_id,
                    "status": "infrastructure_error",
                    "valid": False,
                    "error": message,
                }
            )
            infrastructure_failure = message
            break
        except EnvironmentBuildError as error:
            message = f"{type(error).__name__}: {error}"
            store.mark_prepared(case.case_id, False, "")
            summary.append(
                {
                    "case_id": case.case_id,
                    "status": "environment_unreproduced",
                    "valid": False,
                    "error": message,
                }
            )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            store.mark_prepared(case.case_id, False, "")
            summary.append(
                {
                    "case_id": case.case_id,
                    "status": "validator_error",
                    "valid": False,
                    "error": message,
                }
            )

    if selected_ids:
        summary_name = "batch-summary-selected.json"
    else:
        summary_name = f"batch-summary-{split}.json" if split else "batch-summary-all.json"
    output = configured.artifacts_dir / "validations" / summary_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    valid_count = sum(bool(item["valid"]) for item in summary)
    not_prepared = sum(item["status"] == "not_prepared" for item in summary)
    deferred = sum(item["status"] == "deferred" for item in summary)
    console.print(
        f"Validation batch complete: {valid_count} valid result(s), {attempted} attempted this run, "
        f"{not_prepared} not prepared, {deferred} deferred; summary: {output}"
    )
    if infrastructure_failure:
        console.print(f"[red]Stopped on Docker/WSL infrastructure failure:[/] {infrastructure_failure}")
        raise typer.Exit(2)


@dataset_app.command("status")
def dataset_status(
    adapter: str = "bugsinpy", config: Path = Path("configs/v0.yaml")
) -> None:
    """Show catalog, preparation, validation, enrichment, and split counts."""
    configured = Settings.load(config)
    store = CorpusStore(configured.database)
    catalogued, prepared, excluded = store.connection.execute(
        """SELECT COUNT(*), COALESCE(SUM(prepared), 0),
                  COALESCE(SUM(CASE WHEN exclusion!='' THEN 1 ELSE 0 END), 0)
           FROM dataset_cases WHERE dataset=?""",
        (adapter,),
    ).fetchone()
    table = Table("Stage", "Count")
    table.add_row("Catalogued", str(catalogued))
    table.add_row("Prepared/valid", str(prepared))
    table.add_row("Excluded", str(excluded))
    for split_name in ("train", "dev", "eval"):
        split_count = store.connection.execute(
            """SELECT COUNT(*) FROM splits s JOIN dataset_cases d ON d.id=s.case_id
               WHERE d.dataset=? AND s.split=?""",
            (adapter, split_name),
        ).fetchone()[0]
        record_count = store.connection.execute(
            """SELECT COUNT(*) FROM records r JOIN dataset_cases d ON d.id=r.id
               WHERE d.dataset=? AND r.split=?""",
            (adapter, split_name),
        ).fetchone()[0]
        table.add_row(f"{split_name} split / enriched", f"{split_count} / {record_count}")
    console.print(table)


def registry(config: Path = Path("configs/v0.yaml")) -> VocabularyRegistry:
    return VocabularyRegistry(CorpusStore(settings(config).database).connection)


@vocab_app.command("list")
def vocab_list(config: Path = Path("configs/v0.yaml")) -> None:
    console.print(registry(config).list())


@vocab_app.command("proposals")
def vocab_proposals(config: Path = Path("configs/v0.yaml")) -> None:
    console.print(registry(config).list("proposed"))


@vocab_app.command("approve")
def vocab_approve(proposal_id: str, config: Path = Path("configs/v0.yaml")) -> None:
    registry(config).approve(proposal_id)


@vocab_app.command("merge")
def vocab_merge(
    source: str, target: str, config: Path = Path("configs/v0.yaml")
) -> None:
    registry(config).merge(source, target)


@vocab_app.command("reset-generated")
def vocab_reset_generated(
    yes: bool = typer.Option(False, "--yes"),
    config: Path = Path("configs/v0.yaml"),
) -> None:
    """Clear only generated bug-family and mechanism-primitive registry entries."""
    if not yes:
        raise typer.BadParameter("pass --yes to confirm the registry reset")
    registry(config).clear_generated()
    console.print("Cleared generated hidden-label registry entries")


@vocab_app.command("export")
def vocab_export(
    output: Path = typer.Option(Path("data/artifacts/hidden-registry.json"), "--output"),
    config: Path = Path("configs/v0.yaml"),
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry(config).snapshot(), indent=2), encoding="utf-8")
    console.print(f"Exported hidden-label registry: {output}")


@split_app.command("create")
def split_create(
    config: Path = Path("configs/v0.yaml"), adapter: str = "bugsinpy", root: Path = Path("BugsInPy")
) -> None:
    configured = Settings.load(config)
    store = CorpusStore(configured.database)
    values = create_splits(adapter_for(adapter, root), store, configured.split_seed)
    counts = {
        name: sum(value == name for value in values.values()) for name in ("train", "dev", "eval")
    }
    console.print(f"Persisted project-isolated splits: {counts}")


@corpus_app.command("enrich")
def corpus_enrich(
    split: str = "train",
    limit: int = 100,
    adapter: str = "bugsinpy",
    root: Path = Path("BugsInPy"),
    config: Path = Path("configs/v0.yaml"),
    mock: bool = False,
    case_id: list[str] = typer.Option([], "--case-id"),
    allow_budget_overrun: bool = False,
) -> None:
    configured = Settings.load(config)
    if not mock and not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter("OPENAI_API_KEY is required unless --mock is selected")
    ids = enrich_split(
        adapter_for(adapter, root),
        CorpusStore(configured.database),
        configured,
        split,
        limit,
        mock,
        case_id,
        allow_budget_overrun,
    )
    console.print(f"Enriched and persisted {len(ids)} records")


@corpus_app.command("reset-records")
def corpus_reset_records(
    config: Path = Path("configs/v0.yaml"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    """Clear enriched records and embeddings while preserving LLM cache and validation."""
    if not yes:
        raise typer.BadParameter("pass --yes to confirm the enrichment-record reset")
    store = CorpusStore(Settings.load(config).database)
    store.connection.execute("DELETE FROM records")
    store.connection.execute("DELETE FROM embeddings")
    store.connection.commit()
    console.print("Cleared enriched records and embeddings; LLM cache and validation preserved")


@corpus_app.command("enrich-all")
def corpus_enrich_all(
    split: str = "train",
    adapter: str = "bugsinpy",
    root: Path = Path("BugsInPy"),
    config: Path = Path("configs/v0.yaml"),
    resume: bool = typer.Option(True, help="Skip records already enriched for this split."),
    allow_budget_overrun: bool = False,
) -> None:
    """Enrich every validated case in a split, persisting failures and continuing."""
    if not os.getenv("OPENAI_API_KEY"):
        raise typer.BadParameter("OPENAI_API_KEY is required")
    configured = Settings.load(config)
    source = adapter_for(adapter, root)
    store = CorpusStore(configured.database)
    splits = store.splits() or create_splits(source, store, configured.split_seed)
    existing = {record.corpus_id for record in store.records(split)} if resume else set()
    selected = [
        case
        for case in source.list_cases()
        if splits.get(case.case_id) == split and case.case_id not in existing
    ]
    summary: list[dict[str, object]] = []
    for case in selected:
        validation_path = (
            configured.artifacts_dir
            / "validations"
            / f"{case.case_id.replace(':', '-')}.json"
        )
        if case.dataset == "bugsinpy":
            if not validation_path.exists():
                continue
            try:
                if not json.loads(validation_path.read_text(encoding="utf-8")).get("valid"):
                    continue
            except (json.JSONDecodeError, OSError):
                continue
        try:
            ids = enrich_split(
                source,
                store,
                configured,
                split,
                1,
                False,
                [case.case_id],
                allow_budget_overrun,
            )
            summary.append({"case_id": case.case_id, "status": "enriched", "records": ids})
        except Exception as error:
            summary.append(
                {
                    "case_id": case.case_id,
                    "status": "enrichment_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    output = configured.artifacts_dir / f"enrichment-{split}-batch-summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    count = sum(item["status"] == "enriched" for item in summary)
    console.print(f"Enriched {count}/{len(summary)} remaining valid {split} cases; summary: {output}")


@index_app.command("build")
def index_build(config: Path = Path("configs/v0.yaml")) -> None:
    configured = Settings.load(config)
    count = build_primary_index(
        CorpusStore(configured.database), configured.artifacts_dir / "primary_index.json"
    )
    console.print(f"Built {count}-record causal index; hidden payload excluded")


@run_app.command("case")
def run_case(
    case_id: str,
    arm: str = "retrieved_mechanisms_mmr",
    k: int = 5,
    adapter: str = "bugsinpy",
    root: Path = Path("BugsInPy"),
    config: Path = Path("configs/v0.yaml"),
    mock: bool = False,
    unsafe_local_execution: bool = False,
    allow_budget_overrun: bool = False,
) -> None:
    configured = Settings.load(config)
    if unsafe_local_execution:
        console.print("[yellow]WARNING: generated tests will execute on the host.[/]")
    result, artifact = execute_case(
        configured,
        adapter_for(adapter, root),
        case_id,
        arm,
        k,
        mock,
        unsafe_local_execution,
        allow_budget_overrun,
    )
    console.print(
        f"Completed {result.case_id}: {result.successful_regression_revealing_tests} successful tests"
    )
    console.print(f"Artifacts: {artifact}")


@experiment_app.command("run")
def experiment_run(
    config: Path = Path("configs/v0.yaml"),
    adapter: str = "bugsinpy",
    root: Path = Path("BugsInPy"),
    arm: list[str] = typer.Option([], "--arm"),
    limit: int = 5,
    k: int = 1,
    mock: bool = False,
    unsafe_local_execution: bool = False,
    allow_budget_overrun: bool = False,
    continue_on_error: bool = False,
) -> None:
    if unsafe_local_execution:
        console.print("[yellow]WARNING: generated tests may execute on the host.[/]")
    configured = Settings.load(config)
    arms = arm or [
        "direct",
        "brainstorm",
        "random_mechanisms",
        "retrieved_raw_context",
        "retrieved_mechanisms",
        "retrieved_mechanisms_mmr",
    ]
    report = run_experiment(
        configured,
        adapter_for(adapter, root),
        arms,
        limit,
        mock,
        unsafe_local_execution,
        allow_budget_overrun,
        k,
        continue_on_error,
    )
    console.print(f"Experiment complete: {report}")


@report_app.command("build")
def report_build(run_id: str) -> None:
    path = settings().artifacts_dir / run_id / "REPORT.md"
    raw = path.parent / "results.jsonl"
    if not raw.exists():
        raise typer.BadParameter(f"No result artifacts for {run_id}")
    from proofplane_v0.experiments.report import build_report
    from proofplane_v0.models import EvaluationResult

    results = [
        EvaluationResult.model_validate_json(line)
        for line in raw.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    console.print(build_report(results, path.parent, CorpusStore(settings().database).api_usage()))


if __name__ == "__main__":
    app()
