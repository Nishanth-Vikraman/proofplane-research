from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from proofplane_v0.models import EvaluationResult


def _rate(group: list[EvaluationResult], field: str, denominator: str) -> float:
    bottom = sum(int(getattr(item, denominator)) for item in group)
    return sum(int(getattr(item, field)) for item in group) / bottom if bottom else 0.0


def _bootstrap_success(group: list[EvaluationResult]) -> str:
    if len(group) < 10:
        return "small sample; directional only"
    values = [float(bool(item.successful_regression_revealing_tests)) for item in group]
    rng = random.Random(17)
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(2000))
    return f"95% bootstrap CI {samples[49]:.1%}–{samples[1949]:.1%}"


def build_report(
    results: list[EvaluationResult], output: Path, api_usage: dict[str, Any] | None = None
) -> Path:
    if not results:
        raise ValueError("cannot report an empty experiment")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(item.model_dump_json() + "\n")
    fields = [
        "case_id",
        "split",
        "generation_arm",
        "retrieval_mode",
        "evaluation_bucket",
        "generated_test_count",
        "syntax_valid_count",
        "runnable_count",
        "tests_failing_on_candidate",
        "tests_passing_on_reference",
        "successful_regression_revealing_tests",
    ]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({name: getattr(item, name) for name in fields})

    datasets = sorted({item.case_id.split(":", 1)[0] for item in results})
    lines = [
        "# proofplane-v0 experiment report",
        "",
        f"Datasets: {', '.join(datasets)}. Results are directional; statistical significance is not claimed.",
        "",
        "## Results by generation arm",
        "",
        "| Arm | Cases | Success@K | Valid-test rate | Runnable rate | Candidate-only rate | Uncertainty |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm in sorted({item.generation_arm for item in results}):
        group = [item for item in results if item.generation_arm == arm]
        successes = sum(bool(item.successful_regression_revealing_tests) for item in group)
        lines.append(
            f"| {arm} | {len(group)} | {successes / len(group):.1%} | "
            f"{_rate(group, 'syntax_valid_count', 'generated_test_count'):.1%} | "
            f"{_rate(group, 'runnable_count', 'generated_test_count'):.1%} | "
            f"{_rate(group, 'successful_regression_revealing_tests', 'generated_test_count'):.1%} | "
            f"{_bootstrap_success(group)} |"
        )
    lines += ["", "## Primary comparison", ""]
    present = {item.generation_arm for item in results}
    if {"retrieved_mechanisms_mmr", "brainstorm"} <= present:
        lines.append("Primary comparison present: `retrieved_mechanisms_mmr` versus `brainstorm`.")
    else:
        lines.append(
            "Primary comparison incomplete; both `retrieved_mechanisms_mmr` and `brainstorm` are required."
        )
    lines += [
        "",
        "## Results by evaluation bucket",
        "",
        "| Bucket | Cases/arms | Successful runs |",
        "|---|---:|---:|",
    ]
    for bucket in sorted({item.evaluation_bucket for item in results}):
        group = [item for item in results if item.evaluation_bucket == bucket]
        lines.append(
            f"| {bucket} | {len(group)} | {sum(bool(x.successful_regression_revealing_tests) for x in group)} |"
        )
    lines += ["", "## Retrieval examples", ""]
    retrieved = [item for item in results if item.retrieved_corpus_ids]
    if retrieved:
        for item in retrieved[:10]:
            pairs = ", ".join(
                f"`{record_id}` ({scores.get('total', scores.get('oracle', 0)):.3f})"
                for record_id, scores in zip(item.retrieved_corpus_ids, item.retrieval_scores)
            )
            lines.append(f"- `{item.case_id}` / `{item.generation_arm}`: {pairs}")
    else:
        lines.append("No retrieval-conditioned result was present.")
    total_time = sum(sum(item.execution_times.values()) for item in results)
    lines += [
        "",
        "## API usage and execution time",
        "",
        f"- Differential test execution time: {total_time:.2f} seconds.",
    ]
    if api_usage:
        lines += [
            f"- Input tokens tracked: {api_usage.get('input_tokens', 0)}.",
            f"- Output tokens tracked: {api_usage.get('output_tokens', 0)}.",
            f"- Conservatively reserved API spend: ${api_usage.get('reserved_cost_usd', 0):.2f}.",
        ]
    else:
        lines.append("- API usage was not attached to this report.")
    exclusions = [value for item in results for value in item.exclusions]
    errors = [value for item in results for value in item.errors]
    lines += ["", "## Exclusions and execution errors", ""]
    lines.extend(f"- {value}" for value in exclusions + errors)
    if not exclusions and not errors:
        lines.append("None recorded.")
    oracle = [item for item in results if item.generation_arm.startswith("oracle")]
    lines += ["", "## Oracle diagnostic", ""]
    lines.append(
        f"{len(oracle)} oracle result(s), separately labelled and excluded from primary conclusions."
        if oracle
        else "Oracle retrieval was not run and is excluded from primary conclusions."
    )
    report = output / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
