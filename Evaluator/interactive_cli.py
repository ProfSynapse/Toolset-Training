"""Minimal interactive CLI for LM Studio full-coverage evaluations."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

from .cli import build_metadata
from .config import EvaluatorConfig, LMStudioSettings, PromptFilter, expand_path
from .lmstudio_client import LMStudioClient, LMStudioError
from .lmstudio_cli import DEFAULT_PROMPT_SET, DEFAULT_RESULTS_DIR, default_output_paths
from .prompt_sets import load_prompt_cases
from .reporting import build_run_payload, console_summary, render_markdown, write_json
from .runner import evaluate_cases


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive LM Studio evaluator (full coverage).",
        epilog="Just run `python evaluator` to pick a model and how many runs to execute.",
    )
    parser.add_argument("--host", help="LM Studio host (defaults to LMSTUDIO_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="LM Studio port (defaults to LMSTUDIO_PORT or 1234)")
    parser.add_argument("--model", help="Optional model ID to skip the selection prompt.")
    parser.add_argument("--runs", type=int, help="How many times to run the suite (default: ask interactively).")
    parser.add_argument("--prompt-set", default=str(DEFAULT_PROMPT_SET), help="Prompt set to use (default: full_coverage.json).")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for JSON/MD artifacts.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry attempts for LM Studio calls.")
    parser.add_argument("--dry-run", action="store_true", help="Skip backend calls (schema validation only).")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    _print_banner()

    prompt_path = expand_path(args.prompt_set)
    results_dir = expand_path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not prompt_path.exists():
        print(f"Prompt set not found: {prompt_path}", file=sys.stderr)
        return 1

    list_client = LMStudioClient(
        settings=LMStudioSettings(model="__list__", **_settings_kwargs(args)),
        timeout=args.timeout,
        retries=args.retries,
    )
    model_name = args.model or _select_model(list_client)
    if not model_name:
        return 1

    run_count = args.runs if args.runs and args.runs > 0 else _prompt_run_count()

    settings = LMStudioSettings(
        model=model_name,
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
        seed=None,
        **_settings_kwargs(args),
    )
    client = LMStudioClient(settings=settings, timeout=args.timeout, retries=args.retries)

    try:
        base_config = EvaluatorConfig(
            prompts_path=prompt_path,
            output_path=None,  # Set per run
            save_markdown=True,
            filter=PromptFilter(),
            retries=args.retries,
            request_timeout=args.timeout,
            dry_run=args.dry_run,
        )
        base_config.validate()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 1

    cases = load_prompt_cases(base_config.prompts_path)
    any_errors = False
    any_failures = False

    print(color(f"\nRunning full coverage for: {model_name}", "cyan"))
    print(color(f"Prompt set: {prompt_path}", "cyan"))
    print(color(f"Runs: {run_count}\n", "cyan"))

    for idx in range(run_count):
        json_path, md_path = run_output_paths(model_name, results_dir, idx, run_count)

        config = EvaluatorConfig(
            prompts_path=base_config.prompts_path,
            output_path=json_path,
            save_markdown=True,
            filter=base_config.filter,
            retries=base_config.retries,
            request_timeout=base_config.request_timeout,
            dry_run=base_config.dry_run,
        )
        config.ensure_output_parent()
        md_path.parent.mkdir(parents=True, exist_ok=True)

        print(color(f"--- Run {idx + 1}/{run_count} ---", "magenta"))
        records = evaluate_cases(
            cases,
            client=client,
            dry_run=config.dry_run,
            on_record=_print_record_progress,
        )

        metadata = build_metadata(config, settings, len(cases), len(cases), backend="lmstudio")
        payload = build_run_payload(records, metadata=metadata)
        write_json(config.output_path, payload)
        md_path.write_text(render_markdown(records), encoding="utf-8")

        print(color(console_summary(records), _passfail_color(records)))
        print(color(f"JSON: {json_path}", "yellow"))
        print(color(f"Markdown: {md_path}\n", "yellow"))

        if any(record.error for record in records):
            any_errors = True
        if any((record.validator and not record.validator.passed) for record in records if record.error is None):
            any_failures = True

    if any_errors:
        return 3
    if any_failures:
        return 2
    return 0


def run_output_paths(model_name: str, results_dir: Path, run_index: int, total_runs: int) -> Tuple[Path, Path]:
    """Create per-run JSON/MD paths, suffixed when multiple runs are requested."""
    json_path, md_path = default_output_paths(model_name, results_dir)
    if total_runs > 1:
        json_path = json_path.parent / f"{json_path.stem}_run{run_index + 1}{json_path.suffix}"
        md_path = md_path.parent / f"{md_path.stem}_run{run_index + 1}{md_path.suffix}"
    return json_path, md_path


def _select_model(client: LMStudioClient) -> str | None:
    try:
        models = client.list_models()
    except LMStudioError as exc:
        print(f"Unable to list models from LM Studio: {exc}", file=sys.stderr)
        return None

    if not models:
        print("LM Studio did not return any models.", file=sys.stderr)
        return None

    if len(models) == 1:
        print(f"Using only available model: {models[0]}")
        return models[0]

    print(color("Select a model to evaluate:", "magenta"))
    for idx, model in enumerate(models, start=1):
        print(f"{color(f'[{idx}]', 'yellow')} {model}")

    while True:
        choice = input("Enter a number (default 1): ").strip()
        if not choice:
            return models[0]
        try:
            index = int(choice)
        except ValueError:
            print("Please enter a valid number.", file=sys.stderr)
            continue
        if 1 <= index <= len(models):
            return models[index - 1]
        print(f"Please pick a value between 1 and {len(models)}.", file=sys.stderr)


def _prompt_run_count(default: int = 1) -> int:
    while True:
        raw = input(f"How many runs? (default {default}): ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.", file=sys.stderr)
            continue
        if value > 0:
            return value
        print("Run count must be at least 1.", file=sys.stderr)


def _settings_kwargs(args: argparse.Namespace) -> dict:
    opts = {}
    if args.host:
        opts["host"] = args.host
    if args.port is not None:
        opts["port"] = args.port
    return opts


def _print_record_progress(record) -> None:
    """Emit a one-line status for each prompt as it completes."""
    status, color_name = _record_status(record)
    label = record.case.case_id
    suffix = ""
    if record.error:
        suffix = f" ({record.error})"
    elif record.validator and record.validator.issues and not record.validator.passed:
        suffix = f" ({len(record.validator.issues)} issue(s))"
    print(f"{color(f'[{status}]', color_name)} {label}{suffix}")


def _record_status(record) -> tuple[str, str]:
    if record.error:
        return "ERROR", "red"
    if record.validator and not record.validator.passed:
        return "FAIL", "red"
    if record.validator and record.validator.passed:
        return "PASS", "green"
    # Dry-run or missing validator
    return "SKIP", "yellow"


# -- Styling helpers ------------------------------------------------------- #

ANSI_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "green": "\033[32m",
    "red": "\033[31m",
}


def supports_ansi() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def color(text: str, name: str) -> str:
    if not supports_ansi():
        return text
    start = ANSI_CODES.get(name, "")
    end = ANSI_CODES["reset"] if start else ""
    return f"{start}{text}{end}"


def _print_banner() -> None:
    width = 38
    top = "+" + "=" * (width - 2) + "+"
    mid1 = "|" + "LM Studio Full Coverage".center(width - 2) + "|"
    mid2 = "|" + "Evaluator CLI".center(width - 2) + "|"

    lines = [top, mid1, mid2, top]
    if supports_ansi():
        lines = [color(line, "magenta") for line in lines]
    print("\n".join(lines) + "\n")


def _passfail_color(records) -> str:
    any_errors = any(r.error for r in records)
    any_fail = any((r.validator and not r.validator.passed) for r in records if r.error is None)
    if any_errors or any_fail:
        return "red"
    return "green"


if __name__ == "__main__":
    raise SystemExit(main())
