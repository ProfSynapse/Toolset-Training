#!/usr/bin/env python3
"""
Validator for synthetic Claudesidian tool-calling datasets.

Supports multiple formats:
1. Text format: tool_call: toolName\narguments: {...}
2. OpenAI format: {"tool_calls": [{"type": "function", "function": {...}}]}

Usage:
    python tools/validate_dataset.py path/to/dataset.jsonl
    python tools/validate_dataset.py path/to/dataset.jsonl --format openai
    python tools/validate_dataset.py path/to/dataset.jsonl --format text
    python tools/validate_dataset.py path/to/dataset.jsonl --format auto  # default
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional

# Import existing validator for text format
sys.path.insert(0, str(Path(__file__).parent))
import validate_syngen

SESSION_ID_RE = re.compile(r"^session_\d{13}_[a-z0-9]{9}$")
WORKSPACE_ID_RE = re.compile(r"^ws_\d{13}_[a-z0-9]{9}$")


@dataclass
class ValidationIssue:
    level: str  # "ERROR" or "WARN"
    message: str


@dataclass
class ValidationReport:
    index: int
    format_detected: str  # "text", "openai", or "unknown"
    issues: List[ValidationIssue] = field(default_factory=list)
    label: Optional[bool] = None
    tool_calls_found: int = 0

    def add(self, level: str, message: str) -> None:
        self.issues.append(ValidationIssue(level, message))

    @property
    def is_valid(self) -> bool:
        # For undesirable examples (label=False), allow some errors
        if self.label is False:
            structural_errors = [
                issue for issue in self.issues
                if issue.level == "ERROR" and not any(x in issue.message for x in [
                    "Missing required parameter",
                    "Unexpected parameter",
                    "Invalid parameter value",
                    "Missing required 'context'",
                    "Context object must be the first field"
                ])
            ]
            return len(structural_errors) == 0
        # For desirable examples, all errors are failures
        return all(issue.level != "ERROR" for issue in self.issues)


def detect_format(message: dict) -> str:
    """
    Detect the format of an assistant message.

    Returns: "text", "openai", or "unknown"
    """
    if not isinstance(message, dict):
        return "unknown"

    role = message.get("role")
    if role != "assistant":
        return "text"  # Non-assistant messages are just text

    # Check for OpenAI format
    if "tool_calls" in message:
        return "openai"

    # Check for text format
    content = message.get("content", "")
    if isinstance(content, str) and "tool_call:" in content:
        return "text"

    # Just text content
    return "text"


def extract_tool_calls_openai(message: dict) -> List[Tuple[str, dict]]:
    """
    Extract tool calls from OpenAI format message.

    Returns list of (tool_name, arguments_dict) tuples.
    """
    tool_calls = message.get("tool_calls", [])
    if not isinstance(tool_calls, list):
        raise ValueError("tool_calls must be a list")

    extracted = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            raise ValueError("Each tool_call must be a dict")

        # Validate structure
        if tc.get("type") != "function":
            raise ValueError(f"tool_call type must be 'function', got: {tc.get('type')}")

        func = tc.get("function")
        if not isinstance(func, dict):
            raise ValueError("tool_call.function must be a dict")

        name = func.get("name")
        if not name:
            raise ValueError("Missing function name")

        # Parse arguments (they're a JSON string in OpenAI format)
        arguments_str = func.get("arguments", "")
        if not isinstance(arguments_str, str):
            raise ValueError("function.arguments must be a JSON string")

        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in function.arguments: {e}")

        extracted.append((name, arguments))

    return extracted


def validate_openai_structure(message: dict, report: ValidationReport) -> None:
    """Validate OpenAI format structure."""
    tool_calls = message.get("tool_calls")

    if not isinstance(tool_calls, list):
        report.add("ERROR", "tool_calls must be a list")
        return

    if len(tool_calls) == 0:
        report.add("WARN", "tool_calls is empty (no tool calls)")

    for idx, tc in enumerate(tool_calls, 1):
        if not isinstance(tc, dict):
            report.add("ERROR", f"Tool call #{idx}: Must be a dict")
            continue

        # Validate required fields
        if "id" not in tc:
            report.add("ERROR", f"Tool call #{idx}: Missing 'id' field")
        elif not isinstance(tc["id"], str):
            report.add("ERROR", f"Tool call #{idx}: 'id' must be a string")
        elif not tc["id"].startswith("call_"):
            report.add("WARN", f"Tool call #{idx}: ID should start with 'call_', got: {tc['id']}")

        if "type" not in tc:
            report.add("ERROR", f"Tool call #{idx}: Missing 'type' field")
        elif tc["type"] != "function":
            report.add("ERROR", f"Tool call #{idx}: type must be 'function', got: {tc['type']}")

        if "function" not in tc:
            report.add("ERROR", f"Tool call #{idx}: Missing 'function' field")
            continue

        func = tc["function"]
        if not isinstance(func, dict):
            report.add("ERROR", f"Tool call #{idx}: 'function' must be a dict")
            continue

        # Validate function fields
        if "name" not in func:
            report.add("ERROR", f"Tool call #{idx}: Missing 'function.name'")
        elif not isinstance(func["name"], str):
            report.add("ERROR", f"Tool call #{idx}: 'function.name' must be a string")
        elif "_" not in func["name"]:
            report.add("WARN", f"Tool call #{idx}: Function name '{func['name']}' doesn't follow manager_mode convention")

        if "arguments" not in func:
            report.add("ERROR", f"Tool call #{idx}: Missing 'function.arguments'")
        elif not isinstance(func["arguments"], str):
            report.add("ERROR", f"Tool call #{idx}: 'function.arguments' must be a JSON string, not {type(func['arguments']).__name__}")
        else:
            # Validate it's valid JSON
            try:
                args = json.loads(func["arguments"])
                if not isinstance(args, dict):
                    report.add("ERROR", f"Tool call #{idx}: Parsed arguments must be a dict, got: {type(args).__name__}")
            except json.JSONDecodeError as e:
                report.add("ERROR", f"Tool call #{idx}: Invalid JSON in arguments: {e}")


def validate_assistant_message(message: dict, report: ValidationReport) -> List[Tuple[str, dict]]:
    """
    Validate an assistant message and extract tool calls.

    Returns list of (tool_name, arguments_dict) tuples.
    """
    format_type = detect_format(message)
    report.format_detected = format_type

    tool_calls = []

    if format_type == "openai":
        # Validate OpenAI format structure
        validate_openai_structure(message, report)

        # Extract tool calls
        try:
            tool_calls = extract_tool_calls_openai(message)
            report.tool_calls_found = len(tool_calls)
        except ValueError as e:
            report.add("ERROR", f"Failed to extract OpenAI tool calls: {e}")
            return []

    elif format_type == "text":
        content = message.get("content", "")

        if not isinstance(content, str):
            report.add("ERROR", f"Content must be a string, got: {type(content).__name__}")
            return []

        if not content.strip():
            report.add("ERROR", "Assistant content may not be empty")
            return []

        # Use existing text format validator
        if "tool_call:" in content:
            try:
                tool_calls = validate_syngen.extract_tool_calls(content)
                report.tool_calls_found = len(tool_calls)
            except Exception as e:
                report.add("ERROR", f"Failed to extract text format tool calls: {e}")
                return []

    return tool_calls


def validate_tool_calls(tool_calls: List[Tuple[str, dict]], report: ValidationReport) -> None:
    """Validate extracted tool calls (format-agnostic)."""
    for idx, (tool_name, args) in enumerate(tool_calls, 1):
        if not tool_name:
            report.add("ERROR", f"Tool call #{idx}: Missing tool name")
            continue

        # Validate context (reuse existing validator)
        # Convert to ExampleReport for compatibility
        temp_report = validate_syngen.ExampleReport(index=report.index, label=report.label)
        validate_syngen.validate_context(args, temp_report)

        # Copy issues from temp_report
        for issue in temp_report.issues:
            report.add(issue.level, f"Tool call #{idx} ({tool_name}): {issue.message}")

        # Validate against schema (reuse existing validator)
        temp_report2 = validate_syngen.ExampleReport(index=report.index, label=report.label)
        validate_syngen.validate_tool_against_schema(tool_name, args, temp_report2, idx)

        # Copy issues
        for issue in temp_report2.issues:
            report.add(issue.level, issue.message)


def validate_example(idx: int, example: dict, expected_format: Optional[str] = None) -> ValidationReport:
    """
    Validate a single example.

    Args:
        idx: Line number
        example: The example dict
        expected_format: "text", "openai", or None for auto-detect
    """
    label = example.get("label")
    report = ValidationReport(index=idx, label=label, format_detected="unknown")

    # Validate conversations field
    conversations = example.get("conversations")
    if conversations is None:
        report.add("ERROR", "Missing 'conversations' field")
        return report

    if not isinstance(conversations, list):
        report.add("ERROR", "conversations must be an array")
        return report

    if len(conversations) == 0:
        report.add("ERROR", "conversations array must not be empty")
        return report

    # Find assistant messages and validate
    assistant_messages = [msg for msg in conversations if isinstance(msg, dict) and msg.get("role") == "assistant"]

    if not assistant_messages:
        report.add("WARN", "No assistant messages found")
        return report

    all_tool_calls = []

    for msg in assistant_messages:
        tool_calls = validate_assistant_message(msg, report)
        all_tool_calls.extend(tool_calls)

    # Validate extracted tool calls
    if all_tool_calls:
        validate_tool_calls(all_tool_calls, report)

    # Validate label
    if label is not None and not isinstance(label, bool):
        report.add("ERROR", f"Label must be a boolean if present, got: {type(label).__name__}")

    # Check expected format if specified
    if expected_format and report.format_detected != "unknown":
        if expected_format != report.format_detected and expected_format != "auto":
            report.add("WARN", f"Expected {expected_format} format but detected {report.format_detected}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Claudesidian synthetic data in multiple formats")
    parser.add_argument("path", type=Path, help="Path to JSONL file")
    parser.add_argument(
        "--format",
        choices=["auto", "text", "openai"],
        default="auto",
        help="Expected format (default: auto-detect)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all issues including warnings"
    )
    args = parser.parse_args()

    if not args.path.exists():
        sys.exit(f"File not found: {args.path}")

    reports: List[ValidationReport] = []
    expected_format = None if args.format == "auto" else args.format

    try:
        for idx, payload in validate_syngen.load_jsonl(args.path):
            reports.append(validate_example(idx, payload, expected_format))
    except ValueError as exc:
        sys.exit(str(exc))

    # Count by format
    format_counts = {"text": 0, "openai": 0, "unknown": 0}
    for r in reports:
        format_counts[r.format_detected] = format_counts.get(r.format_detected, 0) + 1

    # Only count failures from label=true examples
    invalid = [r for r in reports if not r.is_valid and r.label is not False]

    # Print issues
    for report in reports:
        if report.issues and (args.verbose or report.label is not False):
            print(f"Example line {report.index} ({report.format_detected} format):")
            for issue in report.issues:
                if args.verbose or issue.level == "ERROR":
                    print(f"  [{issue.level}] {issue.message}")
            print()

    # Print summary
    print("=" * 60)
    print("Validation Summary:")
    print(f"  Total examples: {len(reports)}")
    print(f"  Format distribution:")
    print(f"    - Text format: {format_counts['text']}")
    print(f"    - OpenAI format: {format_counts['openai']}")
    print(f"    - Unknown: {format_counts['unknown']}")
    print(f"  Validation results:")
    print(f"    - Passed: {len(reports) - len(invalid)}")
    print(f"    - Failed: {len(invalid)}")
    print(f"    - label=false (ignored): {len([r for r in reports if r.label is False])}")

    if validate_syngen.TOOL_SCHEMAS:
        print(f"\n✓ Schema validation enabled ({len(validate_syngen.TOOL_SCHEMAS)} tool schemas loaded)")
    else:
        print(f"\n⚠ Schema validation disabled (tool_schemas.json not found)")

    print("=" * 60)

    if invalid:
        sys.exit(f"\nValidation failed: {len(invalid)} example(s) with errors")
    else:
        print("\n✓ All examples passed validation!")


if __name__ == "__main__":
    main()
