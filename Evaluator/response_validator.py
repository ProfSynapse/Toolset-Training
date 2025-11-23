"""Wrapper utilities for validating model responses in multiple formats."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union
import sys
import json
from pathlib import Path

# Add tools directory to path
tools_dir = Path(__file__).parent.parent / 'tools'
if str(tools_dir) not in sys.path:
    sys.path.insert(0, str(tools_dir))

import validate_dataset


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    format: str = "text"  # "text" or "openai"


@dataclass
class ValidatorIssue:
    level: str
    message: str


@dataclass
class ValidationResult:
    passed: bool
    format_detected: str  # "text", "openai", or "unknown"
    issues: List[ValidatorIssue] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "format": self.format_detected,
            "issues": [issue.__dict__ for issue in self.issues],
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "format": tc.format}
                for tc in self.tool_calls
            ],
        }


def validate_assistant_response(response: Union[str, Dict[str, Any]]) -> ValidationResult:
    """
    Validate a single assistant response.

    Args:
        response: Either a string (text format) or dict (OpenAI format message)

    Returns:
        ValidationResult with validation status and extracted tool calls
    """
    # Convert to message format for validation
    if isinstance(response, str):
        message = {"role": "assistant", "content": response}
    elif isinstance(response, dict):
        message = response
    else:
        return ValidationResult(
            passed=False,
            format_detected="unknown",
            issues=[ValidatorIssue(level="ERROR", message=f"Response must be string or dict, got {type(response).__name__}")]
        )

    # Create a validation report
    report = validate_dataset.ValidationReport(index=0, label=True, format_detected="unknown")

    # Validate the message
    tool_calls_data = validate_dataset.validate_assistant_message(message, report)

    # Convert to our format
    issues = [ValidatorIssue(level=issue.level, message=issue.message) for issue in report.issues]
    tool_calls = [
        ToolCall(name=name, arguments=args, format=report.format_detected)
        for name, args in tool_calls_data
    ]

    return ValidationResult(
        passed=report.is_valid,
        format_detected=report.format_detected,
        issues=issues,
        tool_calls=tool_calls
    )


def extract_tool_calls(response: Union[str, Dict[str, Any]]) -> List[ToolCall]:
    """
    Extract tool calls from a response without full validation.

    Args:
        response: Either a string (text format) or dict (OpenAI format message)

    Returns:
        List of ToolCall objects
    """
    result = validate_assistant_response(response)
    return result.tool_calls


# Backward compatibility
def validate_assistant_response_text(content: str) -> ValidationResult:
    """Validate a text format assistant response (backward compatibility)."""
    return validate_assistant_response(content)
