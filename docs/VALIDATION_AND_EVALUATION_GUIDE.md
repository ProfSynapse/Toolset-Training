# Validation and Evaluation Guide

## Overview

The validation and evaluation system has been updated to support multiple tool calling formats:
- **Text format** - Original: `tool_call: toolName\narguments: {...}`
- **OpenAI format** - New: `{"tool_calls": [{"type": "function", ...}]}`

Both systems now auto-detect format and provide consistent validation regardless of format.

## Dataset Validation

### New Universal Validator

**`tools/validate_dataset.py`** - Universal validator supporting both formats

```bash
# Auto-detect format (recommended)
python tools/validate_dataset.py Datasets/syngen_tools_sft_11.23.25_toolcall.jsonl

# Specify expected format
python tools/validate_dataset.py Datasets/syngen_tools_sft_11.23.25_toolcall.jsonl --format openai
python tools/validate_dataset.py Datasets/syngen_tools_sft_11.22.25.jsonl --format text

# Verbose mode (show warnings too)
python tools/validate_dataset.py Datasets/syngen_tools_sft_11.23.25_toolcall.jsonl --verbose
```

### Output Example

```
============================================================
Validation Summary:
  Total examples: 5515
  Format distribution:
    - Text format: 14
    - OpenAI format: 5501
    - Unknown: 0
  Validation results:
    - Passed: 5510
    - Failed: 5
    - label=false (ignored): 0

✓ Schema validation enabled (47 tool schemas loaded)
============================================================
```

### What It Validates

**Format-Agnostic (All Formats):**
- Context object structure (all 7 required fields)
- Context object must be first parameter
- sessionMemory never empty
- ID formats (session_*, ws_*)
- Tool schemas (parameters, types, required fields)
- Tool naming convention (manager_mode)

**OpenAI Format Specific:**
- `tool_calls` is a list
- Each tool call has `id`, `type`, `function`
- `type` is "function"
- `function.arguments` is a JSON string
- IDs start with `call_` (warning if not)

**Text Format Specific:**
- `tool_call:` marker format
- `arguments:` marker format
- Proper line breaks
- JSON structure in arguments

### Legacy Validator

**`tools/validate_syngen.py`** - Original text-only validator (still works)

```bash
python tools/validate_syngen.py Datasets/syngen_tools_sft_11.22.25.jsonl
```

Use this only for text format datasets. For new work, use `validate_dataset.py`.

## Response Validation (Evaluator)

### New Response Validator

**`Evaluator/response_validator.py`** - Universal response validator

```python
from Evaluator import response_validator

# Validate text format response
text_response = """tool_call: contentManager_createContent
arguments: {"context": {...}, "filePath": "note.md", "content": "..."}"""

result = response_validator.validate_assistant_response(text_response)
print(f"Format: {result.format_detected}")  # "text"
print(f"Passed: {result.passed}")
print(f"Tool calls: {len(result.tool_calls)}")

# Validate OpenAI format response
openai_response = {
    "role": "assistant",
    "content": "I'll create that note.",
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "contentManager_createContent",
                "arguments": '{"context": {...}, "filePath": "note.md", "content": "..."}'
            }
        }
    ]
}

result = response_validator.validate_assistant_response(openai_response)
print(f"Format: {result.format_detected}")  # "openai"
print(f"Passed: {result.passed}")
print(f"Tool calls: {len(result.tool_calls)}")

# Extract tool calls without full validation
tool_calls = response_validator.extract_tool_calls(openai_response)
for tc in tool_calls:
    print(f"Tool: {tc.name}")
    print(f"Args: {tc.arguments}")
```

### Integration with Evaluator

Update your evaluation scripts to use the new validator:

```python
# Old way (text only)
from Evaluator import schema_validator
result = schema_validator.validate_assistant_response(content)

# New way (both formats)
from Evaluator import response_validator
result = response_validator.validate_assistant_response(content)
# or
result = response_validator.validate_assistant_response(message_dict)
```

### Response Validation Result

```python
@dataclass
class ValidationResult:
    passed: bool                    # True if validation passed
    format_detected: str            # "text", "openai", or "unknown"
    issues: List[ValidatorIssue]    # Validation errors/warnings
    tool_calls: List[ToolCall]      # Extracted tool calls

@dataclass
class ToolCall:
    name: str                       # Tool name (e.g., "contentManager_createContent")
    arguments: Dict[str, Any]       # Parsed arguments as dict
    format: str                     # Format this was extracted from
```

## Using with Ollama/LM Studio

### Testing Models with OpenAI Format

Once your model is trained on OpenAI format data, test it with native tool calling:

```python
from openai import OpenAI
from Evaluator import response_validator

# Connect to Ollama/LM Studio
client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "contentManager_createContent",
            "description": "Create new content in vault",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "object",
                        "properties": {
                            "sessionId": {"type": "string"},
                            "workspaceId": {"type": "string"},
                            "sessionDescription": {"type": "string"},
                            "sessionMemory": {"type": "string"},
                            "toolContext": {"type": "string"},
                            "primaryGoal": {"type": "string"},
                            "subgoal": {"type": "string"}
                        },
                        "required": ["sessionId", "workspaceId", "sessionDescription",
                                   "sessionMemory", "toolContext", "primaryGoal", "subgoal"]
                    },
                    "filePath": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["context", "filePath", "content"]
            }
        }
    }
]

# Call model
response = client.chat.completions.create(
    model="your-finetuned-mistral",
    messages=[
        {"role": "user", "content": "Create a note about AI"}
    ],
    tools=tools
)

# Validate response
message = response.choices[0].message

# Convert to dict for validation
message_dict = {
    "role": "assistant",
    "content": message.content,
    "tool_calls": [
        {
            "id": tc.id,
            "type": tc.type,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
        }
        for tc in (message.tool_calls or [])
    ]
}

# Validate
result = response_validator.validate_assistant_response(message_dict)
if result.passed:
    print("✓ Valid tool call!")
    for tc in result.tool_calls:
        print(f"  Tool: {tc.name}")
        print(f"  Args: {tc.arguments}")
else:
    print("✗ Invalid tool call!")
    for issue in result.issues:
        print(f"  [{issue.level}] {issue.message}")
```

## Validation Checklist

Before training, validate your dataset:

```bash
# 1. Validate dataset structure and format
python tools/validate_dataset.py Datasets/syngen_tools_sft_11.23.25_toolcall.jsonl --verbose

# 2. Check format distribution
# Should show mostly/all OpenAI format for OpenAI datasets

# 3. Check for validation errors
# Failed examples should be investigated and fixed

# 4. Verify schema coverage
python tools/analyze_tool_coverage.py Datasets/syngen_tools_sft_11.23.25_toolcall.jsonl
```

After training, validate model outputs:

```python
from Evaluator import response_validator

# Get model response
response = model.generate(prompt)

# Validate
result = response_validator.validate_assistant_response(response)

# Check results
assert result.passed, f"Validation failed: {result.issues}"
assert len(result.tool_calls) > 0, "No tool calls found"
assert result.format_detected == "openai", "Wrong format detected"
```

## Common Validation Errors

### Missing Context Fields

```
[ERROR] Tool call #1 (toolName): context missing field 'sessionMemory'
```

**Fix:** Ensure all 7 context fields are present:
- sessionId
- workspaceId
- sessionDescription
- sessionMemory (must never be empty!)
- toolContext
- primaryGoal
- subgoal

### Invalid Arguments JSON (OpenAI)

```
[ERROR] Tool call #1: function.arguments must be a JSON string, not dict
```

**Fix:** In OpenAI format, arguments must be a JSON **string**, not a dict:

```python
# ❌ Wrong
"arguments": {"key": "value"}

# ✅ Correct
"arguments": '{"key": "value"}'
```

### Missing Required Parameters

```
[ERROR] Tool call #1 (contentManager_createContent): Missing required parameter 'filePath'
```

**Fix:** Check tool schema in `tools/tool_schemas.json` and include all required parameters.

### Context Not First Parameter

```
[ERROR] Tool call #1 (toolName): Context object must be the first field in arguments
```

**Fix:** Ensure `context` is the first key in the arguments object.

## Format Detection Logic

The validator auto-detects format using these rules:

1. **OpenAI format** detected if:
   - Message has `tool_calls` field
   - `tool_calls` is a list

2. **Text format** detected if:
   - Message content contains `tool_call:`

3. **Unknown** if:
   - Neither condition matches

## Migration Guide

### Updating from Old Validator

**Before:**
```python
from Evaluator import schema_validator

content = "tool_call: toolName\narguments: {...}"
result = schema_validator.validate_assistant_response(content)
```

**After:**
```python
from Evaluator import response_validator

# Works with both formats!
content = "tool_call: toolName\narguments: {...}"
# or
content = {"role": "assistant", "tool_calls": [...]}

result = response_validator.validate_assistant_response(content)
```

### Updating Dataset Validation Scripts

**Before:**
```bash
python tools/validate_syngen.py dataset.jsonl
```

**After:**
```bash
# Auto-detects format
python tools/validate_dataset.py dataset.jsonl

# Or specify format
python tools/validate_dataset.py dataset.jsonl --format openai
```

## Testing

### Test Dataset Validator

```bash
# Test on OpenAI format
python tools/validate_dataset.py Datasets/syngen_tools_sft_11.23.25_toolcall.jsonl

# Test on text format
python tools/validate_dataset.py Datasets/syngen_tools_sft_11.22.25.jsonl --format text

# Both should pass with appropriate format detection
```

### Test Response Validator

```python
import sys
sys.path.insert(0, 'Evaluator')
import response_validator

# Test text format
text = 'tool_call: test\narguments: {"context": {...}}'
result = response_validator.validate_assistant_response(text)
assert result.format_detected == "text"

# Test OpenAI format
openai = {"role": "assistant", "tool_calls": [...]}
result = response_validator.validate_assistant_response(openai)
assert result.format_detected == "openai"

print("✓ All tests passed!")
```

## Summary

✅ **New validators support both text and OpenAI formats**
✅ **Auto-detection makes migration easy**
✅ **Backward compatible with old code**
✅ **Same validation rules regardless of format**
✅ **Ready for Mistral + Ollama/LM Studio integration**

Use `validate_dataset.py` for datasets and `response_validator` for model outputs.
