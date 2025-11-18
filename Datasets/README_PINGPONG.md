# Ping-Pong Dataset (syngen_tools_sft_pingpong_11.18.25.jsonl)

**Created:** November 18, 2025
**Source:** Restructured from `syngen_tools_sft_11.18.25.jsonl`

## Overview

This dataset restructures the original single-turn tool-calling examples into multi-turn "ping-pong" conversations that match real-world tool execution flow.

## Statistics

- **Total examples:** 5,841
- **Original examples:** 2,676
- **Expansion ratio:** 2.18x

### Completion Type Distribution

| Type | Count | Percentage |
|------|-------|------------|
| Tool call only | 2,650 | 45.4% |
| Text only | 2,552 | 43.7% |
| Text → Tool call | 639 | 10.9% |

### Prompt Type Distribution

| Type | Count | Percentage |
|------|-------|------------|
| User questions | 2,676 | 45.8% |
| Tool results | 3,165 | 54.2% |

## Format Rules

### Allowed Patterns

✅ **Tool call only** (Direct invocation)
```json
{
  "conversations": [
    {"role": "user", "content": "Open the graph view"},
    {"role": "assistant", "content": "tool_call: commandManager_executeCommand\narguments: {...}"}
  ]
}
```

✅ **Text only** (Final response after tool execution)
```json
{
  "conversations": [
    {"role": "user", "content": "Result: {success: true}"},
    {"role": "assistant", "content": "Opened the local graph view!"}
  ]
}
```

✅ **Text → Tool call** (Reasoning before tool invocation)
```json
{
  "conversations": [
    {"role": "user", "content": "Result: {sessions: [...]}"},
    {"role": "assistant", "content": "Found your session from Friday. Loading it now.\n\ntool_call: loadSession\narguments: {...}"}
  ]
}
```

### Design Decision

**Following Anthropic Claude's approach:**
- Text can precede tool calls (allows reasoning/explanation)
- Text cannot follow tool calls in same turn (tool execution interrupts)
- This matches Claude's content block format: `[{type: "text"}, {type: "tool_use"}]`

## Multi-Step Example

Original single-turn conversation:
```
User: "Load my work from last Friday"
Assistant: "tool_call: listSessions
Result: {...}
Found your session. Loading it.
tool_call: loadSession
Result: {...}
Loaded your work"
```

Restructured into ping-pong turns:

**Turn 1:**
```
User: "Load my work from last Friday"
Assistant: "tool_call: listSessions\narguments: {...}"
```

**Turn 2:**
```
User: "Result: {sessions: [...]}"
Assistant: "Found your session from Friday. Loading it now.\n\ntool_call: loadSession\narguments: {...}"
```

**Turn 3:**
```
User: "Result: {sessionLoaded: true}"
Assistant: "Loaded your work from last Friday afternoon."
```

## Benefits

1. **Realistic training:** Matches actual tool execution flow where:
   - Model outputs tool call
   - System executes and returns result
   - Model responds or makes next call

2. **Prompt-completion pairs:** Each example is a clean input→output pair:
   - Prompt: User question OR tool result
   - Completion: Tool call OR text response OR both

3. **Reasoning support:** Allows model to explain its reasoning before calling tools

4. **Better generalization:** Model learns to handle:
   - Initial requests → tool calls
   - Tool results → follow-up actions or responses
   - Multi-step workflows with intermediate reasoning

## Usage

### SFT Training

Default dataset for `Trainers/rtx3090_sft`:

```bash
cd Trainers/rtx3090_sft
./train.sh --model-size 7b
```

Config location: `Trainers/rtx3090_sft/configs/training_config.py`
```python
dataset_file: str = "syngen_tools_sft_pingpong_11.18.25.jsonl"
local_file: str = "../../Datasets/syngen_tools_sft_pingpong_11.18.25.jsonl"
```

### Direct Loading

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="Datasets/syngen_tools_sft_pingpong_11.18.25.jsonl")
```

## Generation Script

Created by: `tools/restructure_sft_dataset.py`

To regenerate:
```bash
python tools/restructure_sft_dataset.py
```

## Quality Notes

- ✅ All examples maintain proper context object structure
- ✅ Single-turn conversations only (no multi-turn chains within examples)
- ✅ All labels are True (positive examples for SFT)
- ⚠️ ~20 examples had unparseable result JSON (nested objects with arrays)
- ✅ These failed examples were handled gracefully (partial segments used)

## Next Steps

Consider:
1. Upload to HuggingFace as separate dataset version
2. Create KTO version with interleaved True/False labels
3. Generate synthetic negative examples for contrastive learning
4. Add error handling examples (failed tool calls)
