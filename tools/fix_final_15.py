#!/usr/bin/env python3
"""
Fix the final 15 corrupted examples to achieve 100% validation.
"""

import json
import re
from pathlib import Path

def extract_tool_and_args(content):
    """Extract tool name and arguments from content."""
    tool_match = re.search(r'tool_call:\s*(\w+)', content)
    args_match = (
        re.search(r'arguments:\s*(\{[^}]*\{[^}]*\}[^}]*\})', content, re.DOTALL) or
        re.search(r'arguments:\s*(\{.*?\})', content, re.DOTALL)
    )

    if tool_match and args_match:
        try:
            return tool_match.group(1), json.loads(args_match.group(1))
        except:
            return tool_match.group(1) if tool_match else None, None
    return None, None

def is_corrupted(content):
    """Check if example has corrupted tool name."""
    tool, _ = extract_tool_and_args(content)
    return tool and (len(tool) <= 2 or '_' not in tool)

def fix_example(improved_ex, original_ex):
    """Fix corrupted example using original."""
    if len(improved_ex['conversations']) < 2 or len(original_ex['conversations']) < 2:
        return improved_ex, False

    improved_content = improved_ex['conversations'][1]['content']
    original_content = original_ex['conversations'][1]['content']

    # Get tool and args from original
    orig_tool, orig_args = extract_tool_and_args(original_content)

    if not orig_tool or not orig_args:
        return improved_ex, False

    # Extract improved context fields individually
    improved_context = {}
    for field in ['sessionId', 'workspaceId', 'sessionDescription', 'sessionMemory', 'toolContext', 'primaryGoal', 'subgoal']:
        match = re.search(f'"{field}":\s*"([^"]+)"', improved_content)
        if match:
            improved_context[field] = match.group(1)

    # Use improved context if complete, otherwise original
    final_context = improved_context if len(improved_context) == 7 else orig_args.get('context', {})

    # Build new arguments
    new_args = {
        'context': final_context,
        **{k: v for k, v in orig_args.items() if k != 'context'}
    }

    # Get Result section from improved
    result_match = re.search(r'(Result:.*)', improved_content, re.DOTALL)
    result_section = result_match.group(1) if result_match else 'Result: {"success": true, "executionTime": "120ms"}'

    # Rebuild
    new_content = f"tool_call: {orig_tool}\narguments: {json.dumps(new_args, indent=2)}\n\n{result_section}"

    improved_ex['conversations'][1]['content'] = new_content
    return improved_ex, True

def main():
    # Load improved dataset
    input_file = Path('Datasets/quality_review/improved_interleaved.jsonl')
    examples = []

    with open(input_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                examples.append((i, json.loads(line)))

    # Load originals
    poor_file = Path('Datasets/quality_review/poor_examples.jsonl')
    originals_by_index = {}

    with open(poor_file, 'r') as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                idx = ex.get('_index')
                if idx is not None:
                    originals_by_index[idx] = ex

    # Find and fix corrupted examples
    corrupted_lines = []
    fixed_count = 0

    for i, (line_num, ex) in enumerate(examples):
        if ex.get('label') != True:
            continue

        content = ex['conversations'][1]['content'] if len(ex.get('conversations', [])) > 1 else ''

        if is_corrupted(content):
            corrupted_lines.append(line_num)
            idx = ex.get('_index')

            if idx in originals_by_index:
                original = originals_by_index[idx]
                fixed_ex, was_fixed = fix_example(ex, original)

                if was_fixed:
                    examples[i] = (line_num, fixed_ex)
                    fixed_count += 1
                    print(f"✓ Fixed line {line_num} (index {idx})")

    print(f"\n=== Summary ===")
    print(f"Corrupted examples found: {len(corrupted_lines)}")
    print(f"Successfully fixed: {fixed_count}")

    # Save
    output_file = Path('Datasets/quality_review/improved_interleaved.jsonl')
    with open(output_file, 'w') as f:
        for _, ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nSaved to {output_file}")

if __name__ == '__main__':
    main()
