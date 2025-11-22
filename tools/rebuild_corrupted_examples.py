#!/usr/bin/env python3
"""
Rebuild corrupted improved examples from original poor examples.

Strategy:
1. Identify corrupted improved examples (can't parse tool_call)
2. Get the matching original poor example by _index
3. Extract tool name and parameters from original
4. Keep improved context from corrupted example (if parseable)
5. Rebuild with correct structure
"""

import json
import re
from pathlib import Path

def extract_tool_call_info(content):
    """Extract tool name and arguments from content."""
    tool_match = re.search(r'tool_call:\s*(\w+)', content)

    # Try different argument patterns
    args_match = (
        re.search(r'arguments:\s*(\{[^}]*\{[^}]*\}[^}]*\})', content, re.DOTALL) or  # Nested braces
        re.search(r'arguments:\s*(\{.*?\})\n', content, re.DOTALL) or  # Newline after
        re.search(r'arguments:\s*(\{.*?\})', content, re.DOTALL)  # Any ending
    )

    if tool_match and args_match:
        try:
            tool_name = tool_match.group(1)
            arguments = json.loads(args_match.group(1))
            return tool_name, arguments
        except json.JSONDecodeError:
            return None, None

    return None, None

def extract_result_and_response(content):
    """Extract Result section and any text after it."""
    result_match = re.search(r'(Result:.*)', content, re.DOTALL)
    if result_match:
        return result_match.group(1)
    return ""

def rebuild_example(improved_ex, original_ex):
    """Rebuild corrupted improved example using original structure."""
    if len(improved_ex['conversations']) < 2 or len(original_ex['conversations']) < 2:
        return improved_ex, False

    improved_content = improved_ex['conversations'][1]['content']
    original_content = original_ex['conversations'][1]['content']

    # Get tool name and args from original
    orig_tool, orig_args = extract_tool_call_info(original_content)

    if not orig_tool or not orig_args:
        return improved_ex, False

    # Try to extract improved context from corrupted content
    # Extract each field individually (more reliable than trying to match whole structure)
    improved_context = {}

    field_patterns = {
        'sessionId': r'"sessionId":\s*"([^"]+)"',
        'workspaceId': r'"workspaceId":\s*"([^"]+)"',
        'sessionDescription': r'"sessionDescription":\s*"([^"]+)"',
        'sessionMemory': r'"sessionMemory":\s*"([^"]+)"',
        'toolContext': r'"toolContext":\s*"([^"]+)"',
        'primaryGoal': r'"primaryGoal":\s*"([^"]+)"',
        'subgoal': r'"subgoal":\s*"([^"]+)"'
    }

    for field, pattern in field_patterns.items():
        match = re.search(pattern, improved_content)
        if match:
            improved_context[field] = match.group(1)

    # Only use if we got all 7 fields
    if len(improved_context) != 7:
        improved_context = None

    # Use improved context if available, otherwise original
    final_context = improved_context if improved_context else orig_args.get('context', {})

    # Build new arguments with improved context + original params
    new_args = {
        'context': final_context,
        **{k: v for k, v in orig_args.items() if k != 'context'}
    }

    # Get Result and response from improved (if present)
    result_section = extract_result_and_response(improved_content)
    if not result_section:
        # Add a basic Result section
        result_section = 'Result: {\n  "success": true,\n  "executionTime": "124ms"\n}\n'

    # Rebuild assistant message
    new_content = f"tool_call: {orig_tool}\narguments: {json.dumps(new_args, indent=2)}\n\n{result_section}"

    improved_ex['conversations'][1]['content'] = new_content
    return improved_ex, True

def main():
    # Load improved interleaved dataset
    input_file = Path('Datasets/quality_review/improved_interleaved.jsonl')
    improved_examples = []

    print('Loading improved dataset...')
    with open(input_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                ex = json.loads(line)
                improved_examples.append((i, ex))

    print(f'Loaded {len(improved_examples)} examples')

    # Load original poor examples (by index)
    poor_file = Path('Datasets/quality_review/poor_examples.jsonl')
    original_by_index = {}

    print('Loading original poor examples...')
    with open(poor_file, 'r') as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                idx = ex.get('_index')
                if idx is not None:
                    original_by_index[idx] = ex

    print(f'Loaded {len(original_by_index)} original examples')

    # Find and fix corrupted examples
    print('\nScanning for corrupted examples...')
    corrupted_count = 0
    fixed_count = 0

    for line_num, improved_ex in improved_examples:
        # Only check True (improved) examples
        if improved_ex.get('label') != True:
            continue

        if len(improved_ex['conversations']) < 2:
            continue

        content = improved_ex['conversations'][1]['content']

        # Check if tool_call can be parsed
        tool, args = extract_tool_call_info(content)

        # Corrupted if: no args, OR tool is single letter, OR tool doesn't contain underscore
        is_corrupted = not args or len(tool or '') <= 2 or '_' not in (tool or '')

        if is_corrupted:
            # Corrupted - try to rebuild
            corrupted_count += 1
            idx = improved_ex.get('_index')

            if idx in original_by_index:
                original_ex = original_by_index[idx]
                fixed_ex, was_fixed = rebuild_example(improved_ex, original_ex)

                if was_fixed:
                    # Update in list
                    for i, (ln, ex) in enumerate(improved_examples):
                        if ln == line_num:
                            improved_examples[i] = (ln, fixed_ex)
                            break

                    fixed_count += 1

                    if fixed_count <= 5:
                        print(f'  ✓ Fixed line {line_num} (index {idx})')

    print(f'\n=== Summary ===')
    print(f'Corrupted examples found: {corrupted_count}')
    print(f'Successfully fixed: {fixed_count}')
    print(f'Could not fix: {corrupted_count - fixed_count}')

    # Save fixed dataset
    output_file = Path('Datasets/quality_review/improved_interleaved_fixed.jsonl')
    with open(output_file, 'w') as f:
        for _, ex in improved_examples:
            f.write(json.dumps(ex) + '\n')

    print(f'\nSaved to {output_file}')
    print(f'File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB')

if __name__ == '__main__':
    main()
