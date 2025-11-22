#!/usr/bin/env python3
"""
Remove examples that fail validation while maintaining True/False interleaving.

For each failing improved (True) example, also remove its paired poor (False) example
to maintain the interleaved pattern.
"""

import json
from pathlib import Path
import subprocess

def get_failing_line_numbers():
    """Run validator and get line numbers of failing examples."""
    result = subprocess.run(
        ['python', 'tools/validate_syngen.py', 'Datasets/quality_review/improved_interleaved.jsonl'],
        capture_output=True,
        text=True,
        timeout=300
    )

    output = result.stdout + result.stderr
    lines = output.split('\n')

    failing_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Example line') and i+1 < len(lines) and 'ERROR' in lines[i+1]:
            line_num = int(line.split()[2].rstrip(':'))
            failing_lines.append(line_num)

    return set(failing_lines)

def main():
    print('Running validator to find failing examples...')
    failing_lines = get_failing_line_numbers()

    print(f'Found {len(failing_lines)} failing examples')

    # Load dataset
    input_file = Path('Datasets/quality_review/improved_interleaved.jsonl')
    all_examples = []

    with open(input_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                all_examples.append((i, json.loads(line)))

    print(f'Loaded {len(all_examples)} total examples')

    # Find pairs to remove
    # In interleaved dataset: odd lines are True, even lines are False
    # If odd line (True) fails, remove both odd and even (the pair)
    lines_to_remove = set()

    for line_num in failing_lines:
        ex = next((ex for ln, ex in all_examples if ln == line_num), None)

        if ex and ex.get('label') == True:
            # This is a True example that failed
            # Remove it and its paired False example
            if line_num % 2 == 1:  # Odd line (True)
                lines_to_remove.add(line_num)  # Remove True
                lines_to_remove.add(line_num + 1)  # Remove paired False
            else:  # Even line  (shouldn't happen, but handle it)
                lines_to_remove.add(line_num)
                lines_to_remove.add(line_num - 1)

    print(f'Will remove {len(lines_to_remove)} examples (pairs) to maintain interleaving')

    # Create filtered dataset
    filtered = [(ln, ex) for ln, ex in all_examples if ln not in lines_to_remove]

    # Verify interleaving
    labels = [ex.get('label') for _, ex in filtered]
    interleaving_valid = all(
        labels[i] != labels[i+1]
        for i in range(len(labels)-1)
    )

    print(f'\nFiltered dataset:')
    print(f'  Total examples: {len(filtered)}')
    print(f'  True examples: {sum(1 for _, ex in filtered if ex.get("label") == True)}')
    print(f'  False examples: {sum(1 for _, ex in filtered if ex.get("label") == False)}')
    print(f'  Interleaving valid: {interleaving_valid}')

    # Save
    output_file = Path('Datasets/quality_review/improved_interleaved_100pct.jsonl')
    with open(output_file, 'w') as f:
        for _, ex in filtered:
            f.write(json.dumps(ex) + '\n')

    print(f'\nSaved to {output_file}')
    print(f'File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB')

if __name__ == '__main__':
    main()
