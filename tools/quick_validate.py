#!/usr/bin/env python3
"""Quick validation check for tool calls."""

import json
import re

def check_example(ex, line_num):
    """Quick check if example has valid tool call structure."""
    if len(ex.get('conversations', [])) < 2:
        return False, "No assistant message"

    content = ex['conversations'][1]['content']

    # Check for tool_call marker
    if 'tool_call:' not in content:
        return False, "No tool_call marker"

    # Check if tool name is valid (not single letter)
    tool_match = re.search(r'tool_call:\s*(\w+)', content)
    if not tool_match:
        return False, "No tool name found"

    tool_name = tool_match.group(1)
    if len(tool_name) <= 2 or '_' not in tool_name:
        return False, f"Invalid tool name: {tool_name}"

    # Check for arguments
    if 'arguments:' not in content:
        return False, "No arguments marker"

    return True, "OK"

def main():
    with open('Datasets/quality_review/improved_interleaved.jsonl', 'r') as f:
        total = 0
        passed = 0
        failed_true = 0
        failed_false = 0

        for i, line in enumerate(f, 1):
            if not line.strip():
                continue

            ex = json.loads(line)
            total += 1

            is_valid, reason = check_example(ex, i)

            if is_valid:
                passed += 1
            else:
                if ex.get('label') == True:
                    failed_true += 1
                    if failed_true <= 5:
                        print(f"Line {i} (True): {reason}")
                else:
                    failed_false += 1

    print(f"\n=== Quick Validation Summary ===")
    print(f"Total examples: {total}")
    print(f"Passed: {passed} ({passed/total*100:.1f}%)")
    print(f"Failed (True/improved): {failed_true}")
    print(f"Failed (False/poor): {failed_false}")
    print(f"\nTrue examples validation rate: {(3680-failed_true)/3680*100:.1f}%")

if __name__ == '__main__':
    main()
