#!/usr/bin/env python3
"""Fix Batch 13 schema violations in syngen_toolset_v1.0.0_copilot.jsonl"""

import json
import re

def fix_batch13():
    input_file = "syngen_toolset_v1.0.0_copilot.jsonl"
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    fixes_applied = 0
    
    # Fix line 193: duplicateNote destinationPath→targetPath
    if len(lines) > 192:
        example = json.loads(lines[192])
        
        for msg in example['conversations']:
            if msg['role'] == 'assistant':
                content = msg['content']
                
                if 'vaultManager_duplicateNote' in content:
                    # Replace destinationPath with targetPath
                    new_content = content.replace('"destinationPath":', '"targetPath":')
                    
                    if new_content != content:
                        msg['content'] = new_content
                        fixes_applied += 1
                        print("✓ Fixed line 193: Changed destinationPath→targetPath in duplicateNote")
        
        lines[192] = json.dumps(example, ensure_ascii=False) + '\n'
    
    # Fix line 195: moveNote sourcePath→path, destinationPath→newPath (3 occurrences)
    if len(lines) > 194:
        example = json.loads(lines[194])
        
        for msg in example['conversations']:
            if msg['role'] == 'assistant':
                content = msg['content']
                
                if 'vaultManager_moveNote' in content:
                    # Replace sourcePath with path and destinationPath with newPath
                    new_content = content.replace('"sourcePath":', '"path":')
                    new_content = new_content.replace('"destinationPath":', '"newPath":')
                    
                    if new_content != content:
                        msg['content'] = new_content
                        fixes_applied += 1
                        print("✓ Fixed line 195: Changed sourcePath→path, destinationPath→newPath in moveNote (3 calls)")
        
        lines[194] = json.dumps(example, ensure_ascii=False) + '\n'
    
    # Write back
    with open(input_file, 'w') as f:
        f.writelines(lines)
    
    print(f"\nTotal fixes applied: {fixes_applied}")
    return fixes_applied

if __name__ == '__main__':
    fix_batch13()
