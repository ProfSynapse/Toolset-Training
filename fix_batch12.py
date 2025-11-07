#!/usr/bin/env python3
"""Fix Batch 12 schema violations in syngen_toolset_v1.0.0_copilot.jsonl"""

import json
import re

def fix_batch12():
    input_file = "syngen_toolset_v1.0.0_copilot.jsonl"
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    fixes_applied = 0
    
    # Fix line 189 (index 188): memoryManager_updateSession
    # Need to add sessionId and change parameters
    if len(lines) > 188:
        example = json.loads(lines[188])
        
        # Find the assistant message with tool call
        for msg in example['conversations']:
            if msg['role'] == 'assistant':
                content = msg['content']
                
                # Fix updateSession call
                if 'memoryManager_updateSession' in content:
                    # Extract the sessionId from context
                    session_match = re.search(r'"sessionId":\s*"([^"]+)"', content)
                    if session_match:
                        session_id = session_match.group(1)
                        
                        # Replace the old parameters with correct ones
                        # Remove sessionDescription and primaryGoal, add sessionId to arguments
                        old_pattern = r'(tool_call: memoryManager_updateSession\narguments: \{[^}]*?"context":[^}]+?}\s*,\s*)"sessionDescription":\s*"([^"]*)",\s*"primaryGoal":\s*"([^"]*)"'
                        
                        def replacer(match):
                            context_part = match.group(1)
                            description = match.group(2)
                            goal = match.group(3)
                            return f'{context_part}"sessionId": "{session_id}", "description": "{description}", "sessionGoal": "{goal}"'
                        
                        new_content = re.sub(old_pattern, replacer, content)
                        
                        if new_content != content:
                            msg['content'] = new_content
                            fixes_applied += 1
                            print(f"✓ Fixed line 189: Added sessionId, changed sessionDescription→description, primaryGoal→sessionGoal")
        
        lines[188] = json.dumps(example, ensure_ascii=False) + '\n'
    
    # Write back
    with open(input_file, 'w') as f:
        f.writelines(lines)
    
    print(f"\nTotal fixes applied: {fixes_applied}")
    return fixes_applied

if __name__ == '__main__':
    fix_batch12()
