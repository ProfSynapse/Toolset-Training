import json
import subprocess

# Line number where we want to insert (between two False examples)
line_number = 1113  # Change this for each insertion

# The True example to insert (edit this for each insertion)
new_example = {
    "conversations": [
        {
            "role": "user", 
            "content": "Update my current session to remind me to review the API documentation tomorrow"
        },
        {
            "role": "assistant", 
            "content": 'tool_call: memoryManager_updateSession\narguments: {"context": {"sessionId": "session_1731034646464_c7d8e9f0g", "workspaceId": "ws_1731034646464_m9n0p1q2r", "sessionDescription": "Task scheduling", "sessionMemory": "User scheduling reminder for tomorrow to review API documentation", "toolContext": "memoryManager: User updating current session with scheduled task reminder", "primaryGoal": "Schedule reminder", "subgoal": "Update session with API review reminder"}, "sessionId": "session_1731034646464_c7d8e9f0g", "memory": "TODO: Review API documentation tomorrow"}\n\nResult: {"success": true, "message": "Updated session with reminder"}\n\nI have updated your session with a reminder to review the API documentation tomorrow.'
        }
    ],
    "label": True
}
# =====================================================================

with open("syngen_toolset_v1.0.0_copilot.jsonl", "r") as f:
    lines = f.readlines()

# Insert the new example
lines.insert(line_number - 1, json.dumps(new_example) + '\n')

# Write back
with open("syngen_toolset_v1.0.0_copilot.jsonl", "w") as f:
    f.writelines(lines)

# Validate: count True-True and False-False pairs
parsed = [json.loads(line) for line in lines]
true_true = sum(1 for i in range(len(parsed)-1) if parsed[i].get('label') is True and parsed[i+1].get('label') is True)
false_false = sum(1 for i in range(len(parsed)-1) if parsed[i].get('label') is False and parsed[i+1].get('label') is False)
total_breaks = true_true + false_false

print(f"\n✅ {len(lines)} lines | TT: {true_true} | FF: {false_false} | Total breaks: {false_false}")

# Run full validator
result = subprocess.run(
    ["python3", "tools/validate_syngen.py", "syngen_toolset_v1.0.0_copilot.jsonl"],
    capture_output=True,
    text=True
)
if result.returncode != 0:
    print("\n⚠️  VALIDATION FAILED!")
    print("Errors found in True examples:")
    # Show only ERROR lines (not warnings)
    error_lines = [line for line in result.stdout.split('\n') if 'ERROR' in line and 'label=false' not in line]
    for line in error_lines[:10]:  # Show first 10 errors
        print(line)
    print("\n❌ Stopping due to validation errors. Please fix before continuing.")
    import sys
    sys.exit(1)
else:
    print("✓ Full validation passed")

# Find next FF pair
print(f"\n🔍 Finding next FF pair...")
pair_type = "FF"
with open("syngen_toolset_v1.0.0_copilot.jsonl", "r") as f:
    lines = f.readlines()

parsed = [json.loads(line) for line in lines]
pair_line = None

for i in range(len(parsed) - 1):
    if pair_type == "FF":
        if parsed[i].get('label') is False and parsed[i+1].get('label') is False:
            pair_line = i + 1
            break

if pair_line:
    first_line = pair_line
    second_line = pair_line + 1
    
    print(f"\n=== FOUND FF PAIR AT LINES {first_line}-{second_line} ===")
    print(f"\nℹ️  Next action: Insert TRUE example at line {second_line}")
    print(f"📝 Update insert_true_example.py: line_number = {second_line}")
    
    # Show snippet of both lines
    first_conv = parsed[first_line - 1]['conversations']
    second_conv = parsed[second_line - 1]['conversations']
    print(f"\nLine {first_line} (False): {first_conv[0]['content'][:80]}")
    print(f"Line {second_line} (False): {second_conv[0]['content'][:80]}")
else:
    print(f"\n🎉 NO {pair_type} PAIRS FOUND! All done!")
