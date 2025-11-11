# KTO Dataset Interleaving Prompt for LLM

## Objective
Fix a KTO (Kahneman-Tversky Optimization) training dataset by ensuring perfect True-False-True-False alternation throughout the entire file. The constraint is **"cannot have 2 falses in a row"** - which means the pattern must never be F-F, but True-False-True-False alternation is required.

## Critical Context: Why This Matters
TRL's KTOTrainer has a CUDA bug: when batch_size=2, if a batch contains only one label type (all True or all False), the forward method fails with "AcceleratorError: CUDA error: invalid configuration argument". Perfect True-False alternation ensures every batch has exactly 1 True and 1 False example, preventing this crash.

## Dataset File
- **Location:** `/Users/jrosenbaum/Documents/Code/Synthetic Conversations/syngen_toolset_v1.0.0_claude.jsonl`
- **Format:** JSONL (one JSON object per line, no system message)
- **Current State:** 1,123 lines with ~415 consecutive True-True pairs remaining
- **Goal:** Break up all consecutive True-True pairs by inserting False examples between them

## What You're Working With

### File Structure
Each line is a complete JSON object:
```json
{
  "conversations": [
    {
      "role": "user",
      "content": "User request text"
    },
    {
      "role": "assistant",
      "content": "tool_call: toolName\narguments: {...}\n\nResult: {...}\n\nAssistant response"
    }
  ],
  "label": true  // or false
}
```

### True Examples (Desirable)
Must have:
- **Complete context object** (ALWAYS FIRST parameter in tool_call arguments):
  - `sessionId`: format `session_<13 digits>_<9 chars>` (e.g., `session_1731020100000_b1c2d3e4f`)
  - `workspaceId`: format `ws_<13 digits>_<9 chars>` (e.g., `ws_1731020100000_g5h6i7j8k`)
  - `sessionDescription`: brief summary of session (1-2 sentences)
  - `sessionMemory`: NEVER empty - 1-2 sentences of prior context (what user is working on)
  - `toolContext`: why this tool is being called
  - `primaryGoal`: user's overall objective
  - `subgoal`: what this specific tool call achieves

- **Proper Claudesidian-MCP tool schema** with realistic parameters
- **Tool result response** with success/failure data and workspaceContext
- **Natural assistant response** interpreting the result

### False Examples (Undesirable)
Demonstrate common errors for contrastive learning:
- Missing required parameters (e.g., filePath, targetPath, path)
- Wrong tool for the task
- Empty or missing sessionMemory (show error in context if intentional)
- Invalid parameter values (e.g., invalid mode, empty string)
- Trying to operate on non-existent files/folders
- Type mismatches (e.g., opening folder as note)
- Session/workspace mismatches or non-existent IDs
- Redundant operations (e.g., moving file to same location)

Must still have:
- Complete, valid context object with proper format
- Non-empty sessionMemory
- Realistic error results (not hallucinated)
- Proper error codes and error messages

## How to Find True-True Pairs

Use this Python code to identify consecutive True-True pairs:
```python
import json

file_path = "/Users/jrosenbaum/Documents/Code/Synthetic Conversations/syngen_toolset_v1.0.0_claude.jsonl"

with open(file_path, 'r') as f:
    lines = f.readlines()

consecutive_pairs = []
for i in range(len(lines)-1):
    try:
        entry1 = json.loads(lines[i])
        entry2 = json.loads(lines[i+1])
        if entry1.get('label') == True and entry2.get('label') == True:
            consecutive_pairs.append((i+1, i+2))  # Line numbers (1-indexed)
    except:
        pass

print(f"Found {len(consecutive_pairs)} consecutive True-True pairs")
print(f"First 20 pairs: {consecutive_pairs[:20]}")
print(f"Last 10 pairs: {consecutive_pairs[-10:]}")
```

## The Insertion Process

### Step 1: Identify a True-True pair
Example: Lines 712-713 are both True
```
Line 712: {"conversations": [...], "label": true}
Line 713: {"conversations": [...], "label": true}
```

### Step 2: Create a False example
The False example should demonstrate an error pattern related to the nearby True examples. Examples of good False patterns:
- **Missing parameter error:** Tool call without required parameter(s)
- **Wrong tool error:** Using vaultLibrarian_search when should use vaultManager_listDirectory
- **Type mismatch error:** Trying to open folder as file
- **Validation error:** Empty path, invalid mode value
- **Not found error:** Non-existent workspace, file, or folder
- **Redundant operation error:** Moving file to same location, deleting non-existent item

### Step 3: Insert between the pair
Replace the original True-True pair with: True → False (new) → True

Example edit pattern:
```
OLD:
{"conversations": [{"role": "user", "content": "...True example 1..."}, ...], "label": true}
{"conversations": [{"role": "user", "content": "...True example 2..."}, ...], "label": true}

NEW:
{"conversations": [{"role": "user", "content": "...True example 1..."}, ...], "label": true}
{"conversations": [{"role": "user", "content": "...False example..."}, ...], "label": false}
{"conversations": [{"role": "user", "content": "...True example 2..."}, ...], "label": true}
```

### Step 4: Use Edit Tool
Use the Edit tool to perform replacements. Always:
1. Read the file first with Read tool
2. Include sufficient context in old_string to make the match unique
3. Replace with: original True + new False + original True
4. Verify JSON remains valid

## False Example Templates

### Template 1: Missing Required Parameter
```json
{"conversations": [
  {"role": "user", "content": "Try [action] without [required param]"},
  {"role": "assistant", "content": "tool_call: vaultManager_duplicateNote\narguments: {\"context\": {\"sessionId\": \"session_1731020120000_x9y8z7a0b\", \"workspaceId\": \"ws_1731020120000_c5d6e7f8g\", \"sessionDescription\": \"Missing params\", \"sessionMemory\": \"Testing without required parameters\", \"toolContext\": \"Attempting operation with incomplete args\", \"primaryGoal\": \"Test missing params\", \"subgoal\": \"Show error\"}, \"sourcePath\": \"Legal/template.md\"}\n\nResult: {\"success\": false, \"error\": \"Missing required parameter 'targetPath'. Cannot duplicate file without specifying target location.\", \"errorCode\": \"MISSING_REQUIRED_PARAM\", \"missingParams\": [\"targetPath\"]}\n\nI need to specify the targetPath parameter."}
], "label": false}
```

### Template 2: Invalid Parameter Value
```json
{"conversations": [
  {"role": "user", "content": "Open file with invalid parameter value"},
  {"role": "assistant", "content": "tool_call: vaultManager_openNote\narguments: {\"context\": {\"sessionId\": \"session_1731020130000_d9e8f7g6h\", \"workspaceId\": \"ws_1731020130000_i5j4k3l2m\", \"sessionDescription\": \"Invalid param\", \"sessionMemory\": \"Testing with invalid parameter value\", \"toolContext\": \"Using unsupported mode\", \"primaryGoal\": \"Test invalid value\", \"subgoal\": \"Show validation error\"}, \"path\": \"Writing/doc.md\", \"mode\": \"invalid_mode\"}\n\nResult: {\"success\": false, \"error\": \"Invalid value for 'mode' parameter. Supported modes are: 'current', 'window', 'split'. Got: 'invalid_mode'\", \"errorCode\": \"INVALID_PARAMETER_VALUE\", \"validValues\": [\"current\", \"window\", \"split\"]}\n\nThe mode parameter only supports 'current', 'window', or 'split'."}
], "label": false}
```

### Template 3: Resource Not Found
```json
{"conversations": [
  {"role": "user", "content": "Try to delete non-existent folder"},
  {"role": "assistant", "content": "tool_call: vaultManager_deleteFolder\narguments: {\"context\": {\"sessionId\": \"session_1731020140000_n9o8p7q6r\", \"workspaceId\": \"ws_1731020140000_s5t4u3v2w\", \"sessionDescription\": \"Delete missing\", \"sessionMemory\": \"Testing deletion of item that doesn't exist\", \"toolContext\": \"Deleting non-existent folder\", \"primaryGoal\": \"Test not found error\", \"subgoal\": \"Show error\"}, \"path\": \"NonExistent/Folder\"}\n\nResult: {\"success\": false, \"error\": \"Folder not found: NonExistent/Folder. The specified folder does not exist.\", \"errorCode\": \"FOLDER_NOT_FOUND\", \"path\": \"NonExistent/Folder\"}\n\nThe folder doesn't exist so it cannot be deleted."}
], "label": false}
```

### Template 4: Type Mismatch
```json
{"conversations": [
  {"role": "user", "content": "Try to open folder as a file"},
  {"role": "assistant", "content": "tool_call: vaultManager_openNote\narguments: {\"context\": {\"sessionId\": \"session_1731020150000_x7y8z9a0b\", \"workspaceId\": \"ws_1731020150000_c1d2e3f4g\", \"sessionDescription\": \"Type mismatch\", \"sessionMemory\": \"Testing opening directory\", \"toolContext\": \"Attempting to open folder\", \"primaryGoal\": \"Test type error\", \"subgoal\": \"Show error\"}, \"path\": \"Projects\"}\n\nResult: {\"success\": false, \"error\": \"Cannot open folder as note. 'Projects' is a folder, not a file. Use vaultManager_listDirectory for folders.\", \"errorCode\": \"INVALID_FILE_TYPE\", \"itemType\": \"folder\"}\n\nI cannot open folders with openNote. I should use listDirectory instead."}
], "label": false}
```

## Claudesidian-MCP Tools Reference

### Common Tool Managers and Their Tools

**vaultManager** (file/folder operations):
- `vaultManager_openNote` - Open file in view
- `vaultManager_createFolder` - Create new folder
- `vaultManager_deleteFolder` - Delete folder
- `vaultManager_deleteNote` - Delete file
- `vaultManager_duplicateNote` - Copy file
- `vaultManager_moveNote` - Move/rename file
- `vaultManager_editFolder` - Rename folder
- `vaultManager_listDirectory` - List folder contents

**contentManager** (file content CRUD):
- `contentManager_readContent` - Read file contents
- `contentManager_createContent` - Create new file with content
- `contentManager_appendContent` - Add to file end
- `contentManager_replaceContent` - Replace text in file
- `contentManager_updateContent` - Update file content

**vaultLibrarian** (advanced search):
- `vaultLibrarian_searchContent` - Search file contents
- `vaultLibrarian_searchDirectory` - Search files/folders by name
- `vaultLibrarian_searchMemory` - Search session history
- `vaultLibrarian_batch` - Execute multiple searches

**memoryManager** (sessions/workspaces):
- `memoryManager_loadWorkspace` - Load workspace context
- `memoryManager_listSessions` - List active sessions
- `memoryManager_loadSession` - Switch to different session

**agentManager** (AI agent operations):
- `agentManager_listAgents` - List configured agents
- `agentManager_createAgent` - Create new agent
- `agentManager_generateImage` - Generate images with agent

**get_tools** (meta-tool):
- Discover available tools for specific managers

## Execution Guidelines

1. **Use Edit Tool Only** - No scripts, no Bash execution for modifications
2. **Read Before Edit** - Always read the file first before editing
3. **Make String Unique** - Include enough context in old_string to ensure unique match
4. **Verify JSON** - Ensure edited file remains valid JSON (one object per line)
5. **Diverse Errors** - Vary False example error types (don't repeat same error pattern too often)
6. **Valid Sessions** - Use properly formatted sessionId and workspaceId even in False examples
7. **Non-Empty Memory** - sessionMemory must always be present and non-empty (even if just noting the test)
8. **Complete Context** - All 7 context fields required, even for False examples

## Progress Tracking

After completing insertions:
1. Check remaining pairs with Python script (run validation)
2. Update progress notes
3. Note: ~415 pairs remaining means ~585 insertions needed to complete

Current progress markers:
- Start: 593 pairs
- Completed: 8 insertions
- Remaining: ~585
- Estimated work: Large-scale systematic insertion task

## Error Handling

If you encounter issues:
1. **JSON parse errors** - Check backslash escaping, quotes are properly paired
2. **Edit match failures** - old_string wasn't unique; add more context
3. **SessionId format** - Must be `session_<13digits>_<9alphanumeric>`
4. **WorkspaceId format** - Must be `ws_<13digits>_<9alphanumeric>`
5. **Missing fields** - All 7 context fields are required

## Tips for Efficiency

1. **Batch operations** - Multiple insertions in one response if possible
2. **Vary patterns** - Don't insert same error type repeatedly
3. **Real examples** - Base False examples on actual Claudesidian error patterns
4. **Contextual errors** - Match False example errors to nearby True examples when possible
5. **Track progress** - Update todo list periodically to show forward motion

## Success Criteria

The task is complete when:
1. No consecutive True-True pairs remain in the file
2. Every pair of consecutive entries alternates: True-False-True-False-True...
3. All True examples use proper Claudesidian tool schemas with complete context
4. All False examples demonstrate realistic error patterns
5. File remains valid JSONL format (one valid JSON object per line)
6. No False-False consecutive pairs (the original constraint: "cannot have 2 falses in a row")
