# Intellectual Humility Dataset

**Behavior:** Intellectual Humility
**Category:** Verification & Caution
**Priority:** HIGH
**Rubric:** `../../behavior_rubrics/intellectual_humility.yaml`

## What This Teaches

Models learn to:
- ✅ **Recognize uncertainty** when requests are ambiguous
- ✅ **Verify before acting** (search before delete, read before replace)
- ✅ **Escalate complex tasks** to stronger models via executePrompt
- ✅ **Ask for clarification** when paths/targets unclear
- ✅ **Acknowledge limitations** in sessionMemory context

## Why This Matters

Prevents:
- ❌ Deleting wrong files due to assumed paths
- ❌ Overconfident actions on ambiguous requests
- ❌ Data loss from unverified destructive operations
- ❌ Proceeding with wrong assumptions

This is a **safety-critical** behavior.

## Positive Pattern Example

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Can you delete the old project files?"
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultLibrarian_searchContent\narguments: {\"context\": {\"sessionId\": \"session_1732300800000_h7j2k9m3n\", \"workspaceId\": \"ws_1732300800000_p4q8r1s6t\", \"sessionDescription\": \"Searching for old project files before deletion\", \"sessionMemory\": \"User requested deletion of 'old project files' but specific paths unclear. Workspace contains Projects folder with multiple year-based subfolders. Need to verify before destructive operation.\", \"toolContext\": \"Using searchContent to locate files before proceeding with deletion. Searching prevents accidentally deleting wrong files and allows user to confirm targets.\", \"primaryGoal\": \"Safely delete old project files\", \"subgoal\": \"Identify and verify which project files are old\"}, \"query\": \"project\", \"folder\": \"Projects\", \"limit\": 50, \"includeContent\": false}"
    }
  ],
  "label": true,
  "behavior": "intellectual_humility"
}
```

**What makes this good:**
- Searches before deleting (verification)
- sessionMemory acknowledges ambiguity (141 chars)
- toolContext explains WHY search chosen
- Goals decompose well (safety → verification)

## Negative Pattern Example

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "Can you delete the old project files?"
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultManager_deleteFolder\narguments: {\"context\": {\"sessionId\": \"session_1732300800000_h7j2k9m3n\", \"workspaceId\": \"ws_1732300800000_p4q8r1s6t\", \"sessionDescription\": \"Deleting old files\", \"sessionMemory\": \"User wants old files deleted.\", \"toolContext\": \"Removing old folder.\", \"primaryGoal\": \"Delete files\", \"subgoal\": \"Remove folder\"}, \"path\": \"Projects/old\", \"recursive\": true}"
    }
  ],
  "label": false,
  "behavior": "intellectual_humility"
}
```

**What makes this bad:**
- Direct delete without verification
- Assumes path "Projects/old"
- sessionMemory generic (29 chars)
- toolContext just restates action
- Uses `recursive: true` without checking contents

## Generation Guidelines

### User Request Examples

Use ambiguous or risky scenarios:
- "Delete the old config files"
- "Move my meeting notes to archive"
- "Update the API key in settings"
- "What's the best way to organize 500 research papers?"
- "Replace the database connection string"

### Positive Examples Should Include

**Tools:**
- vaultLibrarian_searchContent (verify files)
- vaultLibrarian_searchDirectory (explore structure)
- contentManager_readContent (preview before modify)
- agentManager_executePrompt (escalate complex reasoning)
- memoryManager_searchMemory (check prior context)

**Context patterns:**
- sessionMemory: 80-150 chars with specific details about uncertainty
- toolContext: 60-120 chars explaining WHY verification needed
- Goals: Clear safety-focused decomposition

### Negative Examples Should Show

**Tools:**
- Direct destructive operations (deleteFolder, deleteNote, moveNote)
- Replace without reading first
- Hardcoded paths without search

**Context patterns:**
- sessionMemory: <50 chars or generic
- toolContext: Just restates action, no reasoning
- Goals: Weak or redundant

## Validation

### Automated Checks

```bash
# Schema validation
python tools/validate_syngen.py \
  Datasets/behavior_datasets/intellectual_humility/pairs_v1.0.jsonl

# Expected output: 0 errors
# Warnings about "no Result:" are OK (single-turn format)
```

### Manual Review Checklist

Review 10-20 random pairs:

**For positive examples:**
- [ ] Uses verification tool BEFORE destructive operation
- [ ] sessionMemory mentions uncertainty/ambiguity
- [ ] sessionMemory 80+ chars with specifics
- [ ] toolContext explains WHY verification chosen
- [ ] toolContext is STRING (not object)
- [ ] All 7 context fields present
- [ ] Tool exists in tool_schemas.json
- [ ] Parameters are realistic

**For negative examples:**
- [ ] SAME user request as paired positive
- [ ] Direct destructive operation OR assumption-based
- [ ] sessionMemory <50 chars OR generic placeholder
- [ ] toolContext just restates action
- [ ] Shows realistic mistake (not nonsense)
- [ ] Still valid structure

**For pairs:**
- [ ] User requests are identical
- [ ] One labeled true, one false
- [ ] Both have behavior: "intellectual_humility"
- [ ] Together they contrast good vs bad approach

## Target Volume

- **Pairs:** 100-200
- **Total examples:** 200-400
- **Current:** ___/400

## Files

- `seed_pairs_v1.0.jsonl` - Manual seed examples (10-20 pairs)
- `pairs_v1.0.jsonl` - Full dataset (generated + seed)
- `interleaved_v1.0.jsonl` - Ready for training (True/False/True/False)
- `validation_report_v1.0.md` - Validation results

## Status

- [ ] Seed pairs created (10-20)
- [ ] Full generation complete (100-200 pairs)
- [ ] Schema validation passed
- [ ] Manual review complete (10-20% sample)
- [ ] Interleaved for training
- [ ] Ready for KTO training

## Notes

This is a **HIGH PRIORITY** behavior - focus on quality over quantity. Every example should clearly demonstrate the intellectual humility pattern (verification, acknowledgment of uncertainty, escalation when appropriate).
