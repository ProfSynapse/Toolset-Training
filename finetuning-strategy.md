# Claudesidian-MCP Fine-Tuning Strategy: Synthetic Data Generation Plan

**Document Version:** 3.0
**Created:** 2025-11-07
**Updated:** 2025-11-07
**Purpose:** Comprehensive blueprint for generating synthetic training data (multi-turn conversations with complete tool execution flows) for fine-tuning local LLMs to reliably use claudesidian-mcp tools

**Target Framework:** Unsloth (Universal Format)
**Training Focus:** Tool/function calling only (no general conversation)
**Special Features:** KTO paired examples, error recovery, clarification handling

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Findings](#research-findings)
3. [Claudesidian-MCP Tool Architecture](#claudesidian-mcp-tool-architecture)
4. [Multi-Turn Conversation Format](#multi-turn-conversation-format)
5. [Seed Examples by Tool Category](#seed-examples-by-tool-category)
6. [Synthetic Data Generation Pipeline](#synthetic-data-generation-pipeline)
7. [Advanced Techniques: KTO & Contrastive Learning](#advanced-techniques-kto--contrastive-learning)
8. [Quality Assurance & Validation](#quality-assurance--validation)
9. [Implementation Roadmap](#implementation-roadmap)
10. [References & Resources](#references--resources)

---

## 1. Executive Summary

### Objective
Generate high-quality synthetic training data for fine-tuning local LLMs to reliably use the claudesidian-mcp toolset for Obsidian vault operations.

### Key Strategy
1. **Multi-turn conversation datasets** - Complete tool execution flows showing: user query → assistant tool call → tool result → assistant response/next action
2. **Generate synthetic training data** using a more capable teacher model
3. **Focus exclusively on tool-calling competency** - No general conversation, only tool use scenarios
4. **Use Unsloth universal format** - Simple prompt/completion pairs optimized for tool calling
5. **Incorporate KTO paired examples** - Matched good/bad tool call pairs for contrastive learning
6. **Include error recovery scenarios** - Models learn to handle tool failures and use error messages
7. **Add clarification examples** - Model asks follow-up questions when user intent is unclear
8. **Create domain-rich examples** covering all 5 agent categories with realistic Obsidian workflows
9. **Include tool discovery flows** where models first use `get_tools` to determine available tools

### Critical Innovation: Multi-Turn Completeness
**Every training example must show the complete tool execution cycle:**
- User makes a request
- Assistant calls a tool with proper parameters
- **Tool returns a result** (this is often missing in datasets!)
- Assistant processes the result and either:
  - Calls another tool (using information from the first result), OR
  - Provides a final natural language response to the user

This teaches the model that tool calling is not one-shot, but a conversation where results inform next actions.

### Success Metrics
- **Tool Selection Accuracy**: >95% correct tool chosen for task
- **Parameter Accuracy**: >90% correct parameters with proper formatting
- **Multi-turn Execution**: Can process tool results and chain subsequent actions
- **Tool Discovery**: Successfully uses `get_tools` when needed to find available operations
- **Multi-step Planning**: Can chain 2-3 tools correctly in sequence with proper context accumulation
- **Error Recovery**: Recognizes tool failures, reads error messages, and corrects mistakes (>80% recovery rate)
- **Clarification Rate**: Asks clarifying questions when needed (not too often, not too rarely)
- **Result Integration**: Uses tool outputs to inform next steps and final responses
- **No Hallucination**: Only calls tools that exist with valid parameters (>98%)

---

## 2. Research Findings

### 2.1 Synthetic Data Generation (2025 State-of-the-Art)

#### Key Tools & Frameworks
1. **Meta's Synthetic Data Kit** (2025)
   - CLI tool for generating reasoning traces and QA pairs
   - Specifically designed for LLM fine-tuning
   - Supports tool calling and function calling scenarios

2. **NVIDIA Nemotron-4 340B** (2025)
   - Three-model pipeline: base, instruct, and reward models
   - Generates high-quality synthetic data for commercial use
   - Outperforms human-curated data by 73.6% (Gretel Navigator benchmarks)

3. **Gretel Navigator** (Compound AI System)
   - Agentic workflows for synthetic data generation
   - Iterative review and improvement
   - Significantly outperforms baseline approaches in benchmarks

4. **DataDreamer** (Open Source Python Library)
   - Simple code interface for LLM workflows
   - Promotes reproducibility and best practices
   - Good for research and experimentation

#### Best Practices
- **Quality over Quantity**: 5,000-10,000 high-quality examples > 100,000 low-quality
- **Diversity**: Vary user intents, tool combinations, parameter formats
- **Self-Instruct Method**: Use teacher model to generate both instructions and responses
- **Iterative Refinement**: Generate → Review → Filter → Augment → Repeat
- **Multi-Turn Completeness**: ALWAYS include tool results and subsequent actions

### 2.2 ChatML Format for Unsloth

**Primary format for this project** - Standard conversational format supported by Unsloth:

```jsonl
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Key Features**:
- ChatML format with `role`/`content` message pairs
- Full conversation in single JSON record
- System message defines behavior and available tools
- User and assistant messages alternate
- Tool calls, results, and responses embedded in assistant content
- Unsloth applies chat template during processing

### 2.3 Kahneman-Tversky Optimization (KTO)

#### What is KTO?
KTO applies **Prospect Theory** (Kahneman & Tversky) to LLM alignment:
- Humans are more sensitive to **losses** than **gains** (loss aversion)
- Models learn from binary signals: desirable vs. undesirable outputs
- **No paired preference data required** (unlike DPO)

#### Why KTO for Tool Calling?
1. **Data Efficiency**: Only need examples of "good" and "bad" tool calls (no preference pairs)
2. **Natural Fit**: Tool calls are inherently binary (correct or incorrect)
3. **Performance**: Matches or exceeds DPO+SFT performance on various model families
4. **Practical**: Easier to generate negative examples than preference pairs

#### KTO Data Format
```jsonl
{"messages": [...], "label": "desirable"}
{"messages": [...], "label": "undesirable"}
```

#### Negative Example Types for Tool Calling
1. **Wrong Tool**: Correct intent, wrong tool selected
2. **Wrong Parameters**: Correct tool, malformed or missing parameters
3. **Wrong Sequence**: Tools called in illogical order
4. **Hallucinated Tools**: Non-existent tools or parameters
5. **Incomplete Conversations**: Missing tool results or skipping result processing

---

## 3. Claudesidian-MCP Tool Architecture

### 3.1 Agent Overview

Claudesidian-MCP uses an **Agent-Mode Architecture** with 5 active agents (as of v3.1.0):

| Agent | Purpose | Mode Count | Complexity |
|-------|---------|------------|------------|
| **AgentManager** | Custom AI prompts, LLM integration, image generation | 8 modes | High |
| **ContentManager** | Note CRUD operations (create, read, update, delete) | 8 modes | Medium |
| **VaultLibrarian** | Advanced search (content, files, memory) | 4 modes | High |
| **VaultManager** | File/folder operations (move, duplicate, rename) | 9 modes | Low |
| **MemoryManager** | Session/workspace/state management | 9 modes | High |

**Total**: ~38 unique tools (agent.mode combinations)

### 3.2 Tool Naming Convention

Format: `{agentName}_{modeName}`

Examples:
- `contentManager_createContent`
- `vaultLibrarian_searchContent`
- `memoryManager_createSession`
- `vaultManager_openNote`

### 3.3 Meta-Tool: get_tools

**Special tool for tool discovery:**
```json
{
  "name": "get_tools",
  "description": "Discover available tools for specific managers",
  "parameters": {
    "managers": ["array of manager names"]
  }
}
```

This allows models to first discover what tools are available before attempting to use them.

### 3.4 Common Parameter Patterns

#### All Tools Include
```json
{
  "context": {
    "sessionId": "string",
    "workspaceId": "string (optional)",
    "sessionDescription": "string",
    "sessionMemory": "string",
    "toolContext": "string",
    "primaryGoal": "string",
    "subgoal": "string"
  }
}
```

#### Tool-Specific Parameters
- **File paths**: `filePath`, `path`, `sourcePath`, `destinationPath`
- **Content**: `content`, `newContent`, `replacement`
- **Search**: `query`, `limit`, `includeContent`, `snippetLength`
- **Session**: `name`, `description`, `initialContext`

### 3.5 Result Structure (Standardized)

```json
{
  "success": true,
  "error": "string (if success=false)",
  "data": {
    // Tool-specific result data
  },
  "workspaceContext": {
    "workspaceId": "string",
    "workspacePath": ["string"],
    "activeWorkspace": true
  }
}
```

### 3.6 Finding Actual Tool Schemas and Response Formats

**CRITICAL FOR REALISTIC SYNTHETIC DATA**: To properly mock tool results, you need the actual schemas and response formats.

#### Tool Schema Locations

Each tool's schema is defined in its mode file. The pattern is:
```
.obsidian/plugins/claudesidian-mcp/src/agents/{agentName}/modes/{modeName}Mode.ts
```

**Examples**:
- `contentManager_createContent`: `.obsidian/plugins/claudesidian-mcp/src/agents/contentManager/modes/createContentMode.ts`
- `vaultManager_openNote`: `.obsidian/plugins/claudesidian-mcp/src/agents/vaultManager/modes/openNoteMode.ts`
- `vaultLibrarian_searchContent`: `.obsidian/plugins/claudesidian-mcp/src/agents/vaultLibrarian/modes/searchContentMode.ts`
- `vaultLibrarian_searchDirectory`: `.obsidian/plugins/claudesidian-mcp/src/agents/vaultLibrarian/modes/searchDirectoryMode.ts`
- `vaultLibrarian_searchMemory`: `.obsidian/plugins/claudesidian-mcp/src/agents/vaultLibrarian/modes/searchMemoryMode.ts`

#### Schema Methods in Each Mode

Every mode file has two key methods:
1. **`getParameterSchema()`**: Returns JSON Schema for input parameters
2. **`getResultSchema()`**: Returns JSON Schema for output results

**Example from openNoteMode.ts (lines 125-169)**:
```typescript
getParameterSchema(): any {
  return {
    type: 'object',
    properties: {
      path: {
        type: 'string',
        description: 'Path to the note to open'
      },
      mode: {
        type: 'string',
        enum: ['tab', 'split', 'window', 'current'],
        description: 'Where to open the note',
        default: 'current'
      },
      focus: {
        type: 'boolean',
        description: 'Whether to focus the opened note',
        default: true
      }
    },
    required: ['path']
  };
}

getResultSchema(): any {
  return {
    type: 'object',
    properties: {
      success: { type: 'boolean' },
      error: { type: 'string' },
      data: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          opened: { type: 'boolean' },
          mode: { type: 'string' }
        }
      }
    }
  };
}
```

#### Success Response Examples

**Example 1: vaultManager_openNote** (lines 111-115):
```typescript
return this.prepareResult(true, {
  path: file.path,
  opened: true,
  mode: mode
}, undefined, params.context, parseWorkspaceContext(params.workspaceContext) || undefined);
```

**Actual result**:
```json
{
  "success": true,
  "data": {
    "path": "Meeting Notes.md",
    "opened": true,
    "mode": "current"
  },
  "workspaceContext": {
    "workspaceId": "ws_1730995200456_gx7p01v",
    "workspacePath": ["Projects", "Q1 2025"],
    "activeWorkspace": true
  }
}
```

#### Error Response Examples

**Example 1: File not found** (line 80):
```typescript
return this.prepareResult(false, undefined, `Note not found at path: ${normalizedPath}`);
```

**Actual result**:
```json
{
  "success": false,
  "error": "Note not found at path: Meeting Notes.md"
}
```

**Example 2: Missing required parameter** (line 71):
```typescript
return this.prepareResult(false, undefined, 'Path is required');
```

**Actual result**:
```json
{
  "success": false,
  "error": "Path is required"
}
```

#### Common Error Patterns Across All Tools

1. **Missing Required Parameters**:
   ```json
   {"success": false, "error": "Path is required"}
   {"success": false, "error": "Content is required"}
   {"success": false, "error": "Query is required"}
   ```

2. **File/Folder Not Found**:
   ```json
   {"success": false, "error": "Note not found at path: {path}"}
   {"success": false, "error": "Folder not found: {path}"}
   ```

3. **Invalid Paths**:
   ```json
   {"success": false, "error": "Invalid path: {path}"}
   {"success": false, "error": "Path cannot contain '..'"}
   ```

4. **Permission/Access Errors**:
   ```json
   {"success": false, "error": "Cannot access system folder"}
   {"success": false, "error": "File already exists: {path}"}
   ```

5. **Search/Query Errors**:
   ```json
   {"success": false, "error": "No results found for query: {query}"}
   {"success": false, "error": "Invalid search limit: must be between 1 and 100"}
   ```

#### How to Use This for Synthetic Data

**Step 1**: Extract schemas from mode files
```bash
# Find all mode files
find .obsidian/plugins/claudesidian-mcp/src/agents -name "*Mode.ts"

# For each mode, extract getParameterSchema() and getResultSchema()
```

**Step 2**: Study the `execute()` method to see:
- What success responses look like (look for `prepareResult(true, {...})`)
- What error responses look like (look for `prepareResult(false, undefined, "error message")`)
- Edge cases and validation logic

**Step 3**: Mock realistic results in your synthetic data
- Use actual data structures from `getResultSchema()`
- Use actual error messages from `execute()` method
- Include realistic timestamps, IDs, and file paths
- Maintain consistency with the workspace context

**Example**: For `contentManager_createContent`, you'd find:
- **Location**: `.obsidian/plugins/claudesidian-mcp/src/agents/contentManager/modes/createContentMode.ts`
- **Success result**: `{success: true, data: {filePath: "...", created: timestamp}}`
- **Error cases**: "File already exists", "Invalid path", "Content is required"

---

## 4. Multi-Turn Conversation Format

### 4.1 The Critical Pattern

**EVERY training example must follow this complete flow:**

```
User Query
    ↓
Assistant Tool Call
    ↓
Tool Result ← THIS IS CRITICAL AND OFTEN MISSING
    ↓
Assistant Response/Next Action
```

### 4.2 Why Multi-Turn Matters

Most tool-calling datasets only show:
- User query → Tool call

But miss:
- **Tool result** (what the tool actually returned)
- **How the assistant processes that result**
- **How results inform next steps**
- **Error handling** (what to do when tools fail)
- **Clarification** (what to do when user intent is unclear)

Without complete flows, models learn to call tools but not how to:
- Handle the results properly
- Recover from errors
- Ask clarifying questions
- Chain multiple tools based on previous outputs

### 4.3 Primary Format: ChatML Format for Unsloth

**Why**: ChatML is the standard conversational format supported by Unsloth with `role`/`content` message pairs

**Format**: Each record contains a `conversations` array with alternating user/assistant messages

#### Single-Turn Example (Search → Action)
```jsonl
{
  "conversations": [
    {
      "role": "system",
      "content": "You are an AI assistant with access to a workspace and a set of tools organized into bundles. Your job is to accomplish the user's goals by selecting and using the appropriate tools.\n\nAvailable tool bundles:\n- contentManager: Read, create, edit, append, and replace content in workspace files\n- vaultManager: Create, move, duplicate, rename, and delete files and folders\n- vaultLibrarian: Search for files, content, and information across the workspace\n- memoryManager: Manage sessions, workspaces, and context snapshots\n- agentManager: Execute custom AI prompts, generate images, and run batch operations\n\nYou only respond with tool calls to accomplish tasks."
    },
    {
      "role": "user",
      "content": "I need to append the latest OKR recap to the planning note, but I forget which folder it lives in."
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultLibrarian_searchContent\narguments: {\"context\": {\"sessionId\": \"session_1730995200451_k9d4b28\", \"workspaceId\": \"ws_1730995200456_gx7p01v\", \"sessionDescription\": \"Confirm OKR note location\", \"sessionMemory\": \"\", \"toolContext\": \"Locate planning doc before editing\", \"primaryGoal\": \"Find OKR Planning.md\", \"subgoal\": \"Identify folder path\"}, \"query\": \"\\\"OKR Planning.md\\\"\", \"limit\": 5, \"includeContent\": false, \"snippetLength\": 0}\n\nResult: {\"success\": true, \"data\": {\"matches\": [{\"filePath\": \"Projects/2025/OKR Planning.md\", \"score\": 0.94}]}, \"workspaceContext\": {\"workspaceId\": \"ws_1730995200456_gx7p01v\", \"workspacePath\": [\"Projects\", \"2025\"], \"activeWorkspace\": true}}\n\nThe OKR note is under Projects/2025—adding your recap now.\n\ntool_call: contentManager_appendContent\narguments: {\"context\": {\"sessionId\": \"session_1730995200451_k9d4b28\", \"workspaceId\": \"ws_1730995200456_gx7p01v\", \"sessionDescription\": \"Documenting OKR decisions\", \"sessionMemory\": \"Located Projects/2025/OKR Planning.md via vaultLibrarian_searchContent\", \"toolContext\": \"Append recap + next steps\", \"primaryGoal\": \"Update OKR Planning.md\", \"subgoal\": \"Add recap block\"}, \"filePath\": \"Projects/2025/OKR Planning.md\", \"content\": \"\\n### Sync Recap - 2025-11-07\\n- Finalized launch scope for KR1 dashboards.\\n- Marketing owns enablement brief; Ops handles refresh policy.\\n\\n**Next Steps**\\n1. Marketing to deliver enablement brief by Nov 15.\\n2. Ops to publish refresh runbook draft by Nov 18.\\n\"}\n\nResult: {\"success\": true, \"data\": {\"filePath\": \"Projects/2025/OKR Planning.md\", \"appendedLength\": 324, \"lastModified\": \"2025-11-07T15:42:11Z\"}, \"workspaceContext\": {\"workspaceId\": \"ws_1730995200456_gx7p01v\", \"workspacePath\": [\"Projects\", \"2025\", \"OKR Planning.md\"], \"activeWorkspace\": true}}\n\nRecap added to Projects/2025/OKR Planning.md and next steps captured."
    }
  ]
}
```

**Key Features**:
- ChatML format with `conversations` array containing `role`/`content` message objects
- System message defines tool bundles and assistant behavior
- User and assistant messages alternate in conversation flow
- Tool calls, results, and responses all in assistant's `content` field
- Complete multi-turn flows in single record
- Model learns full conversation patterns from examples

#### Multi-Turn Conversation Structure

**Single Interaction (Most Common - 60-70% of dataset)**:
- System message
- User makes request
- Assistant uses tool(s), shows results, and responds
- Done

**Extended Conversation (30-40% of dataset)**:
- System message
- User makes initial request
- Assistant uses tool(s) and responds
- User makes follow-up request
- Assistant continues with more tool calls

**Example - User Follow-Up Pattern**:
```jsonl
{
  "conversations": [
    {
      "role": "system",
      "content": "You are an AI assistant with access to a workspace and a set of tools organized into bundles. Your job is to accomplish the user's goals by selecting and using the appropriate tools.\n\nAvailable tool bundles:\n- contentManager: Read, create, edit, append, and replace content in workspace files\n- vaultManager: Create, move, duplicate, rename, and delete files and folders\n- vaultLibrarian: Search for files, content, and information across the workspace\n- memoryManager: Manage sessions, workspaces, and context snapshots\n- agentManager: Execute custom AI prompts, generate images, and run batch operations\n\nYou only respond with tool calls to accomplish tasks."
    },
    {
      "role": "user",
      "content": "Create a meeting note for Q4 planning"
    },
    {
      "role": "assistant",
      "content": "tool_call: contentManager_createContent\narguments: {\"context\": {\"sessionId\": \"session_1730998000001_abc123\", \"workspaceId\": \"ws_1730998000000_xyz789\", \"sessionDescription\": \"Creating Q4 planning note\", \"sessionMemory\": \"\", \"toolContext\": \"User wants new meeting note\", \"primaryGoal\": \"Create Q4 planning note\", \"subgoal\": \"Initialize with heading\"}, \"filePath\": \"Meetings/Q4 Planning.md\", \"content\": \"# Q4 Planning Session\\n\\n\"}\n\nResult: {\"success\": true, \"data\": {\"filePath\": \"Meetings/Q4 Planning.md\", \"created\": 1730998005000}}\n\nCreated Meetings/Q4 Planning.md with the heading."
    },
    {
      "role": "user",
      "content": "Can you add sections for OKRs and action items?"
    },
    {
      "role": "assistant",
      "content": "tool_call: contentManager_appendContent\narguments: {\"context\": {\"sessionId\": \"session_1730998000001_abc123\", \"workspaceId\": \"ws_1730998000000_xyz789\", \"sessionDescription\": \"Adding structure to Q4 planning note\", \"sessionMemory\": \"Created Meetings/Q4 Planning.md\", \"toolContext\": \"User wants OKRs and action items sections\", \"primaryGoal\": \"Structure Q4 planning note\", \"subgoal\": \"Add sections\"}, \"filePath\": \"Meetings/Q4 Planning.md\", \"content\": \"\\n## OKRs\\n\\n## Action Items\\n\"}\n\nResult: {\"success\": true, \"data\": {\"filePath\": \"Meetings/Q4 Planning.md\", \"appendedLength\": 32, \"lastModified\": \"2025-11-07T16:15:42Z\"}}\n\nAdded OKRs and Action Items sections to your planning note."
    }
  ]
}
```

**Key Points**:
1. **Full conversation in one record**: System + all user/assistant turns in the `conversations` array
2. **Role alternation**: user → assistant → user → assistant
3. **Tool results embedded**: Tool call, result, and response all in assistant's `content` field
4. **sessionMemory accumulates**: Empty string → "Created Meetings/Q4 Planning.md"
5. **Natural conversation flow**: Each turn builds on earlier work

**Why This Works**:
- Unsloth's chat template processing handles the conversation array
- Model learns from complete interaction patterns
- Supports both single and multi-turn naturally
- Training on full conversations teaches context awareness

### 4.4 KTO Paired Examples (Good/Bad Pairs)

**Critical for contrastive learning**: Every good example should have a corresponding bad example showing the same scenario done incorrectly.

#### Good Example (Desirable)
```jsonl
{
  "conversations": [
    {
      "role": "system",
      "content": "You are an AI assistant with access to a workspace and a set of tools organized into bundles. Your job is to accomplish the user's goals by selecting and using the appropriate tools.\n\nAvailable tool bundles:\n- contentManager: Read, create, edit, append, and replace content in workspace files\n- vaultManager: Create, move, duplicate, rename, and delete files and folders\n- vaultLibrarian: Search for files, content, and information across the workspace\n- memoryManager: Manage sessions, workspaces, and context snapshots\n- agentManager: Execute custom AI prompts, generate images, and run batch operations\n\nYou only respond with tool calls to accomplish tasks."
    },
    {
      "role": "user",
      "content": "Create a new project folder called 'Q1 2025'"
    },
    {
      "role": "assistant",
      "content": "tool_call: vaultManager_createFolder\narguments: {\"context\": {\"sessionId\": \"session_1730998100000_def456\", \"workspaceId\": \"ws_1730998000000_xyz789\", \"sessionDescription\": \"Creating Q1 2025 folder\", \"sessionMemory\": \"\", \"toolContext\": \"User wants new project folder\", \"primaryGoal\": \"Create Q1 2025 folder\", \"subgoal\": \"Initialize folder structure\"}, \"path\": \"Projects/Q1 2025\"}\n\nResult: {\"success\": true, \"data\": {\"path\": \"Projects/Q1 2025\", \"created\": 1699564800000}}\n\nCreated folder 'Projects/Q1 2025'."
    }
  ]
}
```

#### Bad Example (Undesirable) - Same Scenario
```jsonl
{
  "conversations": [
    {
      "role": "system",
      "content": "You are an AI assistant with access to a workspace and a set of tools organized into bundles. Your job is to accomplish the user's goals by selecting and using the appropriate tools.\n\nAvailable tool bundles:\n- contentManager: Read, create, edit, append, and replace content in workspace files\n- vaultManager: Create, move, duplicate, rename, and delete files and folders\n- vaultLibrarian: Search for files, content, and information across the workspace\n- memoryManager: Manage sessions, workspaces, and context snapshots\n- agentManager: Execute custom AI prompts, generate images, and run batch operations\n\nYou only respond with tool calls to accomplish tasks."
    },
    {
      "role": "user",
      "content": "Create a new project folder called 'Q1 2025'"
    },
    {
      "role": "assistant",
      "content": "tool_call: contentManager_createContent\narguments: {\"context\": {\"sessionId\": \"session_1730998100000_def456\", \"workspaceId\": \"ws_1730998000000_xyz789\", \"sessionDescription\": \"Creating Q1 2025 folder\", \"sessionMemory\": \"\", \"toolContext\": \"User wants new project folder\", \"primaryGoal\": \"Create Q1 2025 folder\", \"subgoal\": \"Initialize folder structure\"}, \"filePath\": \"Q1 2025\", \"content\": \"\"}"
    }
  ]
}
```

**Why undesirable**:
- Used wrong tool (createContent instead of createFolder)
- Missing tool result processing
- Incomplete conversation

**Pairing Strategy**: For each scenario, create 1 good example + 2-3 bad variations

### 4.5 Error Recovery Examples

#### Positive Example
```jsonl
{
  "messages": [...complete multi-turn flow...],
  "label": "desirable"
}
```

#### Negative Example (Incomplete Flow)
```jsonl
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Create a new note called 'Meeting Notes'"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_456",
        "type": "function",
        "function": {
          "name": "contentManager_createContent",
          "arguments": "{\"filePath\": \"Meeting Notes.md\", \"content\": \"# Meeting\\n\"}"
        }
      }]
    }
    // MISSING: Tool result and assistant response
  ],
  "label": "undesirable"
}
```

**Why undesirable**: Conversation ends after tool call without showing tool result or assistant response.

---

## 5. Seed Examples by Tool Category

*[Keep all existing seed examples from the original document - they already show complete multi-turn flows]*

---

## 6. Synthetic Data Generation Pipeline

### 6.1 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Teacher Model (Capable LLM)               │
│                                                               │
│  Input: Task Template + Agent Schema + Workflow Type         │
│  Output: Multi-turn Conversation with Tool Execution Flows   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Quality Filter (Rule-Based)              │
│                                                               │
│  - Validate tool names exist                                 │
│  - Check parameter completeness                              │
│  - Verify context structure                                  │
│  - Ensure logical tool order                                 │
│  - VERIFY TOOL RESULTS PRESENT AFTER EVERY TOOL CALL         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Negative Example Generator                 │
│                                                               │
│  - Mutate correct examples to create errors                  │
│  - Generate 5 error types per example                        │
│  - Label as "undesirable"                                    │
│  - Include incomplete conversation flows                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Diversity Augmenter                      │
│                                                               │
│  - Paraphrase user queries (10 variants per template)        │
│  - Vary parameter values (paths, content, limits)            │
│  - Add contextual variations (session memory, goals)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      JSONL Formatter                         │
│                                                               │
│  - Convert to OpenAI chat format                             │
│  - Add system prompts                                        │
│  - Include tool results                                      │
│  - Export with labels for KTO                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Train / Validation Split                    │
│                                                               │
│  - 80% training data                                         │
│  - 10% validation data                                       │
│  - 10% test data                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Task Template Categories

Create templates for each agent category:

1. **ContentManager Tasks** (30% of dataset)
   - Create new notes (10%)
   - Read existing notes (10%)
   - Update/append content (5%)
   - Delete content (5%)

2. **VaultManager Tasks** (25% of dataset)
   - Open notes (10%)
   - File operations (move, duplicate, rename) (10%)
   - Folder operations (5%)

3. **VaultLibrarian Tasks** (20% of dataset)
   - Content search (10%)
   - Universal search (5%)
   - Directory search (5%)

4. **MemoryManager Tasks** (10% of dataset)
   - Session management (5%)
   - Workspace operations (5%)

5. **Multi-Tool Workflows** (15% of dataset)
   - 2-step chains (8%)
   - 3-step chains (5%)
   - Tool discovery + execution (2%)

6. **Tool Discovery Scenarios** (3% of dataset)
   - Using `get_tools` to find available operations
   - Exploring agent capabilities before action

7. **Error Recovery Scenarios** (10% of dataset)
   - Missing parameters detected and fixed
   - File not found, create instead
   - Wrong tool chosen, correct with right tool
   - Invalid path, ask for clarification

8. **Clarification Scenarios** (7% of dataset)
   - Ambiguous file names
   - Missing content specifications
   - Unclear user intent
   - Reasonable assumptions vs. questions

### 6.3 Generation Prompts

#### Template for Single-Tool Examples (Unsloth Format)
```
You are generating synthetic training data for fine-tuning an LLM to use workspace management tools.

CRITICAL: Generate COMPLETE multi-turn conversations in Unsloth universal format showing:
1. User query (natural language)
2. Assistant tool call (with all parameters)
3. Tool result (from the tool execution)
4. Assistant response (processing the result)

Tool Bundle: {bundle_name}
Tool: {tool_name}
Tool Schema: {tool_schema_json}

SYSTEM PROMPT TO USE:
You are an AI assistant with access to a workspace and a set of tools organized into bundles. Your job is to accomplish the user's goals by selecting and using the appropriate tools.

Available tool bundles:
- contentManager: Read, create, edit, append, and replace content in workspace files
- vaultManager: Create, move, duplicate, rename, and delete files and folders
- vaultLibrarian: Search for files, content, and information across the workspace
- memoryManager: Manage sessions, workspaces, and context snapshots
- agentManager: Execute custom AI prompts, generate images, and run batch operations

You only respond with tool calls to accomplish tasks.

FORMAT:
{
  "prompt": "<|system|>{system_prompt_above}</s>\n<|user|>{user_query}</s>\n<|assistant|>",
  "completion": "tool_call: {tool_name}\narguments: {json_args}\n</s>\n<|tool|>Result: {tool_result}</s>\n<|assistant|>{final_response}</s>"
}

REQUIREMENTS:
- Use special tokens: <|system|>, <|user|>, <|assistant|>, <|tool|>, </s>
- Complete conversation in single prompt/completion pair
- Tool results embedded in completion
- Final assistant response summarizes action
- NO general conversation - only tool calling scenarios

Vary the following:
- User phrasing (formal, casual, terse, verbose)
- File paths (different folders, file names)
- Content types (notes, tasks, meeting notes, project docs)
- Context (different session scenarios, goals)
- Tool results (success cases, realistic data)

Output format: Unsloth universal format JSONL
```

#### Template for Multi-Tool Workflows
```
You are generating synthetic training data for multi-tool workflows.

CRITICAL: Generate COMPLETE multi-turn conversations with tool chaining:

Workflow Type: {workflow_type}
Tools Involved: {tool_list}
Tool Schemas: {schemas_json}

Generate a user request that requires chaining these tools in sequence.
The conversation must follow this pattern:
1. System message
2. User query
3. Assistant tool call #1
4. Tool result #1 (role: "tool")
5. Assistant tool call #2 (using data from result #1)
6. Tool result #2 (role: "tool")
7. [Optional] Assistant tool call #3
8. [Optional] Tool result #3
9. Final assistant summary (natural language)

CRITICAL REQUIREMENTS:
- Each tool call MUST be followed by its tool result
- sessionMemory MUST accumulate: "" → "Created file X" → "Created file X, opened it"
- Each tool call must reference previous results in context or parameters
- Tool results must be realistic and match tool schemas
- Final assistant message summarizes ALL actions taken
- Realistic error scenarios included (10% failure rate)

Example flow for "Create note and open it":
- User: "Create a meeting note and open it"
- Assistant: tool_call(createContent)
- Tool: {"success": true, "data": {"filePath": "Meeting.md"}}
- Assistant: tool_call(openNote, sessionMemory="Created Meeting.md")
- Tool: {"success": true, "data": {"opened": true}}
- Assistant: "I've created 'Meeting.md' and opened it for you."

Output format: OpenAI chat completion JSONL with complete multi-turn flow
```

### 6.4 Diversity Augmentation Strategies

#### Query Paraphrasing
```python
original = "Create a new note for meeting minutes"
variants = [
    "Make a note to store meeting minutes",
    "I need a new file for today's meeting notes",
    "Start a fresh note for the meeting",
    "Create meeting notes document",
    # ... 6 more variants
]
```

#### Parameter Variations
```python
file_paths = [
    "Meeting Notes.md",
    "Meetings/2025-11-07.md",
    "Daily/Meeting Minutes.md",
    "Work/Meetings/Client Call.md"
]

content_styles = [
    "# Heading\n\nContent",
    "## Section\n- Bullet\n- Points",
    "Title\n===\n\nParagraph text",
    "---\ntags: [meeting]\n---\n\n# Meeting"
]
```

#### Context Variations
```python
contexts = [
    {
        "sessionDescription": "Daily planning session",
        "primaryGoal": "Organize daily tasks",
        "sessionMemory": ""
    },
    {
        "sessionDescription": "Project documentation work",
        "primaryGoal": "Update project docs",
        "sessionMemory": "Created 3 files already"
    },
    # ... more variants
]
```

### 6.5 Quality Filters

#### Validation Rules
```python
def validate_example(example):
    checks = [
        tool_name_exists(example['tool_calls'][0]['function']['name']),
        all_required_params_present(example['tool_calls'][0]['function']['arguments']),
        context_structure_valid(example['tool_calls'][0]['function']['arguments']['context']),
        file_paths_realistic(example['tool_calls'][0]['function']['arguments']),
        tool_result_realistic(example['tool_result']),
        no_hallucinated_fields(example),
        # NEW: Verify multi-turn completeness
        has_tool_result_after_each_call(example),
        has_final_assistant_response(example)
    ]
    return all(checks)
```

#### Error Detection
- Tool name not in approved list
- Missing required parameters (filePath, content, context)
- Malformed context (missing fields)
- Illogical file paths ("//" or ".." patterns)
- Tool results that don't match schema
- Non-existent parameters in arguments
- **Missing tool results after tool calls**
- **Incomplete conversations (no final assistant response)**

### 6.6 Target Dataset Size

| Category | Examples | Positive | Paired Negative | Total |
|----------|----------|----------|-----------------|-------|
| ContentManager | 2,310 | 2,310 | 1,386 | 3,696 |
| VaultManager | 1,925 | 1,925 | 1,155 | 3,080 |
| VaultLibrarian | 1,540 | 1,540 | 924 | 2,464 |
| MemoryManager | 770 | 770 | 462 | 1,232 |
| Multi-Tool Workflows | 1,270 | 1,270 | 762 | 2,032 |
| Tool Discovery | 254 | 254 | 152 | 406 |
| Error Recovery | 847 | 847 | 508 | 1,355 |
| Clarification | 593 | 593 | 355 | 948 |
| **Total** | **9,509** | **9,509** | **5,704** | **15,213** |

**Rationale**:
- 9,509 positive examples for supervised learning (all multi-turn, tool-calling only)
- 5,704 paired negative examples for KTO alignment (~60% pairing ratio)
- All examples include complete tool execution flows
- 10% error recovery scenarios (847 examples)
- 7% clarification scenarios (593 examples)
- 80/10/10 split = 12,170 train / 1,522 val / 1,521 test
- **Format**: Unsloth universal format (simple prompt/completion pairs)

### 6.7 Live Example Generation Workflow & Tracker

To keep JSONL creation disciplined, we will operate a single-example cadence plus a lightweight tracker that we update in small batches.

#### Operational Rhythm (3 positives → 1 negative → tracker update)
1. Pick a scenario from the backlog/tracker that needs coverage and note the target agent + mode.
2. Generate the **positive** multi-turn conversation, grounding every argument/result in the real schema, stamp the record with `"label": "desirable"`, run the Section 6.5 QA checklist, then append it to `Synthetic Conversations/syngen_toolset_v1.0.0.jsonl` (create this file if it does not exist yet; bump the filename version for future dataset iterations).
3. Repeat step 2 until **three** positive examples are completed for the batch; log metadata for each as you go.
4. Select one of the three scenarios (or a closely-related variant) and craft the paired **negative** example, showing the undesirable behavior for that flow. Set `"label": "undesirable"`, run QA again, and append it after the positives to keep the batch grouped.
5. Once the three positives + one negative are written, capture the example IDs in your working notes and update the tracker tables plus the batch log entry below.

#### Logging & Tracker Update Rules
- Keep this document as the single source of truth for counts; reference entries when planning the next batch.
- When updating counts, include the ids you incremented in the batch note to preserve traceability.
- If an example spans multiple categories (e.g., Multi-Tool + Clarification), increment every applicable row.
- Flag any AgentManager usage in the batch note so we can evaluate whether a dedicated row/target is needed.

#### Dataset Coverage Tracker (baseline)

| Category | Target Positives | Completed Positives | Completed Negatives | Last Update | Notes |
|----------|------------------|---------------------|---------------------|-------------|-------|
| ContentManager | 2,310 | 8 | 0 | 2025-11-07 | Added Iceland reminder, gratitude recovery, spark board, song verse, festival tasks |
| VaultManager | 1,925 | 6 | 1 | 2025-11-07 | Includes Weekly Retro move, NaNo folder, festival reorg, book club duplicate |
| VaultLibrarian | 1,540 | 3 | 0 | 2025-11-07 | Coverage: searchContent, searchDirectory (incl. song lyric hunt) |
| MemoryManager | 770 | 1 | 0 | 2025-11-07 | Deep Research Sprint session tracking |
| Multi-Tool Workflows | 1,270 | 9 | 0 | 2025-11-07 | Nine positives chaining 2-4 tools |
| Tool Discovery | 254 | 1 | 0 | 2025-11-07 | Spark board example exercises `get_tools` |
| Error Recovery | 847 | 1 | 0 | 2025-11-07 | Gratitude note example recovers from missing file |
| Clarification | 593 | 2 | 2 | 2025-11-07 | Iceland checklist + song verse (good); two undesirable skip variants |
| **Total** | **9,509** | **9** | **3** | 2025-11-07 | Three batches logged (9 desirable + 3 undesirable) |

#### Batch Notes Log
- *2025-11-07 (Batch 1)*: Added three desirable multi-tool examples (Weekly Retro move, Neon Whale journal entry, NaNo spark board build) plus one undesirable variant for the Weekly Retro scenario. Schemas referenced: `.obsidian/plugins/claudesidian-mcp/src/agents/vaultLibrarian/modes/searchContentMode.ts`, `.../searchDirectoryMode.ts`, `.../vaultManager/modes/moveNoteMode.ts`, `.../createFolderMode.ts`, `.../contentManager/modes/createContentMode.ts`, `.../appendContentMode.ts`, `connector.ts#get_tools`. Workspace IDs: `ws_1730996154321_mt8f6d2qx`, `ws_1730998100123_d7p5s9h2q`, `ws_1730998850001_f4n7s8d3k`, `ws_1730999288777_c4k6n1p0r`. Completed 3 positives + 1 negative; ready for Batch 2.
- *2025-11-07 (Batch 2)*: Logged clarification + reminder update, gratitude error-recovery, and book-club duplicate scenarios, plus an undesirable clarification skip. Schemas checked: `vaultManager/listDirectoryMode.ts`, `contentManager/appendContentMode.ts`, `contentManager/createContentMode.ts`, `vaultManager/duplicateNoteMode.ts`. Workspace IDs: `ws_1731000456789_b4n6p1t2y`, `ws_1731000999000_kv8m2s4dj`, `ws_1731001488000_s4k9n1t6c`, `ws_1731001855000_j6r8t3c1p`. Batch totals: +3 desirable, +1 undesirable (clarification miss).
- *2025-11-07 (Batch 3)*: Added memoryManager_createSession + openNote flow, a lyric-search clarification example, and a festival pitch re-org (folder creation → move → append), plus an undesirable lyric update that skipped discovery. Schemas reviewed: `memoryManager/modes/sessions/CreateSessionMode.ts`, `vaultManager/modes/openNoteMode.ts`, `vaultLibrarian/modes/searchContentMode.ts`, `vaultManager/modes/listDirectoryMode.ts`, `contentManager/modes/appendContentMode.ts`, `vaultManager/modes/createFolderMode.ts`, `vaultManager/modes/moveNoteMode.ts`. Workspace IDs: `ws_1731002400789_r5m8k2h1j`, `ws_1731002954321_q8r5m6n2t`, `ws_1731003384567_m9p1t4r6u`, `ws_1731003555000_w8r3t1c6m`. Batch totals: +3 desirable, +1 undesirable (failed clarification/search).

### 6.8 Context Priming & Schema Verification Checklist

LLMs are stateless: they only see the prompt/completion we generate. Every example must therefore embed all knowledge about file locations, workspace metadata, and tool schemas inside the conversation itself. Use this checklist before writing a single token of JSONL.

#### 1. Review the actual tool schema
- Open the real mode file under `.obsidian/plugins/claudesidian-mcp/src/agents/{agent}/modes/{mode}Mode.ts`.
- Copy the current `schema`, `execute`, and sample response structure into your scratchpad.
- Note required/optional fields, valid enums, default values, and response shape (e.g., `data` payload, timestamps, workspaceContext).
- No synthetic example may be emitted until this schema review is complete for every tool used in that conversation.

#### 2. Prime location + workspace metadata
- Record how the assistant knows the target file/folder (user stated path, prior search, or directory listing). If the path is not explicitly given, the conversation must include a discovery step (`vaultLibrarian_searchDirectory`, `vaultManager_listDirectory`, etc.) before editing.
- Every conversation must mention the active workspace and supply the real `workspaceId` (e.g., `ws_1730995200456_gx7p01v`) in the `context` block. If you also reference a friendly label such as "Claudesidian Core", make sure the actual ID is captured in context plus the tracker note.
- Include realistic `sessionDescription`, `toolContext`, `primaryGoal`, and `subgoal` strings that mirror the user intent and reference any discovery steps already performed.

#### 3. Argument formatting contract
- Always serialize `arguments` with the `context` object first, followed by tool-specific parameters. This ordering makes the supervision signal explicit for downstream models.
- The `context` object must include: `sessionId`, `workspaceId`, `sessionDescription`, `sessionMemory`, `toolContext`, `primaryGoal`, and `subgoal`. Update `sessionMemory` after each tool call so subsequent calls reflect accumulated knowledge.
- Mirror the schema exactly: parameter names, nesting, arrays, and enums must match the mode definition. Do not invent helper fields or shorten names.

#### 4. Enforce multi-turn completeness
- Default expectation: each positive example contains **at least two assistant actions** (clarifying question, `get_tools`, search/listing, write action, etc.) plus their corresponding tool results and a final natural-language summary.
- Show how earlier results inform later steps (e.g., the search output is referenced in the edit call, or a clarifying answer updates `sessionMemory`).
- For single-tool scenarios (rare), justify the missing discovery step by embedding the path and workspace evidence directly in the user message and tracker note.

#### 5. Annotate schema & workspace provenance
- In the tracker batch note, include the tool schemas inspected (file path + git commit, if available) and the workspace ID used for each example.
- If any schema changes mid-run, regenerate affected examples and re-review before resuming generation.

#### 6. Session + workspace identifier formats
- Confirm the current ID generators before every batch. As of 2025-11-07:
  - `SessionService.generateSessionId` (`.obsidian/plugins/claudesidian-mcp/src/services/session/SessionService.ts:148`) emits `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`.
  - `WorkspaceService.createWorkspace` (`.obsidian/plugins/claudesidian-mcp/src/services/WorkspaceService.ts:130`) emits `ws_${Date.now()}_${Math.random().toString(36).substr(2, 9)}` when no ID is provided.
- Sample realistic values: `session_1730995200451_k9d4b28`, `ws_1730995200456_gx7p01v`. Use fresh timestamps per example so IDs do not repeat.
- If the CLAUDesidian release switches to a different scheme (e.g., `s-2025-11-07T15_42_11Z` or UUID-backed `ws-xxxx`), update the samples immediately and log the new format in the tracker notes before generating more data.

#### 7. KTO label enforcement
- Every JSONL object must include `"label": "desirable"` or `"label": "undesirable"` to remain compatible with the planned KTO fine-tuning pipeline.
- Positives (supervised SFT data) should default to `"desirable"`. Paired negatives inherit `"undesirable"` and must be placed directly after the related positive batch.
- The validator script (`tools/validate_syngen.py`) enforces label presence and allowed values—do not skip it before committing new data.

Following this checklist prevents impossible actions (e.g., editing a file whose location was never established) and keeps every example grounded in the real claudesidian tool surface area.

---

## 7. Advanced Techniques: KTO & Contrastive Learning

### 7.1 Kahneman-Tversky Optimization (KTO)

#### Why KTO for Claudesidian-MCP?

1. **Binary Nature of Tool Calling**: Tools either work correctly or fail. This maps perfectly to KTO's "desirable/undesirable" framework.

2. **Data Efficiency**: We don't need preference pairs (A vs B). Just need to label examples as good or bad.

3. **Loss Aversion Learning**: KTO models learn more from mistakes (undesirable) than successes (desirable), which is ideal for tool calling where errors are costly.

4. **Performance**: KTO matches DPO+SFT performance with simpler data requirements.

#### KTO Training Process

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Supervised Fine-Tuning (SFT)                      │
│  - Train on positive examples only                           │
│  - 9,509 examples, 3 epochs                                  │
│  - Learn basic tool calling patterns                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2: KTO Alignment                                      │
│  - Train on positive + negative examples                     │
│  - 15,213 examples (9.5k positive, 5.7k negative)            │
│  - Learn to avoid errors via loss aversion                   │
│  - Optimize utility function (Prospect Theory)               │
└─────────────────────────────────────────────────────────────┘
```

#### KTO Loss Function (Simplified)

```
KTO_loss = λ_D * loss_desirable(x_desirable) + λ_U * loss_undesirable(x_undesirable)

where:
- loss_desirable: Encourages generating desirable outputs
- loss_undesirable: Penalizes undesirable outputs (weighted higher due to loss aversion)
- λ_D, λ_U: Hyperparameters (typically λ_U > λ_D)
```

### 7.2 Negative Example Generation Strategy

#### 5 Error Types to Generate

1. **Wrong Tool (30% of negatives)**
   - Correct intent, wrong tool selected
   - Example: Using `openNote` instead of `createContent`

2. **Missing Parameters (25% of negatives)**
   - Correct tool, missing required params
   - Example: `createContent` without `content` field

3. **Wrong Parameter Values (20% of negatives)**
   - Correct tool and params, but wrong values
   - Example: File path with invalid characters or format

4. **Wrong Sequence (12% of negatives)**
   - Multi-tool workflows in wrong order
   - Example: Opening file before creating it
   - Missing tool result processing (jumping to next tool without seeing result)

5. **Hallucinated Elements (8% of negatives)**
   - Non-existent tools, parameters, or modes
   - Example: Made-up tool like `batchDelete`

6. **Failed Error Recovery (10% of negatives)**
   - Gets error message but ignores it
   - Retries same wrong parameters
   - Gives up instead of trying alternative

7. **Over-clarification (5% of negatives)**
   - Asks unnecessary questions for clear requests
   - Requests too much information
   - Example: User says "create daily note for today" and model asks for date

#### Automated Mutation Rules

```python
def generate_negative_examples(positive_example):
    negatives = []

    # Type 1: Swap to wrong tool (same agent)
    wrong_tool = get_random_tool_from_same_agent(positive_example['tool'])
    negatives.append(mutate_tool_name(positive_example, wrong_tool))

    # Type 2: Remove random required parameter
    negatives.append(remove_parameter(positive_example, random_required_param()))

    # Type 3: Corrupt parameter value
    negatives.append(corrupt_parameter_value(positive_example, 'filePath'))

    # Type 4: Reverse tool order (if multi-tool)
    if is_multi_tool(positive_example):
        negatives.append(reverse_tool_order(positive_example))

    # Type 5: Add hallucinated parameter
    negatives.append(add_fake_parameter(positive_example))

    # Type 6: Remove tool result (incomplete conversation)
    negatives.append(remove_tool_result(positive_example))

    # Type 7: Failed error recovery (if example has error handling)
    if has_error_recovery(positive_example):
        negatives.append(remove_error_correction(positive_example))

    # Type 8: Over-clarification (for clear requests)
    if is_clear_intent(positive_example):
        negatives.append(add_unnecessary_clarification(positive_example))

    return negatives
```

### 7.3 Contrastive Learning Applications

While KTO is the primary technique, we can also apply **contrastive learning** concepts:

#### Positive Pairs (Augmented Examples)
- Same semantic intent, different phrasing
- Example: "Create a note" vs "Make a new file"

#### Negative Pairs (Undesirable Examples)
- Similar tool calls, but incorrect
- Helps model distinguish subtle differences

#### Triplet Loss Approach (Optional)
```
anchor: Correct tool call
positive: Paraphrased version of correct tool call
negative: Incorrect tool call (wrong tool or params)

Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

This could be used in a **two-stage training**:
1. **Stage 1**: Supervised fine-tuning (basic tool calling)
2. **Stage 2**: Contrastive fine-tuning (distinguish correct from incorrect)

---

## 8. Quality Assurance & Validation

### 8.1 Data Quality Metrics

Track these metrics during generation:

| Metric | Target | Importance |
|--------|--------|------------|
| Tool name validity | 100% | Critical |
| Parameter completeness | >98% | Critical |
| Context structure validity | 100% | Critical |
| **Multi-turn completeness** | **100%** | **Critical** |
| **Tool-calling only focus** | **100%** | **Critical** |
| **Paired negative coverage** | **60%** | **Critical** |
| File path realism | >95% | High |
| Query diversity (unique) | >80% | High |
| Error recovery scenarios | 10% | High |
| Clarification scenarios | 7% | High |
| Multi-tool sequence logic | >90% | High |

### 8.2 Validation Stages

#### Stage 1: Automated Validation
```python
def validate_dataset(examples):
    errors = []

    for ex in examples:
        # Check tool name
        if not is_valid_tool(ex['tool_name']):
            errors.append(f"Invalid tool: {ex['tool_name']}")

        # Check required parameters
        required = get_required_params(ex['tool_name'])
        if not all(p in ex['arguments'] for p in required):
            errors.append(f"Missing params in {ex['id']}")

        # Check context structure
        if not validate_context(ex['arguments']['context']):
            errors.append(f"Invalid context in {ex['id']}")

        # NEW: Check multi-turn completeness
        if not has_tool_result_after_call(ex):
            errors.append(f"Missing tool result in {ex['id']}")

        if not has_final_response(ex):
            errors.append(f"Missing final assistant response in {ex['id']}")

    return errors
```

#### Stage 2: Manual Review (Sample)
- Review 100 random examples (2% of dataset)
- Check for:
  - Natural query phrasing
  - Realistic file paths and content
  - Appropriate tool selection
  - Coherent multi-tool workflows
  - **Complete tool execution flows**

#### Stage 3: Model Evaluation (Before Training)
- Test teacher model on generated examples
- Accuracy should be >99% on generated data
- Verify all examples have complete multi-turn flows (user → tool call → tool result → assistant)
- If lower, indicates data quality issues

### 8.3 Test Suite Design

Create a **held-out test set** with specific scenarios:

| Test Category | Count | Description |
|---------------|-------|-------------|
| Single-tool (simple) | 228 | Basic operations with complete tool flow |
| Single-tool (ambiguous) | 152 | Requires context to disambiguate |
| Multi-tool (2-step) | 152 | Sequential tool chains with result processing |
| Multi-tool (3-step) | 76 | Complex workflows with context accumulation |
| Tool discovery | 38 | Using get_tools before execution |
| Error scenarios | 114 | Should recognize tool limitations |
| **Total Test Examples** | **760** | Covers all capabilities |

### 8.4 Continuous Improvement Loop

```
Generate Batch (1540 examples)
        │
        ▼
Validate Automatically
        │
        ▼
   Pass? ──No─→ Analyze Failures ─→ Update Generation Prompts
     │                                        │
    Yes                                       │
     │                                        │
     └────────────────────────────────────────┘
     │
     ▼
Manual Review (Sample)
     │
     ▼
Add to Final Dataset
```

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Week 1)
**Goal**: Set up infrastructure and generate first 1,540 examples

#### Tasks
- [ ] Extract all tool schemas from claudesidian-mcp codebase
- [ ] Create schema JSON files for each of 38 tools (including `get_tools`)
- [ ] Set up teacher model API (choose capable LLM for generation)
- [ ] Write generation prompt templates for Unsloth universal format
- [ ] Implement JSONL formatter for prompt/completion pairs with special tokens
- [ ] Generate 1,540 ContentManager examples (tool-calling only)
- [ ] Include 154 error recovery examples in first batch
- [ ] Include 108 clarification examples in first batch
- [ ] Validate all examples: complete multi-turn, no general conversation
- [ ] Manual review of 77 samples for format compliance

#### Deliverables
- Tool schema library (JSON)
- Generation scripts (Python) for Unsloth format
- 1,540 validated training examples (Unsloth universal format)
- Error recovery examples (154)
- Clarification examples (108)

### Phase 2: Scale Up (Week 2)
**Goal**: Generate full positive dataset (9,509 examples)

#### Tasks
- [ ] Generate 2,310 ContentManager examples (tool-calling only, Unsloth format)
- [ ] Generate 1,925 VaultManager examples (tool-calling only, Unsloth format)
- [ ] Generate 1,540 VaultLibrarian examples (tool-calling only, Unsloth format)
- [ ] Generate 770 MemoryManager examples (tool-calling only, Unsloth format)
- [ ] Generate 1,270 multi-tool workflow examples with complete chains
- [ ] Generate 254 tool discovery examples (get_tools → execution)
- [ ] Generate 847 error recovery examples (10% of total)
- [ ] Generate 593 clarification examples (7% of total)
- [ ] Implement diversity augmentation (paraphrasing, parameter variation)
- [ ] Run automated validation: format, tool results, no general conversation
- [ ] Manual review of 154 random samples for quality

#### Deliverables
- 9,509 positive training examples (Unsloth format, tool-calling only)
- Includes error recovery (847) and clarification (593) scenarios
- Validation report: complete tool flows, format compliance, no general conversation
- Quality metrics on conversation completeness and pairing coverage

### Phase 3: Negative Examples (Week 3)
**Goal**: Generate 5,704 KTO negative examples

#### Tasks
- [ ] Implement KTO paired negative generator (8 mutation types)
- [ ] Generate 1,386 paired negatives for ContentManager
- [ ] Generate 1,155 paired negatives for VaultManager
- [ ] Generate 924 paired negatives for VaultLibrarian
- [ ] Generate 462 paired negatives for MemoryManager
- [ ] Generate 762 paired negatives for multi-tool workflows
- [ ] Generate 508 paired negatives for error recovery (failed recovery)
- [ ] Generate 355 paired negatives for clarification (over/under clarification)
- [ ] Validate: each negative paired with corresponding positive
- [ ] Label all examples (desirable/undesirable)
- [ ] Ensure ~60% of positives have paired negatives

#### Deliverables
- 5,704 paired negative training examples
- 15,213 total examples with KTO labels (9,509 positive + 5,704 negative)
- Pairing manifest showing which negatives match which positives

### Phase 4: Dataset Finalization (Week 4)
**Goal**: Split, validate, and package dataset

#### Tasks
- [ ] Create train/val/test splits (80/10/10)
- [ ] Generate dataset statistics and distribution report
- [ ] Create held-out test suite (760 examples)
  - Include error recovery tests (76)
  - Include clarification tests (53)
  - Include paired positive/negative tests (304)
- [ ] Test teacher model on generated data (should get >99%)
- [ ] Package dataset in Unsloth universal format:
  - Training set JSONL (12,170 examples)
  - Validation set JSONL (1,522 examples)
  - Test set JSONL (1,521 examples)
  - KTO labels file (pairing manifest)
- [ ] Write dataset documentation:
  - README with Unsloth usage instructions
  - Dataset card with statistics
  - Error recovery and clarification guidelines

#### Deliverables
- Final dataset (15,213 examples + 760 test) in Unsloth universal format
- KTO pairing manifest (5,704 pairs)
- Dataset card highlighting:
  - Multi-turn completeness
  - Tool-calling only focus
  - Error recovery capabilities
  - Clarification handling
  - Paired negative examples for KTO

### Phase 5: Pilot Fine-Tuning (Week 5)
**Goal**: Test dataset quality with actual fine-tuning

#### Tasks
- [ ] Select base model with tool-calling support
- [ ] Set up Unsloth fine-tuning environment (local or cloud)
- [ ] Run SFT baseline (positive examples only)
  - Verify multi-turn handling
  - Verify error recovery
  - Verify clarification handling
- [ ] Run KTO training with paired examples
  - Use pairing manifest
  - Weight loss aversion appropriately
- [ ] Evaluate on test suite (760 examples):
  - Tool selection accuracy
  - Error recovery rate
  - Clarification appropriateness
  - Paired positive/negative discrimination
- [ ] Compare to teacher model baseline
- [ ] Analyze failure modes:
  - Incomplete tool flows
  - Ignoring errors
  - Over/under clarification

#### Deliverables
- Fine-tuned model checkpoint (Unsloth format)
- Evaluation report comparing SFT vs KTO
- Error recovery success rate metrics
- Clarification appropriateness metrics
- Paired example discrimination analysis
- Failure mode analysis with recommendations

### Phase 6: Iteration (Week 6+)
**Goal**: Improve based on pilot results

#### Tasks
- [ ] Identify weak areas from evaluation
- [ ] Generate additional targeted examples for weak areas
- [ ] Re-run fine-tuning with augmented dataset
- [ ] Repeat evaluation
- [ ] Document final performance
- [ ] Create deployment guide for local use

#### Deliverables
- Improved dataset (v2)
- Production-ready model
- Deployment documentation

---

## 10. References & Resources

### Research Papers
1. **KTO: Model Alignment as Prospect Theoretic Optimization**
   - Ethayarajh et al. (2024)
   - https://arxiv.org/abs/2402.01306
   - Key insight: Loss aversion for LLM alignment

2. **Self-Instruct: Aligning Language Models with Self-Generated Instructions**
   - Wang et al. (2023)
   - Foundation for synthetic data generation from teacher models

3. **DataDreamer: A Tool for Synthetic Data Generation and Reproducible LLM Workflows**
   - Patel et al. (2024)
   - https://arxiv.org/abs/2402.10379
   - Open source toolkit

### Tools & Frameworks
1. **Meta Synthetic Data Kit**
   - https://github.com/meta-llama/synthetic-data-kit
   - Generate reasoning traces and QA pairs

2. **Gretel Navigator**
   - https://gretel.ai/blog/how-to-create-high-quality-synthetic-data-for-fine-tuning-llms
   - Commercial synthetic data platform

3. **Axolotl** (Fine-tuning framework)
   - https://github.com/OpenAccess-AI-Collective/axolotl
   - Supports KTO, DPO, and custom formats

4. **MLX** (Apple Silicon fine-tuning)
   - https://github.com/ml-explore/mlx-examples/tree/main/llms
   - Efficient local fine-tuning on Mac

5. **Unsloth** (Fast fine-tuning)
   - https://github.com/unslothai/unsloth
   - 2x faster than Hugging Face, supports QLoRA

### Model Context Protocol (MCP)
1. **Anthropic MCP Documentation**
   - https://modelcontextprotocol.io/
   - Official protocol specification

2. **Claudesidian-MCP Repository**
   - https://github.com/ProfSynapse/claudesidian-mcp
   - Source code for tool schemas

### Fine-Tuning Guides
1. **Mistral Fine-Tuning Guide**
   - https://docs.mistral.ai/capabilities/finetuning/
   - Function calling fine-tuning examples

2. **Finetuning Toolshim Models for Tool Calling**
   - https://block.github.io/goose/blog/2025/04/11/finetuning-toolshim/
   - Practical guide from Block (2025)

3. **OpenAI Function Calling Guide**
   - https://platform.openai.com/docs/guides/function-calling
   - Best practices for tool calling datasets

---

## Appendix A: Tool Schema Reference

*[Keep all tool schemas from original document]*

---

## Appendix B: Context Schema Reference

*[Keep context schema from original document]*

---

## Appendix C: Sample Generation Script (Unsloth Format)

**Note**: This script generates data in Unsloth universal format with tool-calling focus.

```python
import json
from typing import List, Dict
# Import your chosen teacher model API

# Tool schema library
TOOL_SCHEMAS = {
    "contentManager_createContent": {
        "name": "contentManager_createContent",
        "description": "Create a new file with content in the vault",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {"type": "string"},
                "content": {"type": "string"},
                "context": {"type": "object"}
            },
            "required": ["filePath", "content", "context"]
        }
    },
    # ... more tools
}

def generate_example(tool_name: str, tool_schema: Dict, include_error: bool = False) -> Dict:
    """Generate a single training example in Unsloth format."""

    error_instruction = "Include an error that gets corrected." if include_error else ""

    prompt = f"""Generate a TOOL-CALLING ONLY training example for: {tool_name}

Tool Schema: {json.dumps(tool_schema, indent=2)}

{error_instruction}

FORMAT (Unsloth universal):
{{
  "prompt": "<|system|>You are an AI assistant with access to a workspace and a set of tools organized into bundles. Your job is to accomplish the user's goals by selecting and using the appropriate tools.\\n\\nAvailable tool bundles:\\n- contentManager: Read, create, edit, append, and replace content in workspace files\\n- vaultManager: Create, move, duplicate, rename, and delete files and folders\\n- vaultLibrarian: Search for files, content, and information across the workspace\\n- memoryManager: Manage sessions, workspaces, and context snapshots\\n- agentManager: Execute custom AI prompts, generate images, and run batch operations\\n\\nYou only respond with tool calls to accomplish tasks.</s>\\n<|user|>{{user_query}}</s>\\n<|assistant|>",
  "completion": "tool_call: {tool_name}\\narguments: {{json}}\\n</s>\\n<|tool|>Result: {{result}}</s>\\n<|assistant|>{{response}}</s>"
}}

CRITICAL REQUIREMENTS:
- TOOL-CALLING ONLY (no general conversation)
- Complete multi-turn flow in single prompt/completion pair
- Use special tokens: <|system|>, <|user|>, <|assistant|>, <|tool|>, </s>
- Tool result must be realistic JSON
- Final assistant response summarizes action taken
{("- Include error recovery: wrong params → error → corrected params → success" if include_error else "")}

Output as JSON."""

    response = your_llm_client.generate(
        messages=[
            {"role": "system", "content": "You generate tool-calling training data in Unsloth format."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )

    return json.loads(response.content)

def generate_clarification_example() -> Dict:
    """Generate example where model asks for clarification."""

    prompt = """Generate a tool-calling example where the user request is AMBIGUOUS.

The model should ask 1-3 clarifying questions before proceeding.

FORMAT:
{
  "prompt": "<|system|>You are an AI assistant with access to a workspace and a set of tools organized into bundles. Your job is to accomplish the user's goals by selecting and using the appropriate tools.\\n\\nAvailable tool bundles:\\n- contentManager: Read, create, edit, append, and replace content in workspace files\\n- vaultManager: Create, move, duplicate, rename, and delete files and folders\\n- vaultLibrarian: Search for files, content, and information across the workspace\\n- memoryManager: Manage sessions, workspaces, and context snapshots\\n- agentManager: Execute custom AI prompts, generate images, and run batch operations\\n\\nYou may ask clarifying questions when the user's request is ambiguous, then use tools to accomplish the task.</s>\\n<|user|>{ambiguous_request}</s>\\n<|assistant|>",
  "completion": "I need more information:\\n1. {question1}\\n2. {question2}</s>\\n<|user|>{user_clarification}</s>\\n<|assistant|>tool_call: {tool}\\narguments: {...}\\n</s>\\n<|tool|>Result: {...}</s>\\n<|assistant|>{response}</s>"
}

Make the user request genuinely ambiguous (missing file name, unclear content, etc.)."""

    response = your_llm_client.generate(
        messages=[{"role": "system", "content": "Generate clarification examples."},
                  {"role": "user", "content": prompt}],
        temperature=0.8
    )

    return json.loads(response.content)

def generate_paired_negative(positive_example: Dict, error_type: str) -> Dict:
    """Generate a negative example paired with a positive one."""

    mutations = {
        "wrong_tool": "Change to wrong tool for the task",
        "missing_param": "Remove a required parameter",
        "incomplete": "Remove tool result and final response",
        "ignore_error": "Include error but don't correct it",
        "over_clarify": "Ask unnecessary questions for clear request"
    }

    prompt = f"""Given this POSITIVE example:
{json.dumps(positive_example, indent=2)}

Create a NEGATIVE version by: {mutations[error_type]}

Keep the same scenario/user request, but make the model response incorrect.
Output in same Unsloth format with "label": "undesirable"."""

    response = your_llm_client.generate(
        messages=[{"role": "system", "content": "Generate paired negative examples."},
                  {"role": "user", "content": prompt}],
        temperature=0.3  # Lower temp for consistent mutations
    )

    negative = json.loads(response.content)
    negative["label"] = "undesirable"
    negative["paired_with"] = positive_example.get("id", "unknown")

    return negative

def generate_dataset(num_examples: int) -> List[Dict]:
    """Generate complete dataset with error recovery and clarification."""

    dataset = []

    # Calculate distribution
    num_clarification = int(num_examples * 0.07)
    num_error_recovery = int(num_examples * 0.10)
    num_standard = num_examples - num_clarification - num_error_recovery

    # Generate standard examples
    for tool_name, tool_schema in TOOL_SCHEMAS.items():
        tool_count = num_standard // len(TOOL_SCHEMAS)
        for _ in range(tool_count):
            example = generate_example(tool_name, tool_schema, include_error=False)
            example["label"] = "desirable"
            example["id"] = f"{tool_name}_{len(dataset)}"
            dataset.append(example)

    # Generate error recovery examples
    for _ in range(num_error_recovery):
        tool_name = random.choice(list(TOOL_SCHEMAS.keys()))
        example = generate_example(tool_name, TOOL_SCHEMAS[tool_name], include_error=True)
        example["label"] = "desirable"
        example["id"] = f"{tool_name}_error_{len(dataset)}"
        dataset.append(example)

    # Generate clarification examples
    for _ in range(num_clarification):
        example = generate_clarification_example()
        example["label"] = "desirable"
        example["id"] = f"clarify_{len(dataset)}"
        dataset.append(example)

    return dataset

def generate_kto_pairs(positive_examples: List[Dict], pairing_ratio: float = 0.6) -> List[Dict]:
    """Generate paired negative examples for KTO training."""

    error_types = ["wrong_tool", "missing_param", "incomplete", "ignore_error", "over_clarify"]
    negatives = []

    # Select examples to pair (60% of positives get paired negatives)
    examples_to_pair = random.sample(positive_examples, int(len(positive_examples) * pairing_ratio))

    for pos_example in examples_to_pair:
        # Generate 1-2 negatives per positive
        num_negatives = random.randint(1, 2)
        for _ in range(num_negatives):
            error_type = random.choice(error_types)
            neg_example = generate_paired_negative(pos_example, error_type)
            negatives.append(neg_example)

    return negatives

if __name__ == "__main__":
    print("Generating positive examples...")
    positives = generate_dataset(6175)

    print("Generating paired negative examples...")
    negatives = generate_kto_pairs(positives, pairing_ratio=0.6)

    # Combine and save
    all_examples = positives + negatives
    random.shuffle(all_examples)

    # Save dataset
    with open("claudesidian_unsloth_train.jsonl", 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    # Save pairing manifest
    pairs = [(p["id"], n["paired_with"]) for n in negatives if "paired_with" in n]
    with open("kto_pairs.json", 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"Generated {len(positives)} positive examples")
    print(f"Generated {len(negatives)} paired negative examples")
    print(f"Total dataset: {len(all_examples)} examples")
```

---

---

## Appendix D: Unsloth Fine-Tuning Configuration

### Recommended Unsloth Settings

```python
from unsloth import FastLanguageModel

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-v0.3",  # or your chosen base model
    max_seq_length=4096,  # Accommodate multi-turn conversations
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # For efficiency
)

# Add special tokens for tool calling
special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "</s>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))

# Configure for training
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    max_steps=1500,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    output_dir="outputs",
)
```

### KTO Training Configuration

```python
# For KTO training with paired examples
from unsloth import KTOTrainer

kto_trainer = KTOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    beta=0.1,  # KTO beta parameter (loss aversion weight)
    desirable_weight=1.0,
    undesirable_weight=1.0,
)

kto_trainer.train()
```

---

## End of Document

**Document Status**: ✅ Ready for implementation

**Version**: 3.0 (Major format change: Migrated from prompt/completion to ChatML format for Unsloth compatibility)

**Key Takeaways**:

1. **Multi-Turn Completeness**: Every tool call must be followed by a result and proper handling
2. **Tool-Calling Only**: No general conversation - focus exclusively on tool use
3. **Error Recovery**: Models must learn to read errors and correct mistakes
4. **Clarification**: Ask questions when needed, but not too often
5. **KTO Paired Examples**: Each positive scenario should have paired negative variations
6. **Unsloth Format**: Simple prompt/completion pairs with special tokens for efficient training

This dataset will train models that don't just call tools, but actually **use them effectively** with proper error handling and clarification when needed.
