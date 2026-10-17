# Cortex Memory

A lightweight, self-hosted long-term memory server for AI agents and coding assistants (Cursor, Claude Desktop, Antigravity), powered by an embedded **SurrealDB** engine (`surrealkv`).

Supports tri-temporal fact versioning, automatic reconciliation of conflicting behavioral instructions, hybrid search (vectors + BM25), and the **Model Context Protocol (MCP)** out of the box.

---

## Features

- **State-Aware Memory**: Ingests observations with automatic superseding of outdated facts (e.g., job role updates or moving cities).
- **Instruction Reconciliation**: Detects and supersedes conflicting behavioral rules (e.g., *"Respond concisely"* $\rightarrow$ *"Respond with detailed code examples"*).
- **Hybrid Retrieval**: Combines dense vector similarity and lexical BM25 matching (RRF) with 1-hop knowledge graph relation expansion.
- **Native MCP Support**: Ready-to-use Model Context Protocol server via stdio or HTTP.
- **Embedded & Local**: Runs self-contained on local disk using SurrealDB's embedded KV engine without needing external database services.

---

## Quickstart

### 1. Installation

```bash
uv sync --extra dev
```

### 2. Configuration

Set your provider API keys in `.env`:

```env
OPENAI_API_KEY="sk-..."
# or
DEEPSEEK_API_KEY="sk-..."
GEMINI_API_KEY="AIzaSy..."
```

Server and provider settings are configured in `config.toml` (and can be overridden via `CORTEX_*` environment variables):

```toml
[server]
host = "127.0.0.1"
port = 8080
api_key = "cortex-local-dev-key"

[database]
url = "surrealkv://./data/cortex.db"
namespace = "cortex"

[llm]
provider = "openai"
base_url = "https://api.deepseek.com/v1"
api_key = "env:DEEPSEEK_API_KEY"
model = "deepseek-chat"

[embedding]
provider = "openai"
base_url = "https://generativelanguage.googleapis.com/v1beta"
api_key = "env:GEMINI_API_KEY"
model = "gemini-embedding-001"
dimension = 768
```

### 3. Run the Server

**HTTP API server:**
```bash
uv run cortex serve
```

**MCP server (stdio for IDEs):**
```bash
uv run cortex mcp --context dev
```

---

## Connecting to Cursor & Claude Desktop

Add to your MCP configuration (`claude_desktop_config.json` or `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "cortex": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/cortex-py",
        "cortex",
        "mcp",
        "--context",
        "dev"
      ]
    }
  }
}
```

---

## CLI Commands

```bash
# Ingest an observation
uv run cortex remember "User writes Python and Rust"

# Semantic search
uv run cortex recall "What programming languages does the user know?"

# Assemble prompt memory block for LLMs
uv run cortex context "Technical stack"

# Forget outdated memories
uv run cortex forget "outdated preference"
```

---

## Testing

```bash
uv run pytest
```
