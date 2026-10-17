# Cortex Memory

A lightweight, self-hosted long-term memory server for AI agents and coding assistants, powered by an embedded **SurrealDB** engine (`surrealkv`).

Supports tri-temporal fact versioning, automatic reconciliation of conflicting behavioral instructions, hybrid search (vectors + BM25), and standard **Model Context Protocol (MCP)** integration.

---

## Features

- **State-Aware Memory**: Ingests observations with automatic superseding of outdated facts.
- **Instruction Reconciliation**: Detects and supersedes conflicting behavioral rules.
- **Hybrid Retrieval**: Combines vector similarity and BM25 ranking (RRF) with 1-hop graph relation expansion.
- **MCP Support**: Ready-to-use Model Context Protocol server via stdio or HTTP.
- **Embedded & Local**: Self-contained on local disk without external database services.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/gjjjhhhb/cortex-memory.git
cd cortex-memory
uv sync --extra dev
```

### 2. Configuration

Set your provider credentials in `.env`:

```env
LLM_API_KEY="your_llm_api_key_here"
LLM_BASE_URL="https://api.your-provider.com/v1"
LLM_MODEL="your-model-name"

EMBEDDER_API_KEY="your_embedder_api_key_here"
EMBEDDER_BASE_URL="https://api.your-provider.com/v1"
EMBEDDER_MODEL="your-embedding-model-name"
EMBEDDER_DIMENSION=1536
```

Server and provider settings are configured in `config.toml` (or overridden via `CORTEX_*` environment variables):

```toml
[server]
host = "127.0.0.1"
port = 8080
api_key = "cortex-local-dev-key"
default_context = "dev"

[database]
url = "surrealkv://./data/cortex.db"
namespace = "cortex"

[llm]
provider = "openai"
base_url = "env:LLM_BASE_URL"
api_key = "env:LLM_API_KEY"
model = "env:LLM_MODEL"
temperature = 0.0

[embedding]
provider = "openai"
base_url = "env:EMBEDDER_BASE_URL"
api_key = "env:EMBEDDER_API_KEY"
model = "env:EMBEDDER_MODEL"
dimension = 1536
cache_size = 10000
```

### 3. Run the Server

**HTTP API:**
```bash
uv run cortex serve
```

**MCP Server (stdio):**
```bash
uv run cortex mcp --context dev
```

---

## IDE Integration (MCP)

Add to your editor's MCP settings:

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
# Store observation
uv run cortex remember "User prefers Python and Rust"

# Semantic search
uv run cortex recall "What programming languages does the user prefer?"

# Context prompt generation
uv run cortex context "Technical stack"

# Forget memories
uv run cortex forget "outdated preference"
```

---

## Testing

```bash
uv run pytest
```
