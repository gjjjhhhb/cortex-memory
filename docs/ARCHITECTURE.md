# Cortex Architecture & Developer Guide

This document describes the architectural layout, data flow, and design patterns implemented in **Cortex**.

---

## 1. Clean Architecture Layers

Cortex follows a strict **Clean Architecture / Hexagonal Architecture** model. Dependencies flow inwards: Transport depends on Engine and Runtime, Engine depends on Infrastructure interfaces and Domain, and Domain has zero external dependencies.

```
+-------------------------------------------------------------------+
|                           Transport Layer                         |
|   cli.py (Typer)  |  http.py (FastAPI)  |  mcp.py (JSON-RPC)     |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                           Runtime Container                       |
|   runtime.py (CortexRuntime / DI Container / AppState)           |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                           Engine Layer (Services)                 |
|   ingestion.py   |  retrieval.py  |  cognitive.py  |  graph.py    |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                       Infrastructure Layer (Ports/Adapters)       |
|   store.py (SurrealDB) | llm.py (Providers) | embedding.py (LRU)  |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                           Domain Layer (Core)                     |
|   models.py      |  enums.py      |  scoring.py    | formatting.py|
+-------------------------------------------------------------------+
```

---

## 2. Layer Responsibilities

### Domain Layer (`cortex/domain/`)
- Contains purely synchronous business logic and data structures.
- Has no database connections, network calls, or framework dependencies.
- **`enums.py`**: Enumerations for operation types, search modes, memory categories.
- **`models.py`**: Pydantic models supporting bidirectional conversion between Python snake_case and SDK camelCase.
- **`scoring.py`**: Mathematical calculations:
  - Vector cosine similarity.
  - BM25 token matching with frequency weighting.
  - Reciprocal Rank Fusion (RRF) for combining vector and lexical rankings.
  - Tri-temporal time-decay weighting.
- **`formatting.py`**: Normalization of entity names and Markdown context generation without section duplicate facts.
- **`prompts.py`**: System extraction and synthesis prompts.

### Infrastructure Layer (`cortex/infra/`)
- Encapsulates external I/O and third-party APIs.
- **`store.py`**: SurrealDB client encapsulation. All SurrealQL queries (`SELECT`, `CREATE`, `UPDATE`, `DELETE`) are isolated inside `CortexStore`.
- **`schema.py`**: Idempotent DDL definitions for tables (`attribute`, `relation`, `instruction`, `memory_chunk`, `entity`, `uncertainty`, `trace`) and MTREE/HNSW vector indexes.
- **`llm.py`**: `LLMProvider` protocol with `MockLLMProvider` and `OpenAiLLMProvider`.
- **`embedding.py`**: `EmbeddingProvider` protocol with `OpenAiEmbeddingProvider`, `GeminiEmbeddingProvider`, `MockEmbeddingProvider`, and `CachedEmbeddingProvider` (LRU memory cache).
- **`ai.py`**: `CompositeAiService` pairing an LLM with an Embedder.

### Engine Layer (`cortex/engine/`)
- Implements the core cognitive and memory algorithms without direct knowledge of HTTP or CLI.
- **`IngestionEngine`**:
  - Handles single turns and batch conversations.
  - Reconciles existing facts against new inputs to detect changes and generate superseding links.
  - Automatically supersedes conflicting behavioral instructions.
  - Supports dry-run preview and direct triples ingestion.
- **`RetrievalEngine`**:
  - Hybrid search combining vector similarity and BM25 via Reciprocal Rank Fusion.
  - 1-hop knowledge graph relation expansion into `context_hits`.
  - Temporal filtering (`as_of`) for historical state reconstruction.
  - Structured Markdown context block assembly.
- **`CognitiveEngine`**:
  - `reflect`: Synthesizes thematic reflections across user memory.
  - `chat`: Interactive dialogue with grounded citation insertion and SSE streaming.
- **`GraphEngine`**:
  - Entity inspection, temporal history, neighborhood traversal, trace querying, and lifecycle retention expiration.

### Transport Layer (`cortex/transport/`)
- Exposes engines to external callers.
- **`http.py`**: FastAPI REST API providing memory and knowledge graph endpoints.
- **`mcp.py`**: Model Context Protocol implementation (JSON-RPC 2.0) supporting both compact single-tool facade and legacy tools.
- **`cli.py`**: Command-line interface with commands for serving, querying, and managing memory.

---

## 3. Data Flow Example: Ingestion & Reconciliation

```
Client (User or Agent)
       |
       | POST /api/v1/{ctx}/facts {"text": "I moved to Berlin", "infer": "full"}
       v
HTTP Router (http.py)
       |
       | IngestionEngine.remember(ctx, request)
       v
IngestionEngine (ingestion.py)
       |
       | 1. Query active facts: store.fetch_active_attributes()
       | 2. Candidate pre-filtering: vector scoring
       | 3. Extract & Reconcile: ai.extract_and_reconcile()
       v
LLM Provider (llm.py)
       |
       | LLM detects city conflict: (Old: Paris, New: Berlin)
       | Returns ExtractedFactOp(op="SUPERSEDE", target_id="attr:123", key="city", value="Berlin")
       v
IngestionEngine
       |
       | 1. Create new attribute (Berlin): store.create("attribute", ...)
       | 2. Supersede old attribute (Paris): store.supersede_attribute("attr:123", new_id)
       | 3. Create graph relation (User -> lives_in -> Berlin): store.create("relation", ...)
       | 4. Record trace: store.record_trace(...)
       v
SurrealDB (store.py)
       |
       | Atomic SurrealQL execution in isolated context
       v
Return RememberResponse with ExtractionDiff
```

---

## 4. Rust Parity Architecture

The architecture was intentionally planned to map 1:1 onto Rust idioms:

| Python Component | Rust Equivalent |
|---|---|
| `cortex/domain/enums.py` | `enum InferMode`, `enum MemoryCategory` with `#[derive(Serialize, Deserialize)]` |
| `cortex/domain/models.py` | Strong Rust structs with `serde(rename_all = "camelCase")` |
| `cortex/domain/scoring.py` | Pure functions using SIMD (`packed_simd` or `simba`) |
| `cortex/infra/store.py` | `struct SurrealStore` using `surrealdb` crate |
| `cortex/infra/llm.py` | `#[async_trait] pub trait LlmProvider` |
| `cortex/infra/embedding.py` | `#[async_trait] pub trait EmbeddingProvider` |
| `cortex/engine/*.py` | Services taking `Arc<SurrealStore>` and `Arc<dyn AiService>` |
| `cortex/runtime.py` | `struct CortexRuntime` wrapped in `Arc<CortexRuntime>` as Axum state |
| `cortex/transport/http.py` | `axum::Router` |
| `cortex/transport/mcp.py` | `tokio::io::BufReader<tokio::io::Stdin>` JSON-RPC loop |
