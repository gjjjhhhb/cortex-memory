# Architecture & Design

Cortex implements a Clean Architecture model where business logic is strictly decoupled from external storage and transports.

---

## 1. Layers

```
+-------------------------------------------------------------------+
|                        Transport Layer                            |
|       cli.py (CLI)   |   http.py (HTTP)   |   mcp.py (MCP)        |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                       Runtime Container                           |
|       runtime.py (CortexRuntime / Dependency Injection)           |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                         Engine Layer                              |
|   ingestion.py   |  retrieval.py  |  cognitive.py  |  graph.py    |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     Infrastructure Layer                          |
|         store.py (Database)   |   llm.py / embedding.py (AI)      |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                         Domain Layer                              |
|     models.py    |   enums.py     |   scoring.py   | formatting.py|
+-------------------------------------------------------------------+
```

---

## 2. Layer Responsibilities

- **Domain (`cortex/domain/`)**: Pure synchronous structures and algorithms (cosine similarity, BM25, RRF scoring, Markdown formatting, DTOs). Zero external dependencies.
- **Infrastructure (`cortex/infra/`)**: Database repository, schema DDL, and pluggable AI/embedding provider implementations with memory caching.
- **Engine (`cortex/engine/`)**: Core logic — fact reconciliation, superseding conflicting rules, hybrid retrieval, graph traversal, and cognitive reflection.
- **Transport (`cortex/transport/`)**: HTTP REST endpoints, standard MCP JSON-RPC protocol (stdio), and CLI commands.

---

## 3. Data Flow

```
Client Request
       |
       v
Transport Layer (HTTP / MCP / CLI)
       |
       v
Ingestion Engine (reconciliation & candidate selection)
       |
       +---> AI Provider (extract entities, attributes & relations)
       |
       +---> Storage Layer (atomic SurrealQL queries)
       |
       v
Client Response
```
