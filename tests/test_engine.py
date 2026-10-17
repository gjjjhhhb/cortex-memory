from datetime import datetime, timezone
import pytest

from cortex.domain.models import (
    RememberRequest,
    RecallRequest,
    ContextRequest,
    LLMExtractionPayload,
)
from cortex.infra.llm import MockLLMProvider
from cortex.infra.embedding import MockEmbeddingProvider, CachedEmbeddingProvider
from cortex.infra.ai import CompositeAiService
from cortex.transport.mcp import handle_mcp_request


@pytest.mark.asyncio
async def test_tri_temporal_supersede_and_as_of(direct_engines):
    ingestion = direct_engines["ingestion"]
    retrieval = direct_engines["retrieval"]
    ctx = "test_temporal"

    t1 = datetime(2025, 1, 10, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc)

    rem1 = RememberRequest(text="Alice lives in Paris", observed_at=t1, infer="full")
    res1 = await ingestion.remember(ctx, rem1)
    assert res1.turn_id is not None
    assert len(res1.extraction.attributes) > 0

    rem2 = RememberRequest(text="Alice moved to Berlin", observed_at=t2, infer="full")
    res2 = await ingestion.remember(ctx, rem2)
    assert len(res2.extraction.superseded) > 0

    rec_now = await retrieval.recall(ctx, RecallRequest(query="Where does Alice live?"))
    assert len(rec_now.hits) > 0
    assert "Berlin" in rec_now.hits[0].text
    for h in rec_now.hits:
        assert "Paris" not in h.text

    t_past = datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    rec_past = await retrieval.recall(
        ctx,
        RecallRequest(query="Where does Alice live?", as_of=t_past),
    )
    assert len(rec_past.hits) > 0
    assert "Paris" in rec_past.hits[0].text


@pytest.mark.asyncio
async def test_instruction_learning_and_superseding(direct_engines):
    ingestion = direct_engines["ingestion"]
    retrieval = direct_engines["retrieval"]
    ctx = "test_instructions"

    rem1 = RememberRequest(text="Always respond briefly and concisely", infer="full")
    res1 = await ingestion.remember(ctx, rem1)
    assert len(res1.extraction.instructions) > 0

    rem2 = RememberRequest(text="Always respond with detailed examples", infer="full")
    res2 = await ingestion.remember(ctx, rem2)
    assert len(res2.extraction.superseded) > 0

    ctx_res = await retrieval.query_context(ctx, ContextRequest(query="How to answer?"))
    assert "Active Instructions" in ctx_res.context
    assert "detailed examples" in ctx_res.context
    assert "concisely" not in ctx_res.context


@pytest.mark.asyncio
async def test_batch_learning_languages_cumulative(direct_engines):
    ingestion = direct_engines["ingestion"]
    retrieval = direct_engines["retrieval"]
    ctx = "test_cumulative"

    await ingestion.remember(ctx, RememberRequest(text="I know Rust", infer="full"))
    await ingestion.remember(ctx, RememberRequest(text="I know Python", infer="full"))

    rec = await retrieval.recall(ctx, RecallRequest(query="programming languages"))
    facts = [h.text for h in rec.hits]
    assert any("Rust" in f for f in facts)
    assert any("Python" in f for f in facts)


@pytest.mark.asyncio
async def test_hybrid_search_scoring(direct_engines):
    ingestion = direct_engines["ingestion"]
    retrieval = direct_engines["retrieval"]
    ctx = "test_hybrid"

    await ingestion.remember(
        ctx,
        RememberRequest(text="Project Apollo uses SurrealDB and Python", infer="none"),
    )
    await ingestion.remember(
        ctx,
        RememberRequest(
            text="Meeting notes about financial quarterly results", infer="none"
        ),
    )

    rec_hybrid = await retrieval.recall(
        ctx, RecallRequest(query="Apollo SurrealDB", mode="hybrid")
    )
    assert len(rec_hybrid.hits) > 0
    assert "Apollo" in rec_hybrid.hits[0].text

    rec_vec = await retrieval.recall(
        ctx, RecallRequest(query="financial results", mode="vector")
    )
    assert len(rec_vec.hits) > 0
    assert "financial" in rec_vec.hits[0].text

    rec_bm25 = await retrieval.recall(
        ctx, RecallRequest(query="Project Apollo", mode="bm25")
    )
    assert len(rec_bm25.hits) > 0
    assert "Apollo" in rec_bm25.hits[0].text


@pytest.mark.asyncio
async def test_mcp_stdio_jsonrpc_protocol(direct_engines):
    ingestion = direct_engines["ingestion"]
    retrieval = direct_engines["retrieval"]
    ctx = "test_mcp"

    init_msg = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client"},
        },
    }
    init_res = await handle_mcp_request(init_msg, ingestion, retrieval, context_id=ctx)
    assert init_res["result"]["serverInfo"]["name"] == "cortex"

    tools_msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    tools_res = await handle_mcp_request(
        tools_msg, ingestion, retrieval, context_id=ctx
    )
    assert len(tools_res["result"]["tools"]) > 0

    call_msg = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "cortex",
            "arguments": {
                "action": "remember",
                "query": "Personal agent memory initialized",
            },
        },
    }
    call_res = await handle_mcp_request(call_msg, ingestion, retrieval, context_id=ctx)
    assert call_res["result"]["isError"] is False


@pytest.mark.asyncio
async def test_context_rendering_deduplication(direct_engines):
    ingestion = direct_engines["ingestion"]
    retrieval = direct_engines["retrieval"]
    ctx = "test_dedup"

    await ingestion.remember(
        ctx,
        RememberRequest(
            text="User lives in Berlin", infer="full", memory_category="identity"
        ),
    )

    ctx_res = await retrieval.query_context(
        ctx, ContextRequest(query="Where does the user live?")
    )

    assert "## User Identity & Profile" in ctx_res.context
    assert "User lives in Berlin" in ctx_res.context

    if "## Relevant Memories" in ctx_res.context:
        mem_part = ctx_res.context.split("## Relevant Memories")[1]
        assert "User lives in Berlin" not in mem_part


@pytest.mark.asyncio
async def test_clean_architecture_provider_composition():
    llm = MockLLMProvider()
    base_embedder = MockEmbeddingProvider(dimension=128)
    cached_embedder = CachedEmbeddingProvider(base_embedder, max_size=50)

    ai = CompositeAiService(llm=llm, embedder=cached_embedder)

    emb1 = await ai.get_embedding("clean architecture test")
    emb2 = await ai.get_embedding("clean architecture test")
    assert emb1 == emb2
    assert len(emb1) == 128
    assert cached_embedder.cache_info()["hits"] == 1

    payload = await ai.extract_and_reconcile(
        input_text="I live in Berlin",
        anchor_time=datetime.now(timezone.utc),
        existing_facts=[],
    )
    assert isinstance(payload, LLMExtractionPayload)
    assert any(e.name == "Berlin" for e in payload.entities)
