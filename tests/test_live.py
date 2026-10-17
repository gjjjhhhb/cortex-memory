import os
import pytest
from httpx import AsyncClient, ASGITransport

from cortex.config import load_config
from cortex import create_app, get_ai_provider


@pytest.fixture(scope="session")
def live_config():
    cfg = load_config()
    if not (cfg.llm.api_key and cfg.embedding.api_key):
        pytest.skip("No live LLM or Embedding API keys found in config/environment.")
    return cfg


@pytest.fixture
async def live_client(live_config):
    app = create_app(live_config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await app.state.store.close()


@pytest.mark.asyncio
async def test_live_ai_service(live_config):
    ai = get_ai_provider(live_config.llm, live_config.embedding)

    emb = await ai.get_embedding("Cortex tri-temporal semantic memory test")
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert all(isinstance(x, float) for x in emb)
    assert len(emb) == live_config.embedding.dimension or len(emb) in (768, 1536, 3072)

    resp = await ai.complete_text(
        "You are a helpful assistant.", "Say: 'CORTEX_LIVE_OK'"
    )
    assert "CORTEX_LIVE_OK" in resp or len(resp.strip()) > 0


@pytest.mark.asyncio
async def test_live_end_to_end_pipeline(live_client: AsyncClient, live_config):
    headers = {"Authorization": f"Bearer {live_config.server.api_key}"}
    ctx = "live_test_ctx"

    rem_resp = await live_client.post(
        f"/api/v1/{ctx}/facts",
        json={
            "text": "Alexander is a principal architect at DeepMind living in London.",
            "infer": "full",
        },
        headers=headers,
    )
    assert rem_resp.status_code == 200
    rem_data = rem_resp.json()
    assert rem_data.get("mode") == "full"
    assert "sessionId" in rem_data

    rec_resp = await live_client.post(
        f"/api/v1/{ctx}/query",
        json={"query": "Where does Alexander live?", "k": 5},
        headers=headers,
    )
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    hits = rec_data.get("hits", [])
    assert len(hits) > 0
    found_london = any(
        "london" in (h.get("text") or "").lower()
        or "deepmind" in (h.get("text") or "").lower()
        for h in hits
    )
    assert found_london, f"Expected hits containing London/DeepMind, got: {hits}"

    ctx_resp = await live_client.post(
        f"/api/v1/{ctx}/context",
        json={"query": "Alexander profession and location"},
        headers=headers,
    )
    assert ctx_resp.status_code == 200
    ctx_text = ctx_resp.json().get("context", "")
    assert len(ctx_text) > 0


@pytest.mark.asyncio
async def test_live_cognitive_reflect_and_chat(live_client: AsyncClient, live_config):
    headers = {"Authorization": f"Bearer {live_config.server.api_key}"}
    ctx = "live_test_cog"

    await live_client.post(
        f"/api/v1/{ctx}/facts",
        json={
            "text": "User prefers dark roast pour-over coffee brewed at 92 degrees Celsius.",
            "infer": "full",
        },
        headers=headers,
    )

    refl_resp = await live_client.post(
        f"/api/v1/{ctx}/reflect",
        json={"query": "Coffee brewing preferences", "persist": False},
        headers=headers,
    )
    assert refl_resp.status_code == 200
    reflection = refl_resp.json().get("reflection", "")
    assert len(reflection) > 0

    chat_resp = await live_client.post(
        f"/api/v1/{ctx}/chat",
        json={"message": "What kind of coffee do I like and at what temperature?"},
        headers=headers,
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    reply = chat_data.get("reply", "")
    citations = chat_data.get("citations", [])
    assert len(reply) > 0
    assert len(citations) > 0
