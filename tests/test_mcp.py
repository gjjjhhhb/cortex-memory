import pytest
from httpx import AsyncClient
from cortex import MockAiProvider


@pytest.mark.asyncio
async def test_compact_mcp_tools_and_capabilities(app_client: AsyncClient):
    headers = {"Authorization": "Bearer cortex-local-dev-key"}

    init_msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    init_resp = await app_client.post("/mcp", json=init_msg, headers=headers)
    assert init_resp.status_code == 200
    caps = init_resp.json()["result"]["capabilities"]
    assert "tools" in caps
    assert "resources" in caps
    assert "prompts" in caps

    list_msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    list_resp = await app_client.post("/mcp", json=list_msg, headers=headers)
    assert list_resp.status_code == 200
    tools = list_resp.json()["result"]["tools"]
    assert len(tools) == 1
    assert tools[0]["name"] == "cortex"


@pytest.mark.asyncio
async def test_compact_mcp_actions(app_client: AsyncClient):
    headers = {"Authorization": "Bearer cortex-local-dev-key"}

    rem_msg = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "cortex",
            "arguments": {
                "action": "remember",
                "query": "Elena is Lead ML Engineer in Zurich",
                "options": {"infer": "full"},
            },
        },
    }
    rem_res = await app_client.post("/mcp", json=rem_msg, headers=headers)
    assert rem_res.status_code == 200
    assert rem_res.json()["result"]["isError"] is False

    rec_msg = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "cortex",
            "arguments": {
                "action": "recall",
                "query": "Zurich",
                "options": {"k": 3},
            },
        },
    }
    rec_res = await app_client.post("/mcp", json=rec_msg, headers=headers)
    assert rec_res.status_code == 200
    rec_text = rec_res.json()["result"]["content"][0]["text"]
    assert "Zurich" in rec_text

    ctx_msg = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "cortex",
            "arguments": {
                "action": "context",
                "query": "Elena role",
            },
        },
    }
    ctx_res = await app_client.post("/mcp", json=ctx_msg, headers=headers)
    assert ctx_res.status_code == 200
    assert ctx_res.json()["result"]["isError"] is False

    help_msg = {
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {
            "name": "cortex",
            "arguments": {"action": "help"},
        },
    }
    help_res = await app_client.post("/mcp", json=help_msg, headers=headers)
    assert help_res.status_code == 200
    assert "remember" in help_res.json()["result"]["content"][0]["text"]

    legacy_msg = {
        "jsonrpc": "2.0",
        "id": 14,
        "method": "tools/call",
        "params": {
            "name": "recall",
            "arguments": {"query": "Zurich", "k": 2},
        },
    }
    leg_res = await app_client.post("/mcp", json=legacy_msg, headers=headers)
    assert leg_res.status_code == 200
    assert leg_res.json()["result"]["isError"] is False


@pytest.mark.asyncio
async def test_mcp_resources_and_prompts(app_client: AsyncClient):
    headers = {"Authorization": "Bearer cortex-local-dev-key"}

    res_list_msg = {"jsonrpc": "2.0", "id": 20, "method": "resources/list"}
    res_list = await app_client.post("/mcp", json=res_list_msg, headers=headers)
    assert res_list.status_code == 200
    resources = res_list.json()["result"]["resources"]
    uris = [r["uri"] for r in resources]
    assert "cortex://profile" in uris
    assert "cortex://stats" in uris

    res_read_msg = {
        "jsonrpc": "2.0",
        "id": 21,
        "method": "resources/read",
        "params": {"uri": "cortex://profile"},
    }
    res_read = await app_client.post("/mcp", json=res_read_msg, headers=headers)
    assert res_read.status_code == 200
    contents = res_read.json()["result"]["contents"]
    assert len(contents) > 0
    assert contents[0]["uri"] == "cortex://profile"

    pr_list_msg = {"jsonrpc": "2.0", "id": 22, "method": "prompts/list"}
    pr_list = await app_client.post("/mcp", json=pr_list_msg, headers=headers)
    assert pr_list.status_code == 200
    prompts = pr_list.json()["result"]["prompts"]
    assert any(p["name"] == "memory-context" for p in prompts)

    pr_get_msg = {
        "jsonrpc": "2.0",
        "id": 23,
        "method": "prompts/get",
        "params": {
            "name": "memory-context",
            "arguments": {"topic": "engineering rules"},
        },
    }
    pr_get = await app_client.post("/mcp", json=pr_get_msg, headers=headers)
    assert pr_get.status_code == 200
    msgs = pr_get.json()["result"]["messages"]
    assert len(msgs) > 0
    assert msgs[0]["role"] == "user"


@pytest.mark.asyncio
async def test_ai_lru_cache():
    ai = MockAiProvider(dimension=64)
    assert ai.embedder.hits == 0

    v1 = await ai.get_embedding("Test semantic cache text")
    assert ai.embedder.hits == 0
    assert ai.embedder.misses == 1

    v2 = await ai.get_embedding("Test semantic cache text")
    assert v1 == v2
    assert ai.embedder.hits == 1

    await ai.get_embedding("Another distinct text")
    assert ai.embedder.hits == 1
    assert ai.embedder.misses == 2


@pytest.mark.asyncio
async def test_chat_sse_streaming(app_client: AsyncClient):
    headers = {"Authorization": "Bearer cortex-local-dev-key"}
    ctx = "test_stream"

    await app_client.post(
        f"/api/v1/{ctx}/facts",
        json={"text": "System architect drinks matcha at 9am", "infer": "full"},
        headers=headers,
    )

    resp = await app_client.post(
        f"/api/v1/{ctx}/chat",
        json={"message": "What does architect drink?", "stream": True},
        headers=headers,
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    content = resp.text
    assert "data: " in content
    assert "[DONE]" in content
