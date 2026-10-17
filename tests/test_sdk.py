import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_error_format_compatibility(app_client: AsyncClient):
    resp = await app_client.post("/api/v1/dev/facts", json={"text": "Hello"})
    assert resp.status_code == 401
    err_json = resp.json()
    assert "detail" in err_json
    assert "message" in err_json
    assert err_json["message"] == "Unauthorized"


@pytest.mark.asyncio
async def test_infer_preview_mode(app_client: AsyncClient, auth_headers: dict):
    headers = auth_headers
    ctx = "test_preview"

    resp = await app_client.post(
        f"/api/v1/{ctx}/facts",
        json={"text": "Alice works at Acme Corp", "infer": "preview"},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "preview"
    assert "preview" in data
    assert "entities" in data["preview"]

    rec_resp = await app_client.post(
        f"/api/v1/{ctx}/query",
        json={"query": "Alice"},
        headers=headers,
    )
    assert rec_resp.status_code == 200
    assert len(rec_resp.json()["hits"]) == 0


@pytest.mark.asyncio
async def test_infer_triples_mode(app_client: AsyncClient, auth_headers: dict):
    headers = auth_headers
    ctx = "test_triples"

    payload = {
        "text": "Direct knowledge",
        "infer": "triples",
        "triples": {
            "attributes": [
                {
                    "entity_type": "user",
                    "entity_name": "Bob",
                    "name": "role",
                    "value": "Architect",
                }
            ],
            "relations": [
                {
                    "source_type": "user",
                    "source_name": "Bob",
                    "target_type": "project",
                    "target_name": "Matrix",
                    "relation_type": "leads",
                }
            ],
        },
    }
    resp = await app_client.post(f"/api/v1/{ctx}/facts", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "triples"

    attrs = await app_client.get(f"/api/v1/{ctx}/attributes", headers=headers)
    assert attrs.status_code == 200
    attr_list = attrs.json()
    assert any(
        a.get("key") == "role" and a.get("value") == "Architect" for a in attr_list
    )

    rels = await app_client.get(f"/api/v1/{ctx}/relations", headers=headers)
    assert rels.status_code == 200
    rel_list = rels.json()
    assert any(r.get("predicate") == "leads" for r in rel_list)


@pytest.mark.asyncio
async def test_remember_batch_whole_conversation(app_client: AsyncClient, auth_headers: dict):
    headers = auth_headers
    ctx = "test_conv_batch"

    payload = {
        "items": [
            {"text": "Hello, my name is Charlie", "scope": "conversation"},
            {
                "text": "Nice to meet you Charlie, what do you do?",
                "scope": "conversation",
            },
            {"text": "I am a Data Scientist at DeepMind", "scope": "conversation"},
        ],
        "extract": "whole_conversation",
    }
    resp = await app_client.post(
        f"/api/v1/{ctx}/facts/batch", json=payload, headers=headers
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data.get("turnIds", [])) == 3

    session_id = data.get("sessionId")
    turns_resp = await app_client.get(
        f"/api/v1/{ctx}/sessions/{session_id}/turns", headers=headers
    )
    assert turns_resp.status_code == 200
    turns = turns_resp.json()
    assert len(turns) == 3


@pytest.mark.asyncio
async def test_forget_dry_run_and_purge(app_client: AsyncClient, auth_headers: dict):
    headers = auth_headers
    ctx = "test_forget_dry"

    await app_client.post(
        f"/api/v1/{ctx}/facts",
        json={"text": "User loves Python programming", "infer": "full"},
        headers=headers,
    )

    dry_resp = await app_client.post(
        f"/api/v1/{ctx}/forget",
        json={"query": "Python", "dry_run": True},
        headers=headers,
    )
    assert dry_resp.status_code == 200
    dry_data = dry_resp.json()
    assert dry_data["deleted"] >= 1
    assert "matches" in dry_data
    assert dry_data.get("dryRun") is True or dry_data.get("dry_run") is True

    rec1 = await app_client.post(
        f"/api/v1/{ctx}/query", json={"query": "Python"}, headers=headers
    )
    assert rec1.status_code == 200
    assert len(rec1.json()["hits"]) >= 1

    del_resp = await app_client.post(
        f"/api/v1/{ctx}/forget",
        json={"query": "Python", "dry_run": False},
        headers=headers,
    )
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] >= 1

    rec2 = await app_client.post(
        f"/api/v1/{ctx}/query", json={"query": "Python"}, headers=headers
    )
    assert rec2.status_code == 200
    assert len(rec2.json()["hits"]) == 0


@pytest.mark.asyncio
async def test_lookup_and_introspection(app_client: AsyncClient, auth_headers: dict):
    headers = auth_headers
    ctx = "test_introspection"

    await app_client.post(
        f"/api/v1/{ctx}/facts",
        json={"text": "Dave works as Tech Lead at Google since 2020", "infer": "full"},
        headers=headers,
    )

    lookup_resp = await app_client.post(
        f"/api/v1/{ctx}/lookup",
        json={"include": ["entities", "attributes", "relations"]},
        headers=headers,
    )
    assert lookup_resp.status_code == 200
    lookup_data = lookup_resp.json()
    assert "entities" in lookup_data
    assert "facts" in lookup_data

    entities_resp = await app_client.get(f"/api/v1/{ctx}/entities", headers=headers)
    assert entities_resp.status_code == 200
    ents = entities_resp.json()
    assert len(ents) >= 1
    ent_type = ents[0].get("entityType") or ents[0].get("entity_type") or "concept"
    ent_name = ents[0]["name"]

    detail_resp = await app_client.get(
        f"/api/v1/{ctx}/entities/{ent_type}/{ent_name}", headers=headers
    )
    assert detail_resp.status_code == 200

    hist_resp = await app_client.get(
        f"/api/v1/{ctx}/entities/{ent_type}/{ent_name}/history", headers=headers
    )
    assert hist_resp.status_code == 200

    neigh_resp = await app_client.get(
        f"/api/v1/{ctx}/entities/{ent_type}/{ent_name}/neighbourhood", headers=headers
    )
    assert neigh_resp.status_code == 200

    prof_resp = await app_client.get(f"/api/v1/{ctx}/profile", headers=headers)
    assert prof_resp.status_code == 200
    assert "markdown" in prof_resp.json()

    me_resp = await app_client.get(f"/api/v1/{ctx}/me", headers=headers)
    assert me_resp.status_code == 200
    assert "principalId" in me_resp.json()

    del_ent = await app_client.delete(
        f"/api/v1/{ctx}/entities/{ent_type}/{ent_name}", headers=headers
    )
    assert del_ent.status_code == 200
    assert del_ent.json()["status"] == "deleted"


@pytest.mark.asyncio
async def test_reflect_and_chat_endpoints(app_client: AsyncClient, auth_headers: dict):
    headers = auth_headers
    ctx = "test_cognitive"

    await app_client.post(
        f"/api/v1/{ctx}/facts",
        json={
            "text": "User drinks matcha latte every morning and writes Elixir code",
            "infer": "full",
        },
        headers=headers,
    )

    refl_resp = await app_client.post(
        f"/api/v1/{ctx}/reflect",
        json={"query": "Morning habits", "persist": True},
        headers=headers,
    )
    assert refl_resp.status_code == 200
    refl_data = refl_resp.json()
    assert "reflection" in refl_data
    assert len(refl_data["reflection"]) > 0

    chat_resp = await app_client.post(
        f"/api/v1/{ctx}/chat",
        json={"message": "What do I drink in the morning?"},
        headers=headers,
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert "reply" in chat_data
    assert "citations" in chat_data


@pytest.mark.asyncio
async def test_uncertainties_traces_and_lifecycle(app_client: AsyncClient, auth_headers: dict):
    headers = auth_headers
    ctx = "test_ops"

    await app_client.post(
        f"/api/v1/{ctx}/facts",
        json={"text": "System testing trace generation", "infer": "full"},
        headers=headers,
    )

    traces_resp = await app_client.get(f"/api/v1/{ctx}/traces", headers=headers)
    assert traces_resp.status_code == 200
    traces = traces_resp.json()
    assert len(traces) >= 1
    trace_id = traces[0]["id"]

    trace_detail = await app_client.get(
        f"/api/v1/{ctx}/traces/{trace_id}", headers=headers
    )
    assert trace_detail.status_code == 200

    stats_resp = await app_client.get(f"/api/v1/{ctx}/traces/stats", headers=headers)
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats.get("totalTraces", stats.get("total_traces", 0)) >= 1

    unc_resp = await app_client.get(f"/api/v1/{ctx}/uncertainty", headers=headers)
    assert unc_resp.status_code == 200

    cons_resp = await app_client.post(
        f"/api/v1/{ctx}/consolidate",
        json={"strategy": "all"},
        headers=headers,
    )
    assert cons_resp.status_code == 200
    assert "mergedCount" in cons_resp.json() or "decisions" in cons_resp.json()

    exp_resp = await app_client.post(f"/api/v1/{ctx}/lifecycle/expire", headers=headers)
    assert exp_resp.status_code == 200
    assert "expired_count" in exp_resp.json() or "expiredCount" in exp_resp.json()


@pytest.mark.asyncio
async def test_mcp_reflect_tool(app_client: AsyncClient, auth_headers: dict):
    headers = auth_headers

    call_msg = {
        "jsonrpc": "2.0",
        "id": 99,
        "method": "tools/call",
        "params": {
            "name": "reflect",
            "arguments": {
                "query": "Synthesize key preferences",
                "persist": False,
            },
        },
    }
    resp = await app_client.post("/mcp", json=call_msg, headers=headers)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["result"]["isError"] is False
    assert len(res_data["result"]["content"]) > 0
