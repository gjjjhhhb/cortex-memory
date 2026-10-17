import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_api_health(app_client: AsyncClient):
    resp = await app_client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_api_remember_recall_and_context(app_client: AsyncClient):
    headers = {"Authorization": "Bearer cortex-local-dev-key"}
    ctx = "dev"

    rem_resp = await app_client.post(
        f"/api/v1/{ctx}/facts",
        json={"text": "User is a Senior Backend Engineer in Berlin", "infer": "full"},
        headers=headers,
    )
    assert rem_resp.status_code == 200
    rem_data = rem_resp.json()
    assert "turnId" in rem_data
    assert "sessionId" in rem_data

    rec_resp = await app_client.post(
        f"/api/v1/{ctx}/query",
        json={"query": "Berlin", "k": 5},
        headers=headers,
    )
    assert rec_resp.status_code == 200
    rec_data = rec_resp.json()
    assert len(rec_data["hits"]) > 0
    assert "Berlin" in rec_data["hits"][0]["text"]
    assert rec_data["tier"] == "hybrid"

    ctx_resp = await app_client.post(
        f"/api/v1/{ctx}/context",
        json={"query": "What is the user's role and location?", "k": 5},
        headers=headers,
    )
    assert ctx_resp.status_code == 200
    ctx_data = ctx_resp.json()
    assert "User Identity & Profile" in ctx_data["context"]
    assert "Berlin" in ctx_data["context"]


@pytest.mark.asyncio
async def test_api_forget(app_client: AsyncClient):
    headers = {"Authorization": "Bearer cortex-local-dev-key"}
    ctx = "dev"

    await app_client.post(
        f"/api/v1/{ctx}/facts",
        json={"text": "User loves chocolate ice cream", "infer": "none"},
        headers=headers,
    )

    rec1 = await app_client.post(
        f"/api/v1/{ctx}/query",
        json={"query": "chocolate ice cream"},
        headers=headers,
    )
    assert len(rec1.json()["hits"]) > 0

    forget_resp = await app_client.post(
        f"/api/v1/{ctx}/forget",
        json={"query": "chocolate ice cream", "purge": True},
        headers=headers,
    )
    assert forget_resp.status_code == 200
    assert forget_resp.json()["deleted"] >= 1

    rec2 = await app_client.post(
        f"/api/v1/{ctx}/query",
        json={"query": "chocolate ice cream"},
        headers=headers,
    )
    assert len(rec2.json()["hits"]) == 0


@pytest.mark.asyncio
async def test_api_idempotency_key(app_client: AsyncClient):
    headers = {
        "Authorization": "Bearer cortex-local-dev-key",
        "Idempotency-Key": "idemp-test-uuid-1234",
    }
    ctx = "dev"

    resp1 = await app_client.post(
        f"/api/v1/{ctx}/facts",
        json={"text": "Idempotent event log"},
        headers=headers,
    )
    assert resp1.status_code == 200
    turn1 = resp1.json()["turnId"]

    resp2 = await app_client.post(
        f"/api/v1/{ctx}/facts",
        json={"text": "Idempotent event log"},
        headers=headers,
    )
    assert resp2.status_code == 200
    turn2 = resp2.json()["turnId"]

    assert turn1 == turn2


@pytest.mark.asyncio
async def test_api_unauthorized_access(app_client: AsyncClient):
    resp = await app_client.post(
        "/api/v1/dev/facts",
        json={"text": "Unauthorized fact attempt"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401
