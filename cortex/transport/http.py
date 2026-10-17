import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from cortex.config import Config, load_config
from cortex.runtime import CortexRuntime
from cortex.transport.mcp import handle_mcp_request
from cortex.domain.models import (
    RememberRequest,
    RememberBatchRequest,
    RecallRequest,
    ContextRequest,
    ForgetRequest,
    HealthResponse,
    LookupRequest,
    ReflectRequest,
    ChatRequest,
    ConsolidateRequest,
    ResolveUncertaintyRequest,
    SessionCreateRequest,
)


def create_app(cfg: Config | None = None) -> FastAPI:
    config = cfg or load_config()
    runtime = CortexRuntime.from_config(config)
    ingestion = runtime.ingestion
    retrieval = runtime.retrieval
    cognitive = runtime.cognitive
    graph = runtime.graph

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await runtime.close()

    app = FastAPI(title="Cortex Agent Memory", version="0.5.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def custom_http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"message": str(exc.detail), "detail": str(exc.detail)},
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    idempotency_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _verify_auth(authorization: str | None = Header(default=None)):
        if config.server.api_key:
            if not authorization or not authorization.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Unauthorized")
            token = authorization[7:].strip()
            if token != config.server.api_key:
                raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        return HealthResponse()

    @app.post("/mcp")
    async def mcp_http(
        req: Request,
        context_id: str = "dev",
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        body = await req.json()
        resp = await handle_mcp_request(
            body,
            ingestion,
            retrieval,
            context_id=context_id,
            mcp_mode=config.server.mcp_mode,
        )
        return JSONResponse(content=resp)

    @app.post("/api/v1/{context_id}/facts")
    async def remember(
        context_id: str,
        req: RememberRequest,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        _verify_auth(authorization)
        if idempotency_key:
            cache_key = (context_id, idempotency_key)
            if cache_key in idempotency_cache:
                return JSONResponse(content=idempotency_cache[cache_key])

        res = await ingestion.remember(context_id, req)
        res_dict = res.model_dump(mode="json", by_alias=True, exclude_none=True)
        if idempotency_key:
            idempotency_cache[(context_id, idempotency_key)] = res_dict
        return JSONResponse(content=res_dict)

    @app.post("/api/v1/{context_id}/facts/batch")
    async def remember_batch(
        context_id: str,
        req: RememberBatchRequest,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        _verify_auth(authorization)
        if idempotency_key:
            cache_key = (context_id, idempotency_key)
            if cache_key in idempotency_cache:
                return JSONResponse(content=idempotency_cache[cache_key])

        res = await ingestion.remember_batch(context_id, req)
        res_dict = res.model_dump(mode="json", by_alias=True, exclude_none=True)
        if idempotency_key:
            idempotency_cache[(context_id, idempotency_key)] = res_dict
        return JSONResponse(content=res_dict)

    @app.post("/api/v1/{context_id}/query")
    async def recall(
        context_id: str,
        req: RecallRequest,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await retrieval.recall(context_id, req)
        return JSONResponse(
            content=res.model_dump(mode="json", by_alias=True, exclude_none=True)
        )

    @app.post("/api/v1/{context_id}/context")
    async def get_context(
        context_id: str,
        req: ContextRequest,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await retrieval.query_context(context_id, req)
        return JSONResponse(
            content=res.model_dump(mode="json", by_alias=True, exclude_none=True)
        )

    @app.post("/api/v1/{context_id}/forget")
    async def forget(
        context_id: str,
        req: ForgetRequest,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await ingestion.forget(context_id, req)
        return JSONResponse(
            content=res.model_dump(mode="json", by_alias=True, exclude_none=True)
        )

    @app.get("/api/v1/{context_id}/inspect")
    async def inspect(
        context_id: str,
        ref: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await retrieval.inspect(context_id, ref)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/state")
    async def get_state(
        context_id: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        await runtime.store.use_context(context_id)
        attr_count = await runtime.store.count_records("attribute")
        entity_count = await runtime.store.count_records("entity")
        inst_count = await runtime.store.count_records("instruction")
        return JSONResponse(
            content={
                "attributesCount": attr_count,
                "entitiesCount": entity_count,
                "instructionsCount": inst_count,
                "contextId": context_id,
            }
        )

    @app.post("/api/v1/{context_id}/lookup")
    async def lookup(
        context_id: str,
        req: LookupRequest,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await retrieval.lookup(context_id, req)
        return JSONResponse(
            content=res.model_dump(mode="json", by_alias=True, exclude_none=True)
        )

    @app.post("/api/v1/{context_id}/reflect")
    async def reflect(
        context_id: str,
        req: ReflectRequest,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await cognitive.reflect(context_id, req)
        return JSONResponse(
            content=res.model_dump(mode="json", by_alias=True, exclude_none=True)
        )

    @app.post("/api/v1/{context_id}/chat")
    async def chat(
        context_id: str,
        req: ChatRequest,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await cognitive.chat(context_id, req)
        if req.stream:

            async def event_generator():
                words = res.reply.split(" ")
                for i, w in enumerate(words):
                    chunk = w + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'delta': chunk})}\n\n"
                    await asyncio.sleep(0.01)
                final_payload = {
                    "reply": res.reply,
                    "citations": res.citations,
                    "traceId": res.trace_id,
                    "sessionId": res.session_id,
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        return JSONResponse(
            content=res.model_dump(mode="json", by_alias=True, exclude_none=True)
        )

    @app.post("/api/v1/{context_id}/consolidate")
    async def consolidate(
        context_id: str,
        req: ConsolidateRequest,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await ingestion.consolidate(context_id, req)
        return JSONResponse(
            content=res.model_dump(mode="json", by_alias=True, exclude_none=True)
        )

    @app.post("/api/v1/{context_id}/lifecycle/expire")
    async def expire(
        context_id: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await ingestion.expire_context(context_id)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/profile")
    async def get_profile(
        context_id: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.get_profile(context_id)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/me")
    async def whoami(
        context_id: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        return JSONResponse(
            content={
                "principalId": "local",
                "contextId": context_id,
                "grants": [
                    "memory:read",
                    "memory:write",
                    "memory:forget",
                    "grant:manage",
                ],
            }
        )

    @app.get("/api/v1/{context_id}/attributes")
    async def list_attributes(
        context_id: str,
        limit: int = 100,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.list_attributes(context_id, limit=limit)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/relations")
    async def list_relations(
        context_id: str,
        limit: int = 100,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.list_relations(context_id, limit=limit)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/entities")
    async def list_entities(
        context_id: str,
        q: str | None = None,
        entity_type: str | None = None,
        limit: int = 100,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.list_entities(
            context_id, q=q, entity_type=entity_type, limit=limit
        )
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/entities/search")
    async def search_entities(
        context_id: str,
        q: str = "",
        type: str | None = None,
        limit: int = 20,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.list_entities(context_id, q=q, entity_type=type, limit=limit)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/entities/top")
    async def top_entities(
        context_id: str,
        type: str | None = None,
        limit: int = 20,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.list_entities(context_id, entity_type=type, limit=limit)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/entities/{entity_type}/{entity_name}")
    async def get_entity(
        context_id: str,
        entity_type: str,
        entity_name: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.get_entity_detail(context_id, entity_type, entity_name)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/entities/{entity_type}/{entity_name}/history")
    async def get_entity_history(
        context_id: str,
        entity_type: str,
        entity_name: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.get_entity_history(context_id, entity_type, entity_name)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/entities/{entity_type}/{entity_name}/history/{key}")
    async def get_entity_history_key(
        context_id: str,
        entity_type: str,
        entity_name: str,
        key: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.get_entity_history(
            context_id, entity_type, entity_name, key=key
        )
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/entities/{entity_type}/{entity_name}/neighbourhood")
    async def get_entity_neighbourhood(
        context_id: str,
        entity_type: str,
        entity_name: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.get_entity_neighbourhood(context_id, entity_type, entity_name)
        return JSONResponse(content=res)

    @app.delete("/api/v1/{context_id}/entities/{entity_type}/{entity_name}")
    async def delete_entity(
        context_id: str,
        entity_type: str,
        entity_name: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.delete_entity(context_id, entity_type, entity_name)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/uncertainty")
    async def list_uncertainties(
        context_id: str,
        resolved: bool | None = None,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.list_uncertainties(context_id, resolved=resolved)
        return JSONResponse(content=res)

    @app.post("/api/v1/{context_id}/uncertainty/{uncertainty_id}/resolve")
    async def resolve_uncertainty(
        context_id: str,
        uncertainty_id: str,
        req: ResolveUncertaintyRequest,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await ingestion.resolve_uncertainty(context_id, uncertainty_id, req)
        return JSONResponse(content=res)

    @app.post("/api/v1/{context_id}/sessions")
    async def create_session(
        context_id: str,
        req: SessionCreateRequest | None = None,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        await runtime.store.use_context(context_id)
        sess_id = f"sess_{uuid.uuid4().hex[:10]}"
        s_rec = f"session:{sess_id}"
        await runtime.store.upsert_session(s_rec, "global")
        return JSONResponse(content={"sessionId": sess_id, "id": s_rec})

    @app.delete("/api/v1/{context_id}/sessions/{session_id}")
    async def delete_session(
        context_id: str,
        session_id: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.delete_session(context_id, session_id)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/sessions/{session_id}/turns")
    async def get_session_turns(
        context_id: str,
        session_id: str,
        limit: int = 100,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.get_session_turns(context_id, session_id, limit=limit)
        return JSONResponse(content=res)

    @app.post("/api/v1/{context_id}/sessions/{session_id}/context")
    async def get_session_context(
        context_id: str,
        session_id: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        turns = await graph.get_session_turns(context_id, session_id, limit=100)
        context_text = "\n".join(
            f"[{t.get('role', 'user')}]: {t.get('content', '')}" for t in turns
        )
        return JSONResponse(content={"context": context_text})

    @app.get("/api/v1/{context_id}/traces")
    async def list_traces(
        context_id: str,
        limit: int = 100,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.list_traces(context_id, limit=limit)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/traces/stats")
    async def get_trace_stats(
        context_id: str,
        windowHours: int = 24,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.get_trace_stats(context_id, window_hours=windowHours)
        return JSONResponse(content=res)

    @app.get("/api/v1/{context_id}/traces/{trace_id}")
    async def get_trace(
        context_id: str,
        trace_id: str,
        authorization: str | None = Header(default=None),
    ):
        _verify_auth(authorization)
        res = await graph.get_trace(context_id, trace_id)
        return JSONResponse(content=res)

    app.state.runtime = runtime
    app.state.store = runtime.store
    app.state.pipeline = ingestion
    app.state.retrieval = retrieval
    app.state.config = config

    return app
