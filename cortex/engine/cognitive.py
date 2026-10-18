import uuid
from datetime import datetime, timezone
from typing import Any

from cortex.domain.models import (
    ReflectRequest,
    ReflectResponse,
    ChatRequest,
    ChatResponse,
    RecallRequest,
)
from cortex.domain.prompts import REFLECT_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
from cortex.infra.store import CortexStore


class CognitiveEngine:
    def __init__(self, store: CortexStore, ai: Any, retrieval: Any):
        self.store = store
        self.ai = ai
        self.retrieval = retrieval

    async def reflect(
        self, context_id: str, req: ReflectRequest, pipeline: Any = None
    ) -> ReflectResponse:
        await self.store.use_context(context_id)
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        recall_req = RecallRequest(query=req.query, k=15, lens=req.lens)
        rec_res = await self.retrieval.recall(context_id, recall_req)

        evidence_lines = []
        evidence_refs = []
        for h in rec_res.hits:
            evidence_lines.append(f"[{h.id}] {h.text}")
            evidence_refs.append(h.id)

        user_prompt = f"Query: {req.query}\n\nEvidence facts:\n" + "\n".join(
            evidence_lines
        )
        reflection_text = await self.ai.complete_text(
            REFLECT_SYSTEM_PROMPT, user_prompt
        )

        persisted_attrs: list[dict[str, Any]] = []
        if req.persist:
            now_iso = datetime.now(timezone.utc).isoformat()
            emb = await self.ai.get_embedding(reflection_text)
            new_attr = {
                "entity": "entity:user",
                "key": "reflection",
                "value": reflection_text[:100],
                "statement": reflection_text,
                "memory_category": "knowledge",
                "valid_from": now_iso,
                "valid_until": None,
                "created_at": now_iso,
                "superseded_by": None,
                "confidence": 0.9,
                "scope": "global",
                "labels": ["source=reflect"],
                "embedding": emb,
            }
            res_attr = await self.store.create("attribute", new_attr)
            clean = {k: v for k, v in new_attr.items() if k != "embedding"}
            clean["id"] = str(res_attr.get("id", ""))
            persisted_attrs.append(clean)

        return ReflectResponse(
            reflection=reflection_text,
            evidence_refs=evidence_refs,
            persisted_attributes=persisted_attrs,
            trace_id=trace_id,
        )

    async def chat(
        self, context_id: str, req: ChatRequest, pipeline: Any = None
    ) -> ChatResponse:
        await self.store.use_context(context_id)
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        session_id = req.session_id or f"sess_{uuid.uuid4().hex[:10]}"

        recall_req = RecallRequest(query=req.message, k=8, lens=req.scopes)
        rec_res = await self.retrieval.recall(context_id, recall_req)

        sources_block = []
        citations = []
        for i, h in enumerate(rec_res.hits, 1):
            sources_block.append(f"[S{i}] {h.text}")
            citations.append(
                {
                    "marker": f"S{i}",
                    "id": h.id,
                    "kind": h.source,
                    "snippet": h.text,
                    "score": h.score,
                }
            )

        user_prompt = (
            f"User message: {req.message}\n\nSources from memory:\n"
            + "\n".join(sources_block)
        )
        reply = await self.ai.complete_text(CHAT_SYSTEM_PROMPT, user_prompt)

        used_citations = [
            c for c in citations if f"[{c['marker']}]" in reply
        ]
        if not used_citations and citations:
            used_citations = citations

        return ChatResponse(
            reply=reply,
            citations=used_citations,
            session_id=session_id,
            trace_id=trace_id,
        )
