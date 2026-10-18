import time
import uuid
from datetime import datetime, timezone
from typing import Any

from cortex.domain.formatting import clean_record, build_markdown_context
from cortex.domain.models import (
    RecallRequest,
    RecallResponse,
    RecallHit,
    TraceInfo,
    ContextRequest,
    ContextQueryResponse,
    LookupRequest,
    LookupResponse,
)
from cortex.domain.scoring import (
    rank_candidates,
    match_lens,
    is_temporally_valid,
    parse_iso,
)
from cortex.infra.store import CortexStore


from cortex.engine.cognitive import CognitiveEngine
from cortex.engine.graph import GraphEngine


class RetrievalEngine:
    def __init__(self, store: CortexStore, ai: Any):
        self.store = store
        self.ai = ai
        self.cognitive = CognitiveEngine(store=store, ai=ai, retrieval=self)
        self.graph = GraphEngine(store=store)

    async def recall(self, context_id: str, req: RecallRequest) -> RecallResponse:
        t0 = time.perf_counter()
        await self.store.use_context(context_id)
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"

        now = datetime.now(timezone.utc)
        if req.as_of:
            attrs = await self.store.query("SELECT * FROM attribute;")
        else:
            attrs = await self.store.query(
                "SELECT * FROM attribute WHERE (valid_until IS NONE OR valid_until > $now) AND superseded_by IS NONE;",
                {"now": now.isoformat()},
            )
        chunks = await self.store.query("SELECT * FROM memory_chunk;")

        candidates: list[dict[str, Any]] = []
        for a in attrs:
            a["_source_type"] = "attribute"
            a["_text"] = a.get("statement") or ""
            candidates.append(a)
        for c in chunks:
            c["_source_type"] = "memory_chunk"
            c["_text"] = c.get("text") or ""
            candidates.append(c)

        filtered: list[dict[str, Any]] = []
        for cand in candidates:
            if req.include:
                if (
                    "facts" in req.include
                    and "passages" not in req.include
                    and cand["_source_type"] != "attribute"
                ):
                    continue
                if (
                    "passages" in req.include
                    and "facts" not in req.include
                    and cand["_source_type"] != "memory_chunk"
                ):
                    continue

            if req.labels:
                cand_labels = cand.get("labels") or []
                if not any(lbl in cand_labels for lbl in req.labels):
                    continue

            cand_scope = cand.get("scope") or "global"
            if req.lens and not match_lens(cand_scope, req.lens):
                continue

            if not is_temporally_valid(
                cand.get("created_at"),
                cand.get("valid_from"),
                cand.get("valid_until"),
                cand.get("superseded_by"),
                req.as_of,
                now,
            ):
                continue

            filtered.append(cand)

        query_emb = await self.ai.get_embedding(req.query)
        query_tokens = set(req.query.lower().split())
        hits = rank_candidates(filtered, query_emb, query_tokens, req.mode, req.k)

        context_hits: list[RecallHit] = []
        entity_refs = set()
        for h in hits:
            if h.entity_name:
                entity_refs.add(h.entity_name.lower())
                if h.entity_name.startswith("entity:"):
                    entity_refs.add(h.entity_name.split(":", 1)[1].lower())
                else:
                    entity_refs.add(f"entity:{h.entity_name.lower()}")
            if h.key:
                entity_refs.add(h.key.lower())

        if entity_refs:
            relations = await self.store.query("SELECT * FROM relation;")
            for rel in relations:
                src = str(rel.get("source_entity", "")).lower()
                tgt = str(rel.get("target_entity", "")).lower()
                if any(
                    r in [src, tgt, f"entity:{src}", f"entity:{tgt}"]
                    for r in entity_refs
                ):
                    rel_text = f"{rel.get('source_entity')} {rel.get('predicate')} {rel.get('target_entity')}"
                    context_hits.append(
                        RecallHit(
                            id=str(rel.get("id", "")),
                            text=rel_text,
                            statement=rel_text,
                            score=1.0,
                            source="entity",
                            memory_category=rel.get("memory_category", "knowledge"),
                            entity_name=rel.get("source_entity"),
                            key=rel.get("predicate"),
                            value=rel.get("target_entity"),
                            valid_from=parse_iso(rel.get("valid_from")),
                            valid_until=parse_iso(rel.get("valid_until")),
                            occurred_at=parse_iso(rel.get("created_at")),
                        )
                    )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        await self.store.record_trace(
            trace_id,
            query=req.query,
            mode=req.mode or "hybrid",
            tier="hybrid",
            query_ms=elapsed_ms,
            hits_count=len(hits),
        )

        return RecallResponse(
            tier="hybrid",
            hits=hits,
            context_hits=context_hits,
            query_ms=elapsed_ms,
            trace=TraceInfo(trace_id=trace_id),
            trace_id=trace_id,
        )

    async def query_context(
        self, context_id: str, req: ContextRequest
    ) -> ContextQueryResponse:
        t0 = time.perf_counter()
        await self.store.use_context(context_id)
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"

        instructions = await self.store.fetch_active_instructions()
        identities = await self.store.query(
            "SELECT * FROM attribute WHERE memory_category = 'identity' AND valid_until IS NONE;"
        )

        recall_res = await self.recall(
            context_id, RecallRequest(query=req.query, k=req.k, as_of=req.as_of)
        )
        context_str = build_markdown_context(
            instructions, identities, [], recall_res.hits
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        await self.store.record_trace(
            trace_id,
            query=req.query,
            mode="context",
            tier="hybrid",
            query_ms=elapsed_ms,
            hits_count=len(recall_res.hits),
        )

        return ContextQueryResponse(
            context=context_str, tier="hybrid", query_ms=elapsed_ms, trace_id=trace_id
        )

    async def inspect(self, context_id: str, ref: str) -> dict[str, Any]:
        await self.store.use_context(context_id)
        if ref.startswith("entity:"):
            ent_body = ref[len("entity:") :]
            ent_name = ent_body.split("/", 1)[1] if "/" in ent_body else ent_body
            recs = await self.store.query(
                "SELECT * FROM entity WHERE name = $name;", {"name": ent_name}
            )
            if not recs:
                norm_id = f"entity:{ent_name.lower().replace(' ', '_')}"
                recs = await self.store.query(f"SELECT * FROM {norm_id};")
            if recs:
                ent = clean_record(recs[0])
                attrs = await self.store.query(
                    "SELECT * FROM attribute WHERE entity = $ent AND valid_until IS NONE;",
                    {"ent": str(ent.get("id"))},
                )
                ent["attributes"] = [clean_record(a) for a in attrs]
                return ent

        target_ref = ref if ":" in ref else f"trace:{ref}"
        try:
            res = await self.store.query(f"SELECT * FROM {target_ref};")
            if res:
                return clean_record(res[0])
        except Exception:
            pass
        return {}

    async def lookup(self, context_id: str, req: LookupRequest) -> LookupResponse:
        await self.store.use_context(context_id)
        target = req.subject or req.query
        if "/" in target:
            parts = target.split("/", 1)
            req_type, req_name = parts[0], parts[1]
        else:
            req_type, req_name = req.entity_type, target

        name_clean = req_name.strip().lower()
        all_entities = await self.store.query("SELECT * FROM entity;")
        matched_entities: list[dict[str, Any]] = []
        best_score = 0.0
        best_ent = None

        for e in all_entities:
            e_name = str(e.get("name", "")).lower()
            e_type = str(e.get("entity_type", "")).lower()
            if req_type and e_type != req_type.lower():
                continue
            score = (
                1.0
                if e_name == name_clean
                else (0.7 if (name_clean in e_name or e_name in name_clean) else 0.0)
            )
            if score > 0:
                e_clean = clean_record(e)
                e_clean["_match_score"] = score
                matched_entities.append(e_clean)
                if score > best_score:
                    best_score = score
                    best_ent = e_clean

        matched_entities.sort(key=lambda x: x.get("_match_score", 0), reverse=True)
        facts: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        uncertainties: list[dict[str, Any]] = []

        ent_ids = [str(e.get("id")) for e in matched_entities]
        ent_names = [str(e.get("name")).lower() for e in matched_entities]

        if ent_ids:
            all_attrs = await self.store.query(
                "SELECT * FROM attribute WHERE valid_until IS NONE AND superseded_by IS NONE;"
            )
            for a in all_attrs:
                if str(a.get("entity")) in ent_ids:
                    facts.append(clean_record(a))
                    if len(facts) >= req.fact_limit:
                        break

            all_rels = await self.store.query("SELECT * FROM relation;")
            for r in all_rels:
                src = str(r.get("source_entity", "")).lower()
                tgt = str(r.get("target_entity", "")).lower()
                if any(n in (src, tgt) for n in ent_names):
                    relations.append(clean_record(r))
                    if len(relations) >= req.relation_limit:
                        break

            all_unc = await self.store.query(
                "SELECT * FROM uncertainty WHERE resolved = false;"
            )
            for u in all_unc:
                if str(u.get("entity_name", "")).lower() in ent_names:
                    uncertainties.append(clean_record(u))
                    if len(uncertainties) >= req.uncertainty_limit:
                        break

        resolution = {
            "status": "resolved" if best_score >= 0.7 else "unresolved",
            "confidence": best_score,
            "subject": f"{best_ent.get('entity_type', 'concept')}/{best_ent.get('name')}"
            if best_ent
            else target,
        }
        coverage = {
            "entities": len(matched_entities),
            "facts": len(facts),
            "relations": len(relations),
            "uncertainty": len(uncertainties),
        }
        return LookupResponse(
            resolution=resolution,
            entities=matched_entities,
            facts=facts,
            relations=relations,
            uncertainty=uncertainties,
            coverage=coverage,
        )

    async def reflect(self, context_id: str, req: Any, pipeline: Any = None) -> Any:
        return await self.cognitive.reflect(context_id, req, pipeline=pipeline)

    async def chat(self, context_id: str, req: Any, pipeline: Any = None) -> Any:
        return await self.cognitive.chat(context_id, req, pipeline=pipeline)

    async def get_profile(self, context_id: str) -> dict[str, Any]:
        return await self.graph.get_profile(context_id)

    async def list_entities(
        self,
        context_id: str,
        q: str | None = None,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return await self.graph.list_entities(
            context_id, q=q, entity_type=entity_type, limit=limit
        )

    async def get_entity_detail(
        self, context_id: str, entity_type: str, entity_name: str
    ) -> dict[str, Any]:
        return await self.graph.get_entity_detail(context_id, entity_type, entity_name)

    async def get_entity_history(
        self,
        context_id: str,
        entity_type: str,
        entity_name: str,
        key: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self.graph.get_entity_history(
            context_id, entity_type, entity_name, key=key
        )

    async def get_entity_neighbourhood(
        self, context_id: str, entity_type: str, entity_name: str
    ) -> dict[str, Any]:
        return await self.graph.get_entity_neighbourhood(
            context_id, entity_type, entity_name
        )

    async def delete_entity(
        self, context_id: str, entity_type: str, entity_name: str
    ) -> dict[str, Any]:
        return await self.graph.delete_entity(context_id, entity_type, entity_name)

    async def list_attributes(
        self, context_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        return await self.graph.list_attributes(context_id, limit=limit)

    async def list_relations(
        self, context_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        return await self.graph.list_relations(context_id, limit=limit)

    async def list_uncertainties(
        self, context_id: str, resolved: bool | None = None
    ) -> list[dict[str, Any]]:
        return await self.graph.list_uncertainties(context_id, resolved=resolved)

    async def list_traces(
        self, context_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        return await self.graph.list_traces(context_id, limit=limit)

    async def get_trace(self, context_id: str, trace_id: str) -> dict[str, Any]:
        return await self.graph.get_trace(context_id, trace_id)

    async def get_trace_stats(
        self, context_id: str, window_hours: int = 24
    ) -> dict[str, Any]:
        return await self.graph.get_trace_stats(context_id, window_hours=window_hours)

    async def get_session_turns(
        self, context_id: str, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        return await self.graph.get_session_turns(context_id, session_id, limit=limit)

    async def delete_session(self, context_id: str, session_id: str) -> dict[str, Any]:
        return await self.graph.delete_session(context_id, session_id)
