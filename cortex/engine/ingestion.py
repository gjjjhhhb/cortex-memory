from datetime import datetime, timezone
from typing import Any
import uuid

from cortex.domain.formatting import format_scope, normalize_entity_id
from cortex.domain.models import (
    RememberRequest,
    RememberResponse,
    RememberBatchRequest,
    RememberBatchResponse,
    ExtractionDiff,
    ForgetRequest,
    ForgetResponse,
    ResolveUncertaintyRequest,
    ConsolidateRequest,
    ConsolidateResponse,
)
from cortex.domain.scoring import cosine_similarity
from cortex.infra.store import CortexStore


class IngestionEngine:
    def __init__(self, store: CortexStore, ai: Any):
        self.store = store
        self.ai = ai

    async def remember(self, context_id: str, req: RememberRequest) -> RememberResponse:
        await self.store.use_context(context_id)
        now = datetime.now(timezone.utc)
        anchor_time = req.observed_at or now
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        scope_str = format_scope(req.scopes or req.scope)

        if req.infer == "preview":
            return await self._remember_preview(
                context_id, req, anchor_time, trace_id, scope_str, now
            )

        session_id = req.session_id or f"sess_{uuid.uuid4().hex[:10]}"
        session_record_id = f"session:{session_id}"
        await self.store.upsert_session(session_record_id, scope_str)

        turn_id = f"turn_{uuid.uuid4().hex[:10]}"
        turn_record_id = f"turn:{turn_id}"
        text_content = req.text or ""
        await self.store.create(
            "turn",
            {
                "id": turn_record_id,
                "session": session_record_id,
                "role": req.role,
                "content": text_content,
                "created_at": now.isoformat(),
            },
        )

        if req.infer == "none":
            return await self._remember_literal(
                text_content,
                scope_str,
                req.labels,
                anchor_time,
                turn_record_id,
                session_id,
                turn_id,
                trace_id,
            )

        if req.infer == "triples":
            return await self._remember_triples(
                req,
                scope_str,
                anchor_time,
                turn_record_id,
                session_id,
                turn_id,
                trace_id,
                now,
            )

        return await self._remember_full(
            text_content,
            anchor_time,
            scope_str,
            req,
            turn_record_id,
            session_id,
            turn_id,
            trace_id,
            now,
        )

    async def _remember_preview(
        self,
        context_id: str,
        req: RememberRequest,
        anchor_time: datetime,
        trace_id: str,
        scope_str: str,
        now: datetime,
    ) -> RememberResponse:
        text_content = req.text or ""
        active_facts = await self.store.fetch_active_attributes(now.isoformat())
        facts_for_llm = [
            {k: v for k, v in f.items() if k != "embedding"} for f in active_facts[:20]
        ]
        active_instructions = await self.store.fetch_active_instructions()
        norm_insts = [
            {**inst, "id": str(inst.get("id", ""))} for inst in active_instructions
        ]

        payload = await self.ai.extract_and_reconcile(
            input_text=text_content,
            anchor_time=anchor_time,
            existing_facts=facts_for_llm,
            existing_instructions=norm_insts,
        )

        diff = ExtractionDiff()
        for ent in payload.entities:
            diff.entities.append(
                {
                    "id": normalize_entity_id(ent.name),
                    "name": ent.name,
                    "entity_type": ent.entity_type,
                }
            )
        for op in payload.fact_operations:
            diff.attributes.append(
                {
                    "entity": normalize_entity_id(op.entity),
                    "key": op.key,
                    "value": op.value,
                    "statement": op.statement,
                    "memory_category": op.memory_category or "knowledge",
                }
            )
            if op.op == "SUPERSEDE" and op.target_attribute_id:
                diff.superseded.append(op.target_attribute_id)
        for rel in payload.relations:
            diff.relations.append(
                {
                    "source_entity": rel.source,
                    "target_entity": rel.target,
                    "predicate": rel.predicate,
                    "memory_category": rel.memory_category or "knowledge",
                }
            )
        for inst in payload.instructions:
            diff.instructions.append({"rule": inst.rule, "active": True})
            if inst.op == "SUPERSEDE" and inst.target_instruction_id:
                diff.superseded.append(inst.target_instruction_id)
        for unc in payload.uncertainties:
            diff.uncertainties.append(
                {"description": unc.description, "entity_name": unc.entity_name}
            )

        await self.store.record_trace(trace_id, query=text_content, mode="preview")
        return RememberResponse(
            mode="preview",
            session_id=req.session_id or f"sess_{uuid.uuid4().hex[:10]}",
            turn_id=f"turn_{uuid.uuid4().hex[:10]}",
            preview=diff,
            trace_id=trace_id,
        )

    async def _remember_literal(
        self,
        text_content: str,
        scope_str: str,
        labels: list[str],
        anchor_time: datetime,
        turn_record_id: str,
        session_id: str,
        turn_id: str,
        trace_id: str,
    ) -> RememberResponse:
        emb = await self.ai.get_embedding(text_content)
        chunk_res = await self.store.create(
            "memory_chunk",
            {
                "text": text_content,
                "scope": scope_str,
                "labels": labels,
                "embedding": emb,
                "created_at": anchor_time.isoformat(),
                "source_turn": turn_record_id,
            },
        )
        await self.store.record_trace(trace_id, query=text_content, mode="none")
        return RememberResponse(
            mode="none",
            session_id=session_id,
            turn_id=turn_id,
            chunk_id=str(chunk_res.get("id", "")),
            trace_id=trace_id,
        )

    async def _remember_triples(
        self,
        req: RememberRequest,
        scope_str: str,
        anchor_time: datetime,
        turn_record_id: str,
        session_id: str,
        turn_id: str,
        trace_id: str,
        now: datetime,
    ) -> RememberResponse:
        diff = ExtractionDiff()
        triples_raw = req.triples or []

        if isinstance(triples_raw, dict):
            await self._process_triples_dict(
                triples_raw, scope_str, anchor_time, turn_record_id, req, now, diff
            )
        else:
            await self._process_triples_list(
                triples_raw, scope_str, anchor_time, turn_record_id, req, now, diff
            )

        await self.store.record_trace(
            trace_id, query=req.text or "triples", mode="triples"
        )
        return RememberResponse(
            mode="triples",
            session_id=session_id,
            turn_id=turn_id,
            extraction=diff,
            trace_id=trace_id,
        )

    async def _process_triples_dict(
        self,
        data: dict[str, Any],
        scope_str: str,
        anchor_time: datetime,
        turn_record_id: str,
        req: RememberRequest,
        now: datetime,
        diff: ExtractionDiff,
    ):
        for e in data.get("entities", []):
            name = str(e.get("name") or e.get("entity_name") or "Unknown")
            etype = e.get("entity_type") or e.get("type") or "concept"
            ent_id = normalize_entity_id(name)
            await self.store.upsert_entity(ent_id, name, etype, "", scope_str)
            diff.entities.append({"id": ent_id, "name": name, "entity_type": etype})

        for a in data.get("attributes", []):
            ename = a.get("entity_name") or a.get("entity") or "Unknown"
            if isinstance(ename, dict):
                ename = ename.get("name", "Unknown")
            ename_str = str(ename)
            ent_id = normalize_entity_id(ename_str)
            etype = a.get("entity_type") or "concept"
            await self.store.upsert_entity(ent_id, ename_str, etype, "", scope_str)
            diff.entities.append(
                {"id": ent_id, "name": ename_str, "entity_type": etype}
            )

            k = a.get("key") or a.get("name") or "fact"
            v = a.get("value")
            stmt = a.get("statement") or f"{ename_str} {k} is {v}."
            stmt_emb = await self.ai.get_embedding(stmt)
            new_attr = {
                "entity": ent_id,
                "key": str(k),
                "value": str(v),
                "statement": stmt,
                "memory_category": a.get("memory_category") or "knowledge",
                "valid_from": (req.observed_at or now).isoformat(),
                "valid_until": None,
                "created_at": anchor_time.isoformat(),
                "superseded_by": None,
                "source_turn": turn_record_id,
                "source_span": None,
                "confidence": 1.0,
                "scope": scope_str,
                "labels": req.labels,
                "embedding": stmt_emb,
            }
            res_attr = await self.store.create("attribute", new_attr)
            clean_attr = {k: v for k, v in new_attr.items() if k != "embedding"}
            clean_attr["id"] = str(res_attr.get("id", ""))
            diff.attributes.append(clean_attr)

        for r in data.get("relations", []):
            s_name = str(
                r.get("source_name")
                or r.get("source_entity")
                or r.get("source")
                or "Unknown"
            )
            t_name = str(
                r.get("target_name")
                or r.get("target_entity")
                or r.get("target")
                or "Unknown"
            )
            rel_type = str(r.get("relation_type") or r.get("predicate") or "related_to")
            rel_data = {
                "source_entity": s_name,
                "target_entity": t_name,
                "predicate": rel_type,
                "memory_category": r.get("memory_category") or "knowledge",
                "valid_from": (req.observed_at or now).isoformat(),
                "valid_until": None,
                "created_at": anchor_time.isoformat(),
                "scope": scope_str,
            }
            res_rel = await self.store.create("relation", rel_data)
            rel_data["id"] = str(res_rel.get("id", ""))
            diff.relations.append(rel_data)

    async def _process_triples_list(
        self,
        items: list[Any],
        scope_str: str,
        anchor_time: datetime,
        turn_record_id: str,
        req: RememberRequest,
        now: datetime,
        diff: ExtractionDiff,
    ):
        for t in items:
            ent_val = t.get("entity")
            if isinstance(ent_val, dict):
                ent_name = ent_val.get("name", "Unknown")
                ent_type = ent_val.get("type", "concept")
            else:
                ent_name = str(ent_val or "Unknown")
                ent_type = "concept"

            ent_id = normalize_entity_id(ent_name)
            await self.store.upsert_entity(ent_id, ent_name, ent_type, "", scope_str)
            diff.entities.append(
                {"id": ent_id, "name": ent_name, "entity_type": ent_type}
            )

            if t.get("key") and t.get("value") is not None:
                stmt = (
                    t.get("statement")
                    or f"{ent_name} {t.get('key')} is {t.get('value')}."
                )
                stmt_emb = await self.ai.get_embedding(stmt)
                new_attr = {
                    "entity": ent_id,
                    "key": str(t.get("key")),
                    "value": str(t.get("value")),
                    "statement": stmt,
                    "memory_category": t.get("memory_category") or "knowledge",
                    "valid_from": (req.observed_at or now).isoformat(),
                    "valid_until": None,
                    "created_at": anchor_time.isoformat(),
                    "superseded_by": None,
                    "source_turn": turn_record_id,
                    "source_span": None,
                    "confidence": 1.0,
                    "scope": scope_str,
                    "labels": req.labels,
                    "embedding": stmt_emb,
                }
                res_attr = await self.store.create("attribute", new_attr)
                clean_attr = {k: v for k, v in new_attr.items() if k != "embedding"}
                clean_attr["id"] = str(res_attr.get("id", ""))
                diff.attributes.append(clean_attr)

            if t.get("target"):
                rel_data = {
                    "source_entity": ent_name,
                    "target_entity": str(t.get("target")),
                    "predicate": str(
                        t.get("predicate") or t.get("key") or "related_to"
                    ),
                    "memory_category": t.get("memory_category") or "knowledge",
                    "valid_from": (req.observed_at or now).isoformat(),
                    "valid_until": None,
                    "created_at": anchor_time.isoformat(),
                    "scope": scope_str,
                }
                res_rel = await self.store.create("relation", rel_data)
                rel_data["id"] = str(res_rel.get("id", ""))
                diff.relations.append(rel_data)

    async def _remember_full(
        self,
        text_content: str,
        anchor_time: datetime,
        scope_str: str,
        req: RememberRequest,
        turn_record_id: str,
        session_id: str,
        turn_id: str,
        trace_id: str,
        now: datetime,
    ) -> RememberResponse:
        active_facts = await self.store.fetch_active_attributes(now.isoformat())
        if len(active_facts) > 20:
            input_emb = await self.ai.get_embedding(text_content)

            def score_fact(f: dict[str, Any]) -> float:
                emb = f.get("embedding")
                return (
                    cosine_similarity(input_emb, emb) if isinstance(emb, list) else 0.0
                )

            active_facts.sort(key=score_fact, reverse=True)
            candidate_facts = active_facts[:20]
        else:
            candidate_facts = active_facts

        facts_for_llm = [
            {k: v for k, v in f.items() if k != "embedding"} for f in candidate_facts
        ]
        active_instructions = await self.store.fetch_active_instructions()
        norm_insts = [
            {**inst, "id": str(inst.get("id", ""))} for inst in active_instructions
        ]

        payload = await self.ai.extract_and_reconcile(
            input_text=text_content,
            anchor_time=anchor_time,
            existing_facts=facts_for_llm,
            existing_instructions=norm_insts,
        )

        diff = ExtractionDiff()
        entity_map: dict[str, str] = {}
        for ent in payload.entities:
            ent_id = normalize_entity_id(ent.name)
            entity_map[ent.name] = ent_id
            await self.store.upsert_entity(
                ent_id, ent.name, ent.entity_type, ent.summary or "", scope_str
            )
            diff.entities.append(
                {"id": ent_id, "name": ent.name, "entity_type": ent.entity_type}
            )

        for op in payload.fact_operations:
            ent_id = entity_map.get(op.entity) or normalize_entity_id(op.entity)
            stmt_emb = await self.ai.get_embedding(op.statement)
            span = None
            if op.source_quote and op.source_quote in text_content:
                start_idx = text_content.find(op.source_quote)
                span = [start_idx, start_idx + len(op.source_quote)]

            v_from = op.valid_from or anchor_time
            new_attr = {
                "entity": ent_id,
                "key": op.key,
                "value": op.value,
                "statement": op.statement,
                "memory_category": op.memory_category
                or req.memory_category
                or "knowledge",
                "valid_from": v_from.isoformat(),
                "valid_until": op.valid_until.isoformat() if op.valid_until else None,
                "created_at": anchor_time.isoformat(),
                "superseded_by": None,
                "source_turn": turn_record_id,
                "source_span": span,
                "confidence": op.confidence,
                "scope": scope_str,
                "labels": req.labels,
                "embedding": stmt_emb,
            }
            res = await self.store.create("attribute", new_attr)
            new_attr_id = str(res.get("id", ""))
            clean_attr = {k: v for k, v in new_attr.items() if k != "embedding"}
            clean_attr["id"] = new_attr_id
            diff.attributes.append(clean_attr)

            if op.op == "SUPERSEDE" and op.target_attribute_id:
                await self.store.supersede_attribute(
                    op.target_attribute_id, new_attr_id, v_from.isoformat()
                )
                diff.superseded.append(op.target_attribute_id)

        for rel in payload.relations:
            rel_data = {
                "source_entity": rel.source,
                "target_entity": rel.target,
                "predicate": rel.predicate,
                "memory_category": rel.memory_category or "knowledge",
                "valid_from": (rel.valid_from or anchor_time).isoformat(),
                "valid_until": rel.valid_until.isoformat() if rel.valid_until else None,
                "created_at": anchor_time.isoformat(),
                "scope": scope_str,
            }
            res_rel = await self.store.create("relation", rel_data)
            rel_data["id"] = str(res_rel.get("id", ""))
            diff.relations.append(rel_data)

        for inst in payload.instructions:
            inst_data = {
                "rule": inst.rule,
                "scope": scope_str,
                "active": True,
                "created_at": anchor_time.isoformat(),
                "source_turn": turn_record_id,
            }
            res_inst = await self.store.create("instruction", inst_data)
            inst_data["id"] = str(res_inst.get("id", ""))
            diff.instructions.append(inst_data)

            if inst.op == "SUPERSEDE" and inst.target_instruction_id:
                await self.store.deactivate_instruction(inst.target_instruction_id)
                diff.superseded.append(inst.target_instruction_id)

        for unc in payload.uncertainties:
            unc_data = {
                "description": unc.description,
                "entity_name": unc.entity_name,
                "resolved": False,
                "created_at": anchor_time.isoformat(),
                "scope": scope_str,
            }
            await self.store.create("uncertainty", unc_data)
            diff.uncertainties.append(unc_data)

        await self.store.record_trace(trace_id, query=text_content, mode="full")
        return RememberResponse(
            mode="full",
            session_id=session_id,
            turn_id=turn_id,
            extraction=diff,
            trace_id=trace_id,
        )

    async def remember_batch(
        self, context_id: str, req: RememberBatchRequest
    ) -> RememberBatchResponse:
        await self.store.use_context(context_id)
        now = datetime.now(timezone.utc)
        session_id = req.session_id or f"sess_{uuid.uuid4().hex[:10]}"
        session_record_id = f"session:{session_id}"
        scope_str = format_scope(req.scopes)
        await self.store.upsert_session(session_record_id, scope_str)

        turn_ids: list[str] = []
        msgs = req.messages or req.items or []
        for msg in msgs:
            tid = f"turn_{uuid.uuid4().hex[:10]}"
            t_rec = f"turn:{tid}"
            await self.store.create(
                "turn",
                {
                    "id": t_rec,
                    "session": session_record_id,
                    "role": msg.role,
                    "content": msg.get_content(),
                    "created_at": (msg.ts or now).isoformat(),
                },
            )
            turn_ids.append(tid)

        if req.extract == "per_message":
            extractions: list[ExtractionDiff] = []
            for msg in msgs:
                rem_req = RememberRequest(
                    text=msg.get_content(),
                    role=msg.role,
                    session_id=session_id,
                    scopes=req.scopes,
                    observed_at=msg.ts,
                    infer="full",
                )
                res = await self.remember(context_id, rem_req)
                if res.extraction:
                    extractions.append(res.extraction)
            return RememberBatchResponse(
                session_id=session_id, turn_ids=turn_ids, extractions=extractions
            )

        transcript = "\n".join(f"[{m.role}]: {m.get_content()}" for m in msgs)
        last_ts = msgs[-1].ts if msgs else now
        rem_req = RememberRequest(
            text=transcript,
            role="user",
            session_id=session_id,
            scopes=req.scopes,
            observed_at=last_ts,
            infer="full",
        )
        res = await self.remember(context_id, rem_req)
        try:
            await self.store.query(
                "DELETE turn WHERE content = $c AND session = $sess;",
                {"c": transcript, "sess": session_record_id},
            )
        except Exception:
            pass
        return RememberBatchResponse(
            session_id=session_id,
            turn_ids=turn_ids,
            extractions=[res.extraction] if res.extraction else [],
        )

    async def forget(self, context_id: str, req: ForgetRequest) -> ForgetResponse:
        await self.store.use_context(context_id)
        now = datetime.now(timezone.utc)
        query_lower = req.query.lower()

        attrs = await self.store.query(
            "SELECT id, statement FROM attribute WHERE valid_until IS NONE;"
        )
        matched_ids = [
            a["id"] for a in attrs if query_lower in a.get("statement", "").lower()
        ]

        chunks = await self.store.query("SELECT id, text FROM memory_chunk;")
        matched_chunk_ids = [
            c["id"] for c in chunks if query_lower in c.get("text", "").lower()
        ]

        total_matches = len(matched_ids) + len(matched_chunk_ids)

        if req.dry_run:
            matches_list = [{"id": str(i)} for i in (matched_ids + matched_chunk_ids)]
            return ForgetResponse(
                deleted=total_matches,
                trace_id=f"tr_{uuid.uuid4().hex[:12]}",
                dry_run=True,
                matches=matches_list,
            )

        deleted_count = 0
        for aid in matched_ids:
            if req.purge:
                await self.store.query(f"DELETE {aid};")
            else:
                await self.store.query(
                    f"UPDATE {aid} SET valid_until = $now;", {"now": now.isoformat()}
                )
            deleted_count += 1

        for cid in matched_chunk_ids:
            await self.store.query(f"DELETE {cid};")
            deleted_count += 1

        return ForgetResponse(
            deleted=deleted_count, trace_id=f"tr_{uuid.uuid4().hex[:12]}"
        )

    async def resolve_uncertainty(
        self, context_id: str, uncertainty_id: str, req: ResolveUncertaintyRequest
    ) -> dict[str, Any]:
        await self.store.use_context(context_id)
        u_rec = (
            uncertainty_id
            if str(uncertainty_id).startswith("uncertainty:")
            else f"uncertainty:{uncertainty_id}"
        )
        recs = await self.store.query(f"SELECT * FROM {u_rec};")
        if not recs:
            raise ValueError(f"Uncertainty {uncertainty_id} not found")
        u = recs[0]

        now_iso = datetime.now(timezone.utc).isoformat()
        await self.store.query(f"UPDATE {u_rec} SET resolved = true;")
        u["resolved"] = True

        ent_name = u.get("entity_name")
        if ent_name and req.accepted_value:
            ent_id = normalize_entity_id(str(ent_name))
            stmt = req.note or f"{ent_name}: {req.accepted_value}"
            emb = await self.ai.get_embedding(stmt)
            await self.store.create(
                "attribute",
                {
                    "entity": ent_id,
                    "key": "resolved_fact",
                    "value": req.accepted_value,
                    "statement": stmt,
                    "memory_category": "knowledge",
                    "valid_from": now_iso,
                    "valid_until": None,
                    "created_at": now_iso,
                    "superseded_by": None,
                    "confidence": 1.0,
                    "scope": u.get("scope", "global"),
                    "embedding": emb,
                },
            )
        return {k: str(v) if hasattr(v, "table_name") else v for k, v in u.items()}

    async def expire_context(self, context_id: str) -> dict[str, int]:
        await self.store.use_context(context_id)
        now_iso = datetime.now(timezone.utc).isoformat()
        active_ctx = await self.store.query(
            "SELECT id FROM attribute WHERE memory_category = 'context' AND valid_until IS NONE;"
        )
        count = len(active_ctx)
        for c in active_ctx:
            cid = c.get("id")
            await self.store.query(
                f"UPDATE {cid} SET valid_until = $now;", {"now": now_iso}
            )
        return {"expired": count, "expired_count": count, "expiredCount": count}

    async def consolidate(
        self, context_id: str, req: ConsolidateRequest
    ) -> ConsolidateResponse:
        await self.store.use_context(context_id)
        now = datetime.now(timezone.utc)
        trace_id = f"tr_{uuid.uuid4().hex[:12]}"
        active_attrs = await self.store.query(
            "SELECT id, entity, key, value, created_at FROM attribute WHERE valid_until IS NONE AND superseded_by IS NONE;"
        )
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for a in active_attrs:
            k = (
                str(a.get("entity", "")).lower(),
                str(a.get("key", "")).lower(),
                str(a.get("value", "")).strip().lower(),
            )
            groups.setdefault(k, []).append(a)

        decisions: list[dict[str, Any]] = []
        merged_count = 0
        for grp_key, items in groups.items():
            if len(items) > 1:
                items_sorted = sorted(
                    items, key=lambda x: str(x.get("created_at", "")), reverse=True
                )
                newest = items_sorted[0]
                newest_id = str(newest.get("id", ""))
                decisions.append(
                    {
                        "decision": "keep",
                        "id": newest_id,
                        "key": grp_key[1],
                        "value": grp_key[2],
                    }
                )
                for dup in items_sorted[1:]:
                    dup_id = str(dup.get("id", ""))
                    decisions.append(
                        {
                            "decision": "supersede",
                            "id": dup_id,
                            "superseded_by": newest_id,
                        }
                    )
                    merged_count += 1
                    if not req.dry_run:
                        await self.store.query(
                            f"UPDATE {dup_id} SET valid_until = $now, superseded_by = $newest_id;",
                            {"now": now.isoformat(), "newest_id": newest_id},
                        )
        return ConsolidateResponse(
            decisions=decisions, merged_count=merged_count, trace_id=trace_id
        )
