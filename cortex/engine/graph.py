from typing import Any
from cortex.domain.formatting import clean_record, normalize_entity_id
from cortex.infra.store import CortexStore


class GraphEngine:
    def __init__(self, store: CortexStore):
        self.store = store

    async def get_profile(self, context_id: str) -> dict[str, Any]:
        await self.store.use_context(context_id)
        identities = await self.store.query(
            "SELECT * FROM attribute WHERE memory_category = 'identity' AND valid_until IS NONE;"
        )
        instructions = await self.store.fetch_active_instructions()
        lines = []
        for i in identities:
            lines.append(f"- {i.get('statement') or i.get('value')}")
        for inst in instructions:
            lines.append(f"- Rule: {inst.get('rule')}")
        return {
            "identities": [clean_record(i) for i in identities],
            "instructions": [clean_record(inst) for inst in instructions],
            "markdown": "\n".join(lines),
        }

    async def list_entities(
        self,
        context_id: str,
        q: str | None = None,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        await self.store.use_context(context_id)
        entities = await self.store.query("SELECT * FROM entity;")
        out = []
        for e in entities:
            if (
                entity_type
                and str(e.get("entity_type", "")).lower() != entity_type.lower()
            ):
                continue
            if q and q.lower() not in str(e.get("name", "")).lower():
                continue
            out.append(clean_record(e))
            if len(out) >= limit:
                break
        return out

    async def get_entity_detail(
        self, context_id: str, entity_type: str, entity_name: str
    ) -> dict[str, Any]:
        await self.store.use_context(context_id)
        recs = await self.store.query(
            "SELECT * FROM entity WHERE name = $name;", {"name": entity_name}
        )
        if not recs:
            norm_id = normalize_entity_id(entity_name)
            recs = await self.store.query(f"SELECT * FROM {norm_id};")
        if not recs:
            return {}
        ent = clean_record(recs[0])
        ent_id = str(ent.get("id"))
        attrs = await self.store.query(
            "SELECT * FROM attribute WHERE entity = $ent AND valid_until IS NONE;",
            {"ent": ent_id},
        )
        ent["attributes"] = [clean_record(a) for a in attrs]
        return ent

    async def get_entity_history(
        self,
        context_id: str,
        entity_type: str,
        entity_name: str,
        key: str | None = None,
    ) -> list[dict[str, Any]]:
        await self.store.use_context(context_id)
        ent_id = normalize_entity_id(entity_name)
        attrs = await self.store.query(
            "SELECT * FROM attribute WHERE entity = $ent;", {"ent": ent_id}
        )
        if not attrs:
            erecs = await self.store.query(
                "SELECT * FROM entity WHERE name = $name;", {"name": entity_name}
            )
            if erecs:
                ent_id = str(erecs[0].get("id"))
                attrs = await self.store.query(
                    "SELECT * FROM attribute WHERE entity = $ent;", {"ent": ent_id}
                )

        out = []
        for a in attrs:
            if key and str(a.get("key", "")).lower() != key.lower():
                continue
            out.append(clean_record(a))
        out.sort(key=lambda x: str(x.get("created_at", "")), reverse=True)
        return out

    async def get_entity_neighbourhood(
        self, context_id: str, entity_type: str, entity_name: str
    ) -> dict[str, Any]:
        await self.store.use_context(context_id)
        name_lower = entity_name.lower()
        rels = await self.store.query("SELECT * FROM relation;")
        neighbours = []
        for r in rels:
            src = str(r.get("source_entity", "")).lower()
            tgt = str(r.get("target_entity", "")).lower()
            if src == name_lower:
                neighbours.append(
                    {
                        "neighbour": r.get("target_entity"),
                        "predicate": r.get("predicate"),
                        "direction": "outgoing",
                    }
                )
            elif tgt == name_lower:
                neighbours.append(
                    {
                        "neighbour": r.get("source_entity"),
                        "predicate": r.get("predicate"),
                        "direction": "incoming",
                    }
                )
        return {"entity": entity_name, "neighbours": neighbours}

    async def delete_entity(
        self, context_id: str, entity_type: str, entity_name: str
    ) -> dict[str, Any]:
        await self.store.use_context(context_id)
        ent_id = normalize_entity_id(entity_name)
        await self.store.query(f"DELETE {ent_id};")
        return {"deleted": True, "entity": entity_name, "status": "deleted"}

    async def list_attributes(
        self, context_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        await self.store.use_context(context_id)
        attrs = await self.store.query(
            "SELECT * FROM attribute WHERE valid_until IS NONE;"
        )
        return [clean_record(a) for a in attrs[:limit]]

    async def list_relations(
        self, context_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        await self.store.use_context(context_id)
        rels = await self.store.query("SELECT * FROM relation;")
        return [clean_record(r) for r in rels[:limit]]

    async def list_uncertainties(
        self, context_id: str, resolved: bool | None = None
    ) -> list[dict[str, Any]]:
        await self.store.use_context(context_id)
        if resolved is not None:
            uncs = await self.store.query(
                "SELECT * FROM uncertainty WHERE resolved = $res;", {"res": resolved}
            )
        else:
            uncs = await self.store.query("SELECT * FROM uncertainty;")
        return [clean_record(u) for u in uncs]

    async def list_traces(
        self, context_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        await self.store.use_context(context_id)
        traces = await self.store.query("SELECT * FROM trace ORDER BY created_at DESC;")
        return [clean_record(t) for t in traces[:limit]]

    async def get_trace(self, context_id: str, trace_id: str) -> dict[str, Any]:
        await self.store.use_context(context_id)
        t_rec = trace_id if trace_id.startswith("trace:") else f"trace:{trace_id}"
        traces = await self.store.query(f"SELECT * FROM {t_rec};")
        if traces:
            return clean_record(traces[0])
        return {}

    async def get_trace_stats(
        self, context_id: str, window_hours: int = 24
    ) -> dict[str, Any]:
        await self.store.use_context(context_id)
        traces = await self.store.query("SELECT query_ms, tier, mode FROM trace;")
        total = len(traces)
        if total == 0:
            return {
                "totalQueries": 0,
                "avgLatencyMs": 0.0,
                "p95LatencyMs": 0.0,
                "tierBreakdown": {},
                "cacheHitRatio": 0.0,
            }
        latencies = sorted(int(t.get("query_ms") or 0) for t in traces)
        avg_lat = sum(latencies) / total
        p95_lat = latencies[int(0.95 * total)] if total > 1 else latencies[0]

        tier_counts: dict[str, int] = {}
        cache_hits = 0
        for t in traces:
            tier = str(t.get("tier") or "hybrid")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if tier == "cache":
                cache_hits += 1

        return {
            "totalQueries": total,
            "totalTraces": total,
            "total_traces": total,
            "avgLatencyMs": round(avg_lat, 2),
            "p95LatencyMs": round(float(p95_lat), 2),
            "tierBreakdown": tier_counts,
            "cacheHitRatio": round(cache_hits / total, 3),
        }

    async def get_session_turns(
        self, context_id: str, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        await self.store.use_context(context_id)
        s_rec = (
            session_id if session_id.startswith("session:") else f"session:{session_id}"
        )
        turns = await self.store.query(
            "SELECT * FROM turn WHERE session = $sess ORDER BY created_at ASC;",
            {"sess": s_rec},
        )
        return [clean_record(t) for t in turns[:limit]]

    async def delete_session(self, context_id: str, session_id: str) -> dict[str, Any]:
        await self.store.use_context(context_id)
        s_rec = (
            session_id if session_id.startswith("session:") else f"session:{session_id}"
        )
        await self.store.query(f"DELETE {s_rec};")
        await self.store.query("DELETE turn WHERE session = $sess;", {"sess": s_rec})
        return {"deleted": True, "sessionId": session_id}
