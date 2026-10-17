import logging
from datetime import datetime, timezone
from typing import Any
from surrealdb import AsyncSurreal

from cortex.infra.schema import get_schema_ddl

logger = logging.getLogger(__name__)


def unwrap_query_result(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    if isinstance(raw, list):
        if len(raw) > 0 and isinstance(raw[0], dict) and "result" in raw[0]:
            return raw[0]["result"] or []
        if len(raw) > 0 and isinstance(raw[0], list):
            return raw[0]
        return raw
    if isinstance(raw, dict) and "result" in raw:
        return raw["result"] or []
    return []


class CortexStore:
    def __init__(
        self,
        url: str = "mem://",
        namespace: str = "cortex",
        dimension: int = 1536,
        database: str = "cortex",
        username: str | None = None,
        password: str | None = None,
    ):
        self.url = url
        self.namespace = namespace
        self.dimension = dimension
        self.database = database
        self.username = username
        self.password = password
        self.db: AsyncSurreal | None = None
        self._current_context: str | None = None

    async def connect(self):
        if not self.db:
            self.db = AsyncSurreal(self.url)
            if hasattr(self.db, "connect"):
                try:
                    await self.db.connect()
                except Exception:
                    pass
            if self.username and self.password:
                try:
                    await self.db.signin({"user": self.username, "pass": self.password})
                except Exception:
                    pass

    async def use_context(self, context_id: str):
        await self.connect()
        assert self.db is not None
        if self._current_context != context_id:
            await self.db.use(namespace=self.namespace, database=context_id)
            self._current_context = context_id
            ddl = get_schema_ddl(self.dimension)
            try:
                await self.db.query(ddl)
            except Exception as e:
                logger.debug(f"Schema init query result: {e}")

    async def query(
        self, sql: str, vars: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        assert self.db is not None
        try:
            res = await self.db.query(sql, vars or {})
            return unwrap_query_result(res)
        except Exception as e:
            logger.error(
                f"Error executing SurrealQL: {e} | Query: {sql} | Vars: {vars}"
            )
            raise e

    async def create(self, table: str, data: dict[str, Any]) -> dict[str, Any]:
        assert self.db is not None
        if table == "relation":
            sql = "CREATE relation CONTENT $data;"
            res = await self.db.query(sql, {"data": data})
            unwrapped = unwrap_query_result(res)
            if isinstance(unwrapped, list) and len(unwrapped) > 0:
                return unwrapped[0]
            return unwrapped if isinstance(unwrapped, dict) else {}
        res = await self.db.create(table, data)
        if isinstance(res, list) and len(res) > 0:
            return res[0]
        return res

    async def close(self):
        if self.db:
            try:
                await self.db.close()
            except Exception:
                pass
            self.db = None

    async def record_trace(
        self,
        trace_id: str,
        query: str,
        mode: str,
        tier: str = "direct",
        query_ms: int = 0,
        hits_count: int = 0,
    ):
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            await self.query(
                """UPSERT trace CONTENT {
                    id: $id,
                    query: $query,
                    mode: $mode,
                    tier: $tier,
                    query_ms: $query_ms,
                    hits_count: $hits_count,
                    created_at: $created_at
                };""",
                {
                    "id": f"trace:{trace_id}",
                    "query": (query or "")[:200],
                    "mode": mode,
                    "tier": tier,
                    "query_ms": query_ms,
                    "hits_count": hits_count,
                    "created_at": now_iso,
                },
            )
        except Exception:
            pass

    async def upsert_session(self, session_record_id: str, scope: str):
        await self.query(
            "UPSERT session CONTENT { id: $id, scope: $scope, created_at: time::now() };",
            {"id": session_record_id, "scope": scope},
        )

    async def upsert_entity(
        self,
        ent_id: str,
        name: str,
        entity_type: str,
        summary: str = "",
        scope: str = "global",
    ):
        await self.query(
            "UPSERT entity CONTENT { id: $id, name: $name, entity_type: $entity_type, summary: $summary, scope: $scope, aliases: [] };",
            {
                "id": ent_id,
                "name": name,
                "entity_type": entity_type,
                "summary": summary,
                "scope": scope,
            },
        )

    async def supersede_attribute(
        self, target_attribute_id: str, new_attr_id: str, valid_until: str
    ):
        await self.query(
            f"UPDATE {target_attribute_id} SET valid_until = $valid_until, superseded_by = $superseded_by;",
            {"valid_until": valid_until, "superseded_by": new_attr_id},
        )

    async def deactivate_instruction(self, instruction_id: str):
        await self.query(f"UPDATE {instruction_id} SET active = false;")

    async def fetch_active_instructions(self) -> list[dict[str, Any]]:
        return await self.query(
            "SELECT id, rule, scope FROM instruction WHERE active = true;"
        )

    async def fetch_active_attributes(self, now_iso: str) -> list[dict[str, Any]]:
        return await self.query(
            """SELECT id, entity.name as entity_name, key, value, statement, memory_category, embedding 
               FROM attribute 
               WHERE (valid_until IS NONE OR valid_until > $now) AND superseded_by IS NONE;""",
            {"now": now_iso},
        )

    async def count_records(self, table: str) -> int:
        res = await self.query(f"SELECT count() FROM {table} GROUP ALL;")
        return res[0].get("count", 0) if res else 0
