import json
import sys
from typing import Any

from cortex.domain.models import (
    ConsolidateRequest,
    ContextRequest,
    ForgetRequest,
    LookupRequest,
    RecallRequest,
    ReflectRequest,
    RememberRequest,
    ResolveUncertaintyRequest,
)

MCP_COMPACT_TOOLS = [
    {
        "name": "cortex",
        "description": (
            "Personal agent memory & knowledge graph. "
            "Core actions: 'remember' (save fact/text in query), 'recall' (search query), "
            "'context' (markdown prompt block for query), 'reflect' (synthesize insights), "
            "'lookup' (entity resolution & graph), 'forget' (delete matching query). "
            "Graph & ops actions: 'inspect', 'entities', 'history', 'neighbourhood', "
            "'uncertainty', 'resolve_uncertainty', 'consolidate', 'expire', 'profile', 'stats', 'help' (full docs)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "remember",
                        "recall",
                        "context",
                        "reflect",
                        "lookup",
                        "forget",
                        "inspect",
                        "entities",
                        "history",
                        "neighbourhood",
                        "uncertainty",
                        "resolve_uncertainty",
                        "consolidate",
                        "expire",
                        "profile",
                        "stats",
                        "help",
                    ],
                    "description": "Operation to execute.",
                },
                "query": {
                    "type": "string",
                    "description": "Text to remember, search query, entity name/ref, or help topic.",
                },
                "options": {
                    "type": "object",
                    "description": "Optional parameters (k, mode, infer, scope, persist, dry_run, purge, entity_type, key, accepted_value, strategy).",
                },
            },
            "required": ["action"],
        },
    }
]

MCP_TOOLS = [
    {
        "name": "remember",
        "description": "Store a new observation, user preference, identity detail or conversation turn with state-aware reconciliation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The natural language utterance to remember.",
                },
                "session_id": {"type": "string"},
                "scope": {"type": "string"},
                "infer": {
                    "type": "string",
                    "enum": ["full", "none", "preview", "triples"],
                    "default": "full",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "recall",
        "description": "Search memory and knowledge using hybrid Vector + BM25 retrieval.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to search memory for.",
                },
                "k": {"type": "integer", "default": 10},
                "mode": {
                    "type": "string",
                    "enum": ["hybrid", "vector", "bm25"],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "context",
        "description": "Assemble a structured markdown context block (instructions + profile + relevant memory) for prompt injection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "forget",
        "description": "Soft-delete or purge memories matching a natural language query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "purge": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "inspect",
        "description": "Inspect a specific memory record by ID (e.g. entity:user or attribute:id).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
            },
            "required": ["ref"],
        },
    },
    {
        "name": "reflect",
        "description": "Synthesize patterns, habits, contradictions and insights across stored memories.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic or question to reflect on.",
                },
                "persist": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to store synthesized reflections as new memories.",
                },
            },
            "required": ["query"],
        },
    },
]

HELP_DOCS = {
    "cortex": "Universal Agent Memory & Knowledge Graph Interface",
    "actions": {
        "remember": {
            "query": "Text to store",
            "options": {
                "infer": "full|none|preview|triples",
                "scope": "global/project",
                "session_id": "str",
            },
        },
        "recall": {
            "query": "Search query",
            "options": {"k": 10, "mode": "hybrid|vector|bm25", "lens": "str"},
        },
        "context": {"query": "Task or topic", "options": {"k": 10}},
        "reflect": {
            "query": "Topic to synthesize insights on",
            "options": {"persist": False},
        },
        "lookup": {
            "query": "Subject entity to resolve",
            "options": {"include": ["entities", "facts", "relations"]},
        },
        "forget": {
            "query": "Pattern to delete",
            "options": {"dry_run": False, "purge": False},
        },
        "inspect": {"query": "Record ID (e.g. entity:user or attribute:id)"},
        "entities": {
            "query": "Filter substring (optional)",
            "options": {"entity_type": "concept|person", "limit": 100},
        },
        "history": {
            "query": "Entity name",
            "options": {"entity_type": "concept", "key": "attribute_name"},
        },
        "neighbourhood": {
            "query": "Entity name",
            "options": {"entity_type": "concept"},
        },
        "uncertainty": {"options": {"resolved": None}},
        "resolve_uncertainty": {
            "query": "Uncertainty ID",
            "options": {"accepted_value": "value", "note": "optional note"},
        },
        "consolidate": {"options": {"strategy": "all"}},
        "expire": {"description": "Soft-expires transient context facts"},
        "profile": {"description": "Returns user identity attributes and active rules"},
        "stats": {"description": "Returns latency and trace statistics"},
        "help": {"description": "Shows this guide"},
    },
}


class McpServer:
    def __init__(
        self,
        pipeline: Any,
        retrieval: Any,
        context_id: str = "dev",
        mcp_mode: str = "compact",
    ):
        self.pipeline = pipeline
        self.retrieval = retrieval
        self.context_id = context_id
        self.mcp_mode = mcp_mode

    async def handle_request(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {},
                    },
                    "serverInfo": {"name": "cortex", "version": "0.5.0"},
                },
            }

        if msg_id is None or method in ("notifications/initialized", "initialized"):
            return None

        if method == "tools/list":
            tools = MCP_COMPACT_TOOLS if self.mcp_mode == "compact" else MCP_TOOLS
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}}

        if method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "resources": [
                        {
                            "uri": "cortex://profile",
                            "name": "User Profile & Active Rules",
                            "description": "Identity facts and operational behavioral instructions stored in Cortex.",
                            "mimeType": "application/json",
                        },
                        {
                            "uri": "cortex://stats",
                            "name": "Cortex Execution Statistics",
                            "description": "Aggregated execution trace stats and query latency metrics.",
                            "mimeType": "application/json",
                        },
                    ]
                },
            }

        if method == "resources/read":
            return await self._handle_resource_read(msg_id, params.get("uri", ""))

        if method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "prompts": [
                        {
                            "name": "memory-context",
                            "description": "Inject structured memory context block for a query or task directly into prompt.",
                            "arguments": [
                                {
                                    "name": "topic",
                                    "description": "The current task, question, or domain topic.",
                                    "required": True,
                                }
                            ],
                        }
                    ]
                },
            }

        if method == "prompts/get":
            return await self._handle_prompt_get(
                msg_id, params.get("name"), params.get("arguments", {})
            )

        if method == "tools/call":
            return await self._handle_tool_call(
                msg_id, params.get("name"), params.get("arguments", {})
            )

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }

    async def _handle_resource_read(self, msg_id: Any, uri: str) -> dict[str, Any]:
        if uri == "cortex://profile":
            prof = await self.retrieval.get_profile(self.context_id)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(prof, ensure_ascii=False, indent=2),
                        }
                    ]
                },
            }
        elif uri == "cortex://stats":
            stats = await self.retrieval.get_trace_stats(self.context_id)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(stats, ensure_ascii=False, indent=2),
                        }
                    ]
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32602, "message": f"Resource not found: {uri}"},
        }

    async def _handle_prompt_get(
        self, msg_id: Any, name: str | None, args: dict[str, Any]
    ) -> dict[str, Any]:
        if name == "memory-context":
            topic = args.get("topic", "")
            ctx_res = await self.retrieval.query_context(
                self.context_id, ContextRequest(query=topic)
            )
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "description": f"Cortex memory context for '{topic}'",
                    "messages": [
                        {
                            "role": "user",
                            "content": {"type": "text", "text": ctx_res.context},
                        }
                    ],
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32602, "message": f"Unknown prompt: {name}"},
        }

    async def _handle_tool_call(
        self, msg_id: Any, tool_name: str | None, args: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            if tool_name == "cortex":
                res_content = await self._dispatch_cortex_action(args)
            elif tool_name == "remember":
                req = RememberRequest(**args)
                res = await self.pipeline.remember(self.context_id, req)
                res_content = res.model_dump(mode="json", by_alias=True)
            elif tool_name == "recall":
                req = RecallRequest(**args)
                res = await self.retrieval.recall(self.context_id, req)
                res_content = res.model_dump(mode="json", by_alias=True)
            elif tool_name == "context":
                req = ContextRequest(**args)
                res = await self.retrieval.query_context(self.context_id, req)
                res_content = res.model_dump(mode="json", by_alias=True)
            elif tool_name == "forget":
                req = ForgetRequest(**args)
                res = await self.pipeline.forget(self.context_id, req)
                res_content = res.model_dump(mode="json", by_alias=True)
            elif tool_name == "inspect":
                ref = args.get("ref", "")
                res_content = await self.retrieval.inspect(self.context_id, ref)
            elif tool_name == "reflect":
                req = ReflectRequest(**args)
                res = await self.retrieval.reflect(
                    self.context_id, req, pipeline=self.pipeline
                )
                res_content = res.model_dump(mode="json", by_alias=True)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(res_content, ensure_ascii=False),
                        }
                    ],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True,
                },
            }

    async def _dispatch_cortex_action(self, args: dict[str, Any]) -> Any:
        action = args.get("action")
        query = args.get("query", "")
        options = args.get("options") or {}

        handlers = {
            "remember": self._action_remember,
            "recall": self._action_recall,
            "context": self._action_context,
            "reflect": self._action_reflect,
            "lookup": self._action_lookup,
            "forget": self._action_forget,
            "inspect": self._action_inspect,
            "entities": self._action_entities,
            "history": self._action_history,
            "neighbourhood": self._action_neighbourhood,
            "uncertainty": self._action_uncertainty,
            "resolve_uncertainty": self._action_resolve_uncertainty,
            "consolidate": self._action_consolidate,
            "expire": self._action_expire,
            "profile": self._action_profile,
            "stats": self._action_stats,
            "help": self._action_help,
        }

        handler = handlers.get(action)
        if not handler:
            raise ValueError(f"Unknown action: {action}")
        return await handler(query, options)

    async def _action_remember(self, query: str, options: dict[str, Any]) -> Any:
        text = query or options.get("text", "")
        cleaned = {k: v for k, v in options.items() if k != "text"}
        req = RememberRequest(text=text, **cleaned)
        res = await self.pipeline.remember(self.context_id, req)
        return res.model_dump(mode="json", by_alias=True)

    async def _action_recall(self, query: str, options: dict[str, Any]) -> Any:
        q = query or options.get("query", "")
        cleaned = {k: v for k, v in options.items() if k != "query"}
        req = RecallRequest(query=q, **cleaned)
        res = await self.retrieval.recall(self.context_id, req)
        return res.model_dump(mode="json", by_alias=True)

    async def _action_context(self, query: str, options: dict[str, Any]) -> Any:
        q = query or options.get("query", "")
        cleaned = {k: v for k, v in options.items() if k != "query"}
        req = ContextRequest(query=q, **cleaned)
        res = await self.retrieval.query_context(self.context_id, req)
        return res.model_dump(mode="json", by_alias=True)

    async def _action_reflect(self, query: str, options: dict[str, Any]) -> Any:
        q = query or options.get("query", "")
        cleaned = {k: v for k, v in options.items() if k != "query"}
        req = ReflectRequest(query=q, **cleaned)
        res = await self.retrieval.reflect(self.context_id, req, pipeline=self.pipeline)
        return res.model_dump(mode="json", by_alias=True)

    async def _action_lookup(self, query: str, options: dict[str, Any]) -> Any:
        q = query or options.get("query", "")
        cleaned = {k: v for k, v in options.items() if k != "query"}
        req = LookupRequest(query=q, **cleaned)
        res = await self.retrieval.lookup(self.context_id, req)
        return res.model_dump(mode="json", by_alias=True)

    async def _action_forget(self, query: str, options: dict[str, Any]) -> Any:
        q = query or options.get("query", "")
        cleaned = {k: v for k, v in options.items() if k != "query"}
        req = ForgetRequest(query=q, **cleaned)
        res = await self.pipeline.forget(self.context_id, req)
        return res.model_dump(mode="json", by_alias=True)

    async def _action_inspect(self, query: str, options: dict[str, Any]) -> Any:
        ref = query or options.get("ref", "")
        return await self.retrieval.inspect(self.context_id, ref)

    async def _action_entities(self, query: str, options: dict[str, Any]) -> Any:
        q = query or options.get("q")
        etype = options.get("entity_type") or options.get("type")
        limit = options.get("limit", 100)
        return await self.retrieval.list_entities(
            self.context_id, q=q, entity_type=etype, limit=limit
        )

    async def _action_history(self, query: str, options: dict[str, Any]) -> Any:
        ename = query or options.get("entity_name", "")
        etype = options.get("entity_type", "concept")
        key = options.get("key")
        return await self.retrieval.get_entity_history(
            self.context_id, etype, ename, key=key
        )

    async def _action_neighbourhood(self, query: str, options: dict[str, Any]) -> Any:
        ename = query or options.get("entity_name", "")
        etype = options.get("entity_type", "concept")
        return await self.retrieval.get_entity_neighbourhood(
            self.context_id, etype, ename
        )

    async def _action_uncertainty(self, query: str, options: dict[str, Any]) -> Any:
        return await self.retrieval.list_uncertainties(
            self.context_id, resolved=options.get("resolved")
        )

    async def _action_resolve_uncertainty(
        self, query: str, options: dict[str, Any]
    ) -> Any:
        uid = query or options.get("id") or options.get("uncertainty_id", "")
        req = ResolveUncertaintyRequest(**options)
        return await self.pipeline.resolve_uncertainty(self.context_id, uid, req)

    async def _action_consolidate(self, query: str, options: dict[str, Any]) -> Any:
        req = ConsolidateRequest(**options)
        res = await self.pipeline.consolidate(self.context_id, req)
        return res.model_dump(mode="json", by_alias=True)

    async def _action_expire(self, query: str, options: dict[str, Any]) -> Any:
        return await self.pipeline.expire_context(self.context_id)

    async def _action_profile(self, query: str, options: dict[str, Any]) -> Any:
        return await self.retrieval.get_profile(self.context_id)

    async def _action_stats(self, query: str, options: dict[str, Any]) -> Any:
        return await self.retrieval.get_trace_stats(self.context_id)

    async def _action_help(self, query: str, options: dict[str, Any]) -> Any:
        return HELP_DOCS


async def handle_mcp_request(
    msg: dict[str, Any],
    pipeline: Any,
    retrieval: Any,
    context_id: str = "dev",
    mcp_mode: str = "compact",
) -> dict[str, Any] | None:
    server = McpServer(pipeline, retrieval, context_id=context_id, mcp_mode=mcp_mode)
    return await server.handle_request(msg)


async def run_stdio_mcp(
    pipeline: Any,
    retrieval: Any,
    context_id: str = "dev",
    mcp_mode: str = "compact",
):
    server = McpServer(pipeline, retrieval, context_id=context_id, mcp_mode=mcp_mode)
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            req_data = json.loads(line)
            resp = await server.handle_request(req_data)
            if resp:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as e:
            err_resp = {"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
