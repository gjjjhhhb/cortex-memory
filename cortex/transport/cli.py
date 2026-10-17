import asyncio
import json
import typer
import uvicorn

from cortex.config import load_config
from cortex.runtime import CortexRuntime
from cortex.domain.models import (
    RememberRequest,
    RecallRequest,
    ContextRequest,
    ReflectRequest,
    ChatRequest,
    ForgetRequest,
    LookupRequest,
)
from cortex.transport.mcp import run_stdio_mcp
from cortex.transport.http import create_app

app = typer.Typer(help="Cortex: Personal Agent Memory Server (v0.5.0)")


@app.command()
def serve(
    config_path: str = typer.Option(None, "--config", "-c", help="Path to config.toml"),
):
    cfg = load_config(config_path)
    fastapi_app = create_app(cfg)
    typer.echo(f"Starting Cortex server on http://{cfg.server.host}:{cfg.server.port}")
    uvicorn.run(fastapi_app, host=cfg.server.host, port=cfg.server.port)


@app.command()
def mcp(
    context: str = typer.Option("dev", "--context", help="Target memory context ID"),
    mode: str = typer.Option(
        "compact", "--mode", "-m", help="MCP tool mode (compact or full)"
    ),
):
    runtime = CortexRuntime.from_config()
    asyncio.run(
        run_stdio_mcp(runtime.ingestion, runtime.retrieval, context, mcp_mode=mode)
    )


@app.command()
def remember(
    text: str = typer.Argument(..., help="Text to remember"),
    context: str = typer.Option("dev", "--context", help="Context ID"),
    infer: str = typer.Option(
        "full", "--infer", help="Extraction mode (full, none, preview, triples)"
    ),
):
    async def _run():
        runtime = CortexRuntime.from_config()
        req = RememberRequest(text=text, infer=infer)
        res = await runtime.ingestion.remember(context, req)
        typer.echo(
            json.dumps(
                res.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False
            )
        )

    asyncio.run(_run())


@app.command()
def recall(
    query: str = typer.Argument(..., help="Query to recall from memory"),
    context: str = typer.Option("dev", "--context", help="Context ID"),
    limit: int = typer.Option(5, "--limit", "-k", help="Max results"),
    mode: str = typer.Option(
        "hybrid", "--mode", "-m", help="Search mode (hybrid, vector, bm25)"
    ),
):
    async def _run():
        runtime = CortexRuntime.from_config()
        req = RecallRequest(query=query, k=limit, mode=mode)
        res = await runtime.retrieval.recall(context, req)
        typer.echo(
            json.dumps(
                res.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False
            )
        )

    asyncio.run(_run())


@app.command()
def context(
    query: str = typer.Argument(..., help="Task or query to assemble context for"),
    context_id: str = typer.Option("dev", "--context", help="Context ID"),
):
    async def _run():
        runtime = CortexRuntime.from_config()
        req = ContextRequest(query=query)
        res = await runtime.retrieval.query_context(context_id, req)
        typer.echo(res.context)

    asyncio.run(_run())


@app.command()
def reflect(
    query: str = typer.Argument(..., help="Topic or question to reflect on"),
    context: str = typer.Option("dev", "--context", help="Context ID"),
    persist: bool = typer.Option(
        False, "--persist", "-p", help="Persist reflections as memories"
    ),
):
    async def _run():
        runtime = CortexRuntime.from_config()
        req = ReflectRequest(query=query, persist=persist)
        res = await runtime.cognitive.reflect(context, req)
        typer.echo(res.reflection)

    asyncio.run(_run())


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to chat with memory agent"),
    context: str = typer.Option("dev", "--context", help="Context ID"),
):
    async def _run():
        runtime = CortexRuntime.from_config()
        req = ChatRequest(message=message)
        res = await runtime.cognitive.chat(context, req)
        typer.echo(res.reply)
        if res.citations:
            typer.echo("\nCitations:")
            for c in res.citations:
                typer.echo(f"  [{c.get('marker')}]: {c.get('snippet')}")

    asyncio.run(_run())


@app.command()
def forget(
    query: str = typer.Argument(..., help="Topic or fact to forget"),
    context: str = typer.Option("dev", "--context", help="Context ID"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview matches without deleting"
    ),
    purge: bool = typer.Option(
        False, "--purge", help="Hard-delete records instead of soft-delete"
    ),
):
    async def _run():
        runtime = CortexRuntime.from_config()
        req = ForgetRequest(query=query, dry_run=dry_run, purge=purge)
        res = await runtime.ingestion.forget(context, req)
        typer.echo(
            json.dumps(
                res.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False
            )
        )

    asyncio.run(_run())


@app.command()
def lookup(
    subject: str = typer.Argument(
        ..., help="Entity name or subject to resolve in graph"
    ),
    context: str = typer.Option("dev", "--context", help="Context ID"),
):
    async def _run():
        runtime = CortexRuntime.from_config()
        req = LookupRequest(query=subject)
        res = await runtime.retrieval.lookup(context, req)
        typer.echo(
            json.dumps(
                res.model_dump(mode="json", by_alias=True), indent=2, ensure_ascii=False
            )
        )

    asyncio.run(_run())


@app.command()
def inspect(
    ref: str = typer.Argument(..., help="Record ID (e.g. entity:user or attribute:id)"),
    context: str = typer.Option("dev", "--context", help="Context ID"),
):
    async def _run():
        runtime = CortexRuntime.from_config()
        res = await runtime.retrieval.inspect(context, ref)
        typer.echo(json.dumps(res, indent=2, ensure_ascii=False))

    asyncio.run(_run())


@app.command()
def profile(
    context: str = typer.Option("dev", "--context", help="Context ID"),
):
    async def _run():
        runtime = CortexRuntime.from_config()
        res = await runtime.graph.get_profile(context)
        if res.get("markdown"):
            typer.echo(res["markdown"])
        else:
            typer.echo(json.dumps(res, indent=2, ensure_ascii=False))

    asyncio.run(_run())


@app.command()
def stats(
    context: str = typer.Option("dev", "--context", help="Context ID"),
):
    async def _run():
        runtime = CortexRuntime.from_config()
        res = await runtime.graph.get_trace_stats(context)
        typer.echo(json.dumps(res, indent=2, ensure_ascii=False))

    asyncio.run(_run())


if __name__ == "__main__":
    app()
