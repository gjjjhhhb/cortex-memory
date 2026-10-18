from datetime import datetime
from typing import Any


def clean_val(v: Any, strip_embedding: bool = True) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, list):
        return [clean_val(x, strip_embedding=strip_embedding) for x in v]
    if isinstance(v, dict):
        return {
            k: clean_val(val, strip_embedding=strip_embedding)
            for k, val in v.items()
            if not (strip_embedding and k == "embedding")
        }
    return str(v)


def clean_record(rec: Any, strip_embedding: bool = True) -> Any:
    if not isinstance(rec, dict):
        return clean_val(rec, strip_embedding=strip_embedding)
    return {
        k: clean_val(v, strip_embedding=strip_embedding)
        for k, v in rec.items()
        if not (strip_embedding and k == "embedding")
    }


def format_scope(scopes: Any) -> str:
    if not scopes:
        return "global"
    if isinstance(scopes, str):
        return scopes
    if isinstance(scopes, list):
        if len(scopes) > 0 and isinstance(scopes[0], list):
            return "/".join(scopes[0])
        return "/".join(str(s) for s in scopes)
    return "global"


def normalize_entity_id(name: str) -> str:
    raw = str(name).strip().lower()
    user_aliases = {"user", "i", "me", "my", "self", "пользователь", "я", "мне", "меня"}
    if raw in user_aliases or raw.endswith(":user"):
        return "entity:user"
    clean = raw.replace("entity:", "")
    clean = "".join(c if c.isalnum() else "_" for c in clean).strip("_")
    return f"entity:{clean or 'item'}"


def build_markdown_context(
    instructions: list[dict[str, Any]],
    identities: list[dict[str, Any]],
    contexts: list[dict[str, Any]],
    relevant_hits: list[Any],
) -> str:
    seen: set[str] = set()
    sections: list[str] = []

    if instructions:
        sections.append("## Active Instructions")
        for inst in instructions:
            rule = inst.get("rule", "")
            seen.add(rule.strip().lower())
            seen.add(str(inst.get("id", "")))
            sections.append(f"- {rule}")
        sections.append("")

    if identities:
        sections.append("## User Identity & Profile")
        for idn in identities:
            stmt = idn.get("statement", "")
            seen.add(stmt.strip().lower())
            seen.add(str(idn.get("id", "")))
            sections.append(f"- {stmt}")
        sections.append("")

    if contexts:
        sections.append("## Current Context")
        for ctx_item in contexts:
            stmt = ctx_item.get("statement", "")
            seen.add(stmt.strip().lower())
            seen.add(str(ctx_item.get("id", "")))
            sections.append(f"- {stmt}")
        sections.append("")

    deduped = [
        h
        for h in relevant_hits
        if (getattr(h, "statement", None) or getattr(h, "text", "")).strip().lower()
        not in seen
        and getattr(h, "id", "") not in seen
        and (getattr(h, "bm25_score", 0.0) > 0.0 or getattr(h, "vec_score", 0.0) >= 0.45 or getattr(h, "score", 0.0) >= 0.5)
    ]

    if deduped:
        sections.append("## Relevant Memories")
        for i, h in enumerate(deduped, 1):
            sections.append(f"- {h.text} [S{i}]")
        sections.append("")

    return "\n".join(sections).strip()
