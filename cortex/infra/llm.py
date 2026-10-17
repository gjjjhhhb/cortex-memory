import json
from datetime import datetime
from typing import Any, Protocol
import httpx

from cortex.config import LLMConfig
from cortex.domain.models import (
    LLMExtractionPayload,
    ExtractedEntity,
    ExtractedFactOp,
    ExtractedInstruction,
    ExtractedRelation,
    ExtractedUncertainty,
)
from cortex.domain.prompts import EXTRACTION_SYSTEM_PROMPT


class LLMProvider(Protocol):
    async def extract_and_reconcile(
        self,
        input_text: str,
        anchor_time: datetime,
        existing_facts: list[dict[str, Any]],
        existing_instructions: list[dict[str, Any]] | None = None,
    ) -> LLMExtractionPayload: ...

    async def complete_text(self, system_prompt: str, user_prompt: str) -> str: ...


class MockLLMProvider:
    async def extract_and_reconcile(
        self,
        input_text: str,
        anchor_time: datetime,
        existing_facts: list[dict[str, Any]],
        existing_instructions: list[dict[str, Any]] | None = None,
    ) -> LLMExtractionPayload:
        text_lower = input_text.lower()
        entities: list[ExtractedEntity] = []
        fact_ops: list[ExtractedFactOp] = []
        relations: list[ExtractedRelation] = []
        instructions: list[ExtractedInstruction] = []
        uncertainties: list[ExtractedUncertainty] = []

        instruction_triggers = [
            "always ",
            "never ",
            "всегда ",
            "никогда ",
            "отвечай ",
            "respond ",
        ]
        if any(tr in text_lower for tr in instruction_triggers):
            target_inst_id = None
            if existing_instructions:
                for inst in existing_instructions:
                    old_rule = str(inst.get("rule", "")).lower()
                    brevity_words = ["short", "brief", "concise", "кратко"]
                    detail_words = ["detail", "detailed", "long", "подробно"]
                    if (
                        any(w in old_rule for w in brevity_words)
                        and any(w in text_lower for w in detail_words)
                    ) or (
                        any(w in old_rule for w in detail_words)
                        and any(w in text_lower for w in brevity_words)
                    ):
                        target_inst_id = str(inst.get("id"))
                        break
            instructions.append(
                ExtractedInstruction(
                    op="SUPERSEDE" if target_inst_id else "ADD",
                    target_instruction_id=target_inst_id,
                    rule=input_text.strip(),
                )
            )

        city_match = None
        if "berlin" in text_lower or "берлин" in text_lower:
            city_match = "Berlin"
        elif "paris" in text_lower or "париж" in text_lower:
            city_match = "Paris"

        entities.append(
            ExtractedEntity(name="User", entity_type="person", summary="Primary user")
        )

        if city_match:
            entities.append(
                ExtractedEntity(name=city_match, entity_type="place", summary="City")
            )
            target_id = None
            for f in existing_facts:
                if f.get("key") == "city" and f.get("value") != city_match:
                    target_id = str(f.get("id"))
                    break

            fact_ops.append(
                ExtractedFactOp(
                    op="SUPERSEDE" if target_id else "ADD",
                    target_attribute_id=target_id,
                    entity="User",
                    key="city",
                    value=city_match,
                    statement=f"User lives in {city_match}",
                    memory_category="identity",
                    valid_from=anchor_time,
                    valid_until=None,
                    confidence=1.0,
                    source_quote=city_match,
                )
            )
            relations.append(
                ExtractedRelation(
                    source="User",
                    predicate="lives_in",
                    target=city_match,
                    memory_category="identity",
                    valid_from=anchor_time,
                )
            )

        has_tech = False
        for tech in ["rust", "python", "typescript", "go"]:
            if tech in text_lower:
                has_tech = True
                fact_ops.append(
                    ExtractedFactOp(
                        op="ADD",
                        entity="User",
                        key="knows_language",
                        value=tech.capitalize(),
                        statement=f"User knows {tech.capitalize()}",
                        memory_category="knowledge",
                        valid_from=anchor_time,
                        valid_until=None,
                        confidence=1.0,
                        source_quote=tech,
                    )
                )

        if not instructions and not city_match and not has_tech and input_text.strip():
            fact_ops.append(
                ExtractedFactOp(
                    op="ADD",
                    entity="User",
                    key="note",
                    value=input_text.strip(),
                    statement=input_text.strip(),
                    memory_category="knowledge",
                    valid_from=anchor_time,
                    valid_until=None,
                    confidence=1.0,
                    source_quote=input_text.strip(),
                )
            )

        return LLMExtractionPayload(
            entities=entities,
            fact_operations=fact_ops,
            relations=relations,
            instructions=instructions,
            uncertainties=uncertainties,
        )

    async def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        if (
            "[S1]" in user_prompt
            or "Sources:" in user_prompt
            or "[s1]" in user_prompt.lower()
        ):
            return f"Based on your memory [S1]: {user_prompt[:120]}"
        return f"Synthesized reflection on: {user_prompt[:120]}"


class OpenAiLLMProvider:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = httpx.AsyncClient(timeout=60.0)

    async def extract_and_reconcile(
        self,
        input_text: str,
        anchor_time: datetime,
        existing_facts: list[dict[str, Any]],
        existing_instructions: list[dict[str, Any]] | None = None,
    ) -> LLMExtractionPayload:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        user_prompt = json.dumps(
            {
                "anchor_time": anchor_time.isoformat(),
                "existing_facts": existing_facts,
                "existing_instructions": existing_instructions or [],
                "input_text": input_text,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "response_format": {"type": "json_object"},
        }
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"

        resp = await self.client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        res_data = resp.json()
        content = res_data["choices"][0]["message"]["content"]
        parsed_json = json.loads(content)
        return LLMExtractionPayload.model_validate(parsed_json)

    async def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg.temperature,
        }
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        resp = await self.client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        res_data = resp.json()
        return res_data["choices"][0]["message"]["content"]


def build_llm_provider(cfg: LLMConfig) -> LLMProvider:
    if cfg.provider == "mock":
        return MockLLMProvider()
    return OpenAiLLMProvider(cfg)
