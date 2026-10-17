from datetime import datetime
from typing import Any

from cortex.config import LLMConfig, EmbeddingConfig
from cortex.infra.llm import LLMProvider, MockLLMProvider, build_llm_provider
from cortex.infra.embedding import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    CachedEmbeddingProvider,
    build_embedding_provider,
)
from cortex.domain.models import LLMExtractionPayload


class CompositeAiService:
    def __init__(self, llm: LLMProvider, embedder: EmbeddingProvider):
        self.llm = llm
        self.embedder = embedder

    async def get_embedding(self, text: str) -> list[float]:
        return await self.embedder.get_embedding(text)

    async def extract_and_reconcile(
        self,
        input_text: str,
        anchor_time: datetime,
        existing_facts: list[dict[str, Any]],
        existing_instructions: list[dict[str, Any]] | None = None,
    ) -> LLMExtractionPayload:
        return await self.llm.extract_and_reconcile(
            input_text=input_text,
            anchor_time=anchor_time,
            existing_facts=existing_facts,
            existing_instructions=existing_instructions,
        )

    async def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        return await self.llm.complete_text(system_prompt, user_prompt)


class MockAiProvider(CompositeAiService):
    def __init__(self, dimension: int = 768, cache_size: int = 1000):
        super().__init__(
            llm=MockLLMProvider(),
            embedder=CachedEmbeddingProvider(
                MockEmbeddingProvider(dimension=dimension), max_size=cache_size
            ),
        )


def get_ai_provider(llm_cfg: LLMConfig, emb_cfg: EmbeddingConfig) -> CompositeAiService:
    return CompositeAiService(
        llm=build_llm_provider(llm_cfg),
        embedder=build_embedding_provider(emb_cfg),
    )
