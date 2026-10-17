import hashlib
import re
from typing import Any, Protocol

import httpx

from cortex.config import EmbeddingConfig


class EmbeddingProvider(Protocol):
    async def get_embedding(self, text: str) -> list[float]: ...


class MockEmbeddingProvider:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    async def get_embedding(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        words = re.findall(r"\w+", text.lower())
        if not words:
            vec[0] = 1.0
            return vec

        tokens: list[str] = []
        for w in words:
            tokens.append(w)
            if len(w) >= 4:
                tokens.append(w[:4])

        for tok in tokens:
            h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
            vec[h % self.dimension] += 1.0

        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


class OpenAiEmbeddingProvider:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.client = httpx.AsyncClient(timeout=45.0)

    async def get_embedding(self, text: str) -> list[float]:
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        payload: dict[str, Any] = {"input": text, "model": self.cfg.model}
        url = f"{self.cfg.base_url.rstrip('/')}/embeddings"

        resp = await self.client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]


class GeminiEmbeddingProvider:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.client = httpx.AsyncClient(timeout=45.0)

    async def get_embedding(self, text: str) -> list[float]:
        model_name = self.cfg.model
        if model_name.startswith("models/"):
            model_name = model_name[len("models/") :]
        url = f"{self.cfg.base_url.rstrip('/')}/models/{model_name}:embedContent?key={self.cfg.api_key}"
        payload = {
            "model": f"models/{model_name}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.cfg.dimension,
        }
        resp = await self.client.post(
            url, json=payload, headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]["values"]


class CachedEmbeddingProvider:
    def __init__(self, inner: EmbeddingProvider, max_size: int = 10000):
        self.inner = inner
        self.max_size = max_size
        self.cache: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0

    async def get_embedding(self, text: str) -> list[float]:
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]

        self.misses += 1
        emb = await self.inner.get_embedding(text)
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[cache_key] = emb
        return emb

    def cache_info(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "currsize": len(self.cache),
            "maxsize": self.max_size,
        }


def build_embedding_provider(cfg: EmbeddingConfig) -> EmbeddingProvider:
    if cfg.provider == "mock":
        base: EmbeddingProvider = MockEmbeddingProvider(dimension=cfg.dimension)
    elif (
        cfg.provider == "gemini" or "generativelanguage.googleapis.com" in cfg.base_url
    ):
        base = GeminiEmbeddingProvider(cfg)
    else:
        base = OpenAiEmbeddingProvider(cfg)
    return CachedEmbeddingProvider(base, max_size=cfg.cache_size)
