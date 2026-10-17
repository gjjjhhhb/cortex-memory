from cortex.config import Config, load_config
from cortex.engine.cognitive import CognitiveEngine
from cortex.engine.graph import GraphEngine
from cortex.engine.ingestion import IngestionEngine
from cortex.engine.retrieval import RetrievalEngine
from cortex.infra.ai import get_ai_provider
from cortex.infra.store import CortexStore


class CortexRuntime:
    def __init__(self, config: Config):
        self.config = config
        self.store = CortexStore(
            url=config.database.url,
            namespace=config.database.namespace,
            dimension=config.embedding.dimension,
            database=config.database.database,
            username=config.database.username,
            password=config.database.password,
        )
        self.ai = get_ai_provider(config.llm, config.embedding)
        self.ingestion = IngestionEngine(store=self.store, ai=self.ai)
        self.retrieval = RetrievalEngine(store=self.store, ai=self.ai)
        self.cognitive = CognitiveEngine(
            store=self.store, ai=self.ai, retrieval=self.retrieval
        )
        self.graph = GraphEngine(store=self.store)

    @classmethod
    def from_config(cls, config: Config | None = None) -> "CortexRuntime":
        cfg = config or load_config()
        return cls(cfg)

    async def close(self):
        await self.store.close()
