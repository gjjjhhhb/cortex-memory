import pytest
from httpx import AsyncClient, ASGITransport

from cortex.config import Config, DatabaseConfig, LLMConfig, EmbeddingConfig
from cortex import (
    create_app,
    CortexStore,
    MockAiProvider,
    IngestionEngine,
    RetrievalEngine,
    CognitiveEngine,
    GraphEngine,
)


@pytest.fixture
def mock_config():
    return Config(
        server=Config().server,
        database=DatabaseConfig(url="mem://", namespace="cortex_test"),
        llm=LLMConfig(provider="mock"),
        embedding=EmbeddingConfig(provider="mock", dimension=1536, cache_size=100),
    )


@pytest.fixture
async def app_client(mock_config):
    app = create_app(mock_config)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await app.state.store.close()


@pytest.fixture
async def direct_engines(mock_config):
    store = CortexStore(
        url=mock_config.database.url,
        namespace=mock_config.database.namespace,
        dimension=mock_config.embedding.dimension,
    )
    ai = MockAiProvider(dimension=mock_config.embedding.dimension)
    ingestion = IngestionEngine(store=store, ai=ai)
    retrieval = RetrievalEngine(store=store, ai=ai)
    cognitive = CognitiveEngine(store=store, ai=ai, retrieval=retrieval)
    graph = GraphEngine(store=store)

    yield {
        "store": store,
        "ai": ai,
        "ingestion": ingestion,
        "retrieval": retrieval,
        "cognitive": cognitive,
        "graph": graph,
        "config": mock_config,
    }
    await store.close()
§
