import os
from pathlib import Path
from typing import Any

import tomllib
from dotenv import load_dotenv
from pydantic import BaseModel, Field


def _resolve(val: Any, default: Any = "") -> Any:
    if val is None:
        return default
    s = str(val)
    if s.startswith("env:"):
        return os.environ.get(s[4:], default) or default
    return val


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = "cortex-local-dev-key"
    default_context: str = "dev"
    mcp_mode: str = "compact"


class DatabaseSettings(BaseModel):
    url: str = "surrealkv://./data/cortex.db"
    namespace: str = "cortex"
    database: str = "cortex"
    username: str | None = None
    password: str | None = None


class LLMSettings(BaseModel):
    provider: str = "mock"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.0
    structured_output_mode: str = "json_object"


class EmbeddingSettings(BaseModel):
    provider: str = "mock"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    dimension: int = 1536
    cache_size: int = 10000


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)

    def with_memory_db(self, namespace: str = "test") -> "Settings":
        return Settings(
            server=self.server.model_copy(),
            database=DatabaseSettings(
                url="mem://", namespace=namespace, database=self.database.database
            ),
            llm=self.llm.model_copy(),
            embedding=self.embedding.model_copy(),
        )

    def with_mock_providers(self, dimension: int = 1536) -> "Settings":
        return Settings(
            server=self.server.model_copy(),
            database=self.database.model_copy(),
            llm=LLMSettings(provider="mock"),
            embedding=EmbeddingSettings(provider="mock", dimension=dimension),
        )


ServerConfig = ServerSettings
DatabaseConfig = DatabaseSettings
LLMConfig = LLMSettings
EmbeddingConfig = EmbeddingSettings
Config = Settings


def load_config(config_path: str | Path | None = None) -> Settings:
    load_dotenv()
    path = Path(config_path) if config_path else Path("config.toml")
    data: dict[str, Any] = {}
    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)

    srv = data.get("server", {})
    db = data.get("database", {})
    llm = data.get("llm", {})
    emb = data.get("embedding", {})

    return Settings(
        server=ServerSettings(
            host=os.environ.get("CORTEX_HOST")
            or _resolve(srv.get("host"), "127.0.0.1"),
            port=int(os.environ.get("CORTEX_PORT") or _resolve(srv.get("port"), 8080)),
            api_key=str(
                _resolve(
                    os.environ.get("CORTEX_API_KEY") or srv.get("api_key"),
                    "cortex-local-dev-key",
                )
            ),
            default_context=os.environ.get("CORTEX_DEFAULT_CONTEXT")
            or _resolve(srv.get("default_context"), "dev"),
            mcp_mode=os.environ.get("CORTEX_MCP_MODE")
            or _resolve(srv.get("mcp_mode"), "compact"),
        ),
        database=DatabaseSettings(
            url=os.environ.get("CORTEX_DB_URL")
            or _resolve(db.get("url"), "surrealkv://./data/cortex.db"),
            namespace=os.environ.get("CORTEX_DB_NAMESPACE")
            or _resolve(db.get("namespace"), "cortex"),
            database=os.environ.get("CORTEX_DB_DATABASE")
            or _resolve(db.get("database"), "cortex"),
            username=os.environ.get("CORTEX_DB_USERNAME") or db.get("username"),
            password=os.environ.get("CORTEX_DB_PASSWORD") or db.get("password"),
        ),
        llm=LLMSettings(
            provider=os.environ.get("CORTEX_LLM_PROVIDER")
            or _resolve(llm.get("provider"), "mock"),
            base_url=str(
                _resolve(
                    os.environ.get("CORTEX_LLM_BASE_URL")
                    or os.environ.get("LLM_BASE_URL")
                    or os.environ.get("OPENAI_BASE_URL")
                    or llm.get("base_url"),
                    "",
                )
            ),
            api_key=str(
                _resolve(
                    os.environ.get("CORTEX_LLM_API_KEY")
                    or os.environ.get("LLM_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                    or llm.get("api_key"),
                    "",
                )
            ),
            model=str(
                _resolve(
                    os.environ.get("CORTEX_LLM_MODEL")
                    or os.environ.get("LLM_MODEL")
                    or os.environ.get("OPENAI_MODEL")
                    or llm.get("model"),
                    "",
                )
            ),
            temperature=float(
                os.environ.get("CORTEX_LLM_TEMPERATURE")
                or _resolve(llm.get("temperature"), 0.0)
            ),
            structured_output_mode=str(
                os.environ.get("CORTEX_LLM_STRUCTURED_OUTPUT_MODE")
                or _resolve(llm.get("structured_output_mode"), "json_object")
            ),
        ),
        embedding=EmbeddingSettings(
            provider=os.environ.get("CORTEX_EMBEDDING_PROVIDER")
            or _resolve(emb.get("provider"), "mock"),
            base_url=str(
                _resolve(
                    os.environ.get("CORTEX_EMBEDDING_BASE_URL")
                    or os.environ.get("EMBEDDER_BASE_URL")
                    or os.environ.get("EMBEDDING_BASE_URL")
                    or emb.get("base_url"),
                    "",
                )
            ),
            api_key=str(
                _resolve(
                    os.environ.get("CORTEX_EMBEDDING_API_KEY")
                    or os.environ.get("EMBEDDER_API_KEY")
                    or os.environ.get("EMBEDDING_API_KEY")
                    or emb.get("api_key"),
                    "",
                )
            ),
            model=str(
                _resolve(
                    os.environ.get("CORTEX_EMBEDDING_MODEL")
                    or os.environ.get("EMBEDDER_MODEL")
                    or os.environ.get("EMBEDDING_MODEL")
                    or emb.get("model"),
                    "",
                )
            ),
            dimension=int(
                _resolve(
                    os.environ.get("CORTEX_EMBEDDING_DIMENSION")
                    or os.environ.get("EMBEDDER_DIMENSION")
                    or emb.get("dimension"),
                    1536,
                )
            ),
            cache_size=int(
                _resolve(
                    os.environ.get("CORTEX_EMBEDDING_CACHE_SIZE")
                    or emb.get("cache_size"),
                    10000,
                )
            ),
        ),
    )
