import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomllib


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    api_key: str = "cortex-local-dev-key"
    default_context: str = "dev"
    mcp_mode: str = "compact"


@dataclass
class DatabaseConfig:
    url: str = "surrealkv://./data/cortex.db"
    namespace: str = "cortex"
    database: str = "cortex"
    username: str | None = None
    password: str | None = None


@dataclass
class LLMConfig:
    provider: str = "mock"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "deepseek-flash"
    temperature: float = 0.0
    structured_output_mode: str = "json_object"


@dataclass
class EmbeddingConfig:
    provider: str = "mock"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    cache_size: int = 10000


@dataclass
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def with_memory_db(self, namespace: str = "test") -> "Config":
        return Config(
            server=self.server,
            database=DatabaseConfig(url="mem://", namespace=namespace),
            llm=self.llm,
            embedding=self.embedding,
        )

    def with_mock_providers(self, dimension: int = 1536) -> "Config":
        return Config(
            server=self.server,
            database=self.database,
            llm=LLMConfig(provider="mock"),
            embedding=EmbeddingConfig(provider="mock", dimension=dimension),
        )


def _resolve_env_var(val: str, default: str = "") -> str:
    if isinstance(val, str) and val.startswith("env:"):
        return os.environ.get(val[4:], default) or default
    return val or default


def load_config(config_path: str | Path | None = None) -> Config:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    path = Path(config_path) if config_path else Path("config.toml")
    data: dict[str, Any] = {}
    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)

    server_data = data.get("server", {})
    database_data = data.get("database", {})
    llm_data = data.get("llm", {})
    embedding_data = data.get("embedding", {})

    return Config(
        server=ServerConfig(
            host=os.environ.get("CORTEX_HOST", server_data.get("host", "127.0.0.1")),
            port=int(os.environ.get("CORTEX_PORT", server_data.get("port", 8080))),
            api_key=_resolve_env_var(
                os.environ.get(
                    "CORTEX_API_KEY", server_data.get("api_key", "cortex-local-dev-key")
                )
            ),
            default_context=os.environ.get(
                "CORTEX_DEFAULT_CONTEXT", server_data.get("default_context", "dev")
            ),
            mcp_mode=os.environ.get(
                "CORTEX_MCP_MODE", server_data.get("mcp_mode", "compact")
            ),
        ),
        database=DatabaseConfig(
            url=os.environ.get(
                "CORTEX_DB_URL",
                database_data.get("url", "surrealkv://./data/cortex.db"),
            ),
            namespace=os.environ.get(
                "CORTEX_DB_NAMESPACE", database_data.get("namespace", "cortex")
            ),
            database=os.environ.get(
                "CORTEX_DB_DATABASE", database_data.get("database", "cortex")
            ),
            username=os.environ.get(
                "CORTEX_DB_USERNAME", database_data.get("username", None)
            ),
            password=os.environ.get(
                "CORTEX_DB_PASSWORD", database_data.get("password", None)
            ),
        ),
        llm=LLMConfig(
            provider=os.environ.get(
                "CORTEX_LLM_PROVIDER", llm_data.get("provider", "mock")
            ),
            base_url=_resolve_env_var(
                os.environ.get(
                    "CORTEX_LLM_BASE_URL",
                    os.environ.get(
                        "LLM_BASE_URL",
                        os.environ.get("OPENAI_BASE_URL", llm_data.get("base_url", "")),
                    ),
                )
            ),
            api_key=_resolve_env_var(
                os.environ.get(
                    "CORTEX_LLM_API_KEY",
                    os.environ.get(
                        "LLM_API_KEY",
                        os.environ.get("OPENAI_API_KEY", llm_data.get("api_key", "")),
                    ),
                )
            ),
            model=_resolve_env_var(
                os.environ.get(
                    "CORTEX_LLM_MODEL",
                    os.environ.get(
                        "LLM_MODEL",
                        os.environ.get("OPENAI_MODEL", llm_data.get("model", "")),
                    ),
                )
            ),
            temperature=float(
                os.environ.get(
                    "CORTEX_LLM_TEMPERATURE", llm_data.get("temperature", 0.0)
                )
            ),
            structured_output_mode=os.environ.get(
                "CORTEX_LLM_STRUCTURED_OUTPUT_MODE",
                llm_data.get("structured_output_mode", "json_object"),
            ),
        ),
        embedding=EmbeddingConfig(
            provider=os.environ.get(
                "CORTEX_EMBEDDING_PROVIDER", embedding_data.get("provider", "mock")
            ),
            base_url=_resolve_env_var(
                os.environ.get(
                    "CORTEX_EMBEDDING_BASE_URL",
                    os.environ.get(
                        "EMBEDDER_BASE_URL",
                        os.environ.get(
                            "EMBEDDING_BASE_URL", embedding_data.get("base_url", "")
                        ),
                    ),
                )
            ),
            api_key=_resolve_env_var(
                os.environ.get(
                    "CORTEX_EMBEDDING_API_KEY",
                    os.environ.get(
                        "EMBEDDER_API_KEY",
                        os.environ.get(
                            "EMBEDDING_API_KEY", embedding_data.get("api_key", "")
                        ),
                    ),
                )
            ),
            model=_resolve_env_var(
                os.environ.get(
                    "CORTEX_EMBEDDING_MODEL",
                    os.environ.get(
                        "EMBEDDER_MODEL",
                        os.environ.get(
                            "EMBEDDING_MODEL", embedding_data.get("model", "")
                        ),
                    ),
                )
            ),
            dimension=int(
                os.environ.get(
                    "CORTEX_EMBEDDING_DIMENSION", embedding_data.get("dimension", 1536)
                )
            ),
            cache_size=int(
                os.environ.get(
                    "CORTEX_EMBEDDING_CACHE_SIZE",
                    embedding_data.get("cache_size", 10000),
                )
            ),
        ),
    )
