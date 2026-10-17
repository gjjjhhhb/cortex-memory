from enum import StrEnum


class InferMode(StrEnum):
    FULL = "full"
    TRIPLES = "triples"
    PREVIEW = "preview"
    NONE = "none"


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    VECTOR = "vector"
    BM25 = "bm25"
    GRAPH = "graph"


class ScopeView(StrEnum):
    STRICT = "strict"
    CROSS_TEAM = "crossTeam"
    MERGED = "merged"


class FactOpType(StrEnum):
    ADD = "ADD"
    SUPERSEDE = "SUPERSEDE"
    MERGE = "MERGE"


class MemoryCategory(StrEnum):
    IDENTITY = "identity"
    KNOWLEDGE = "knowledge"
    CONTEXT = "context"


class HitSource(StrEnum):
    ENTITY = "entity"
    ATTRIBUTE = "attribute"
    MEMORY_CHUNK = "memory_chunk"
    CHUNK = "chunk"
    SECTION = "section"


class ExtractionStrategy(StrEnum):
    WHOLE_CONVERSATION = "whole_conversation"
    PER_MESSAGE = "per_message"
