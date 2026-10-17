from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field
from cortex.domain.enums import (
    InferMode,
    SearchMode,
    ScopeView,
    FactOpType,
    MemoryCategory,
    HitSource,
    ExtractionStrategy,
)


def to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(w.capitalize() for w in parts[1:])


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class RememberRequest(CamelModel):
    text: str | None = None
    infer: InferMode | str = InferMode.FULL
    scopes: list[list[str]] | list[str] | str | None = None
    scope: list[list[str]] | list[str] | str | None = None
    labels: list[str] = Field(default_factory=list)
    session_id: str | None = None
    role: str = "user"
    memory_category: MemoryCategory | str | None = None
    observed_at: datetime | None = None
    triples: list[dict[str, Any]] | dict[str, Any] | None = None


class BatchMessage(CamelModel):
    role: str = "user"
    content: str | None = None
    text: str | None = None
    ts: datetime | None = None

    def get_content(self) -> str:
        return self.content or self.text or ""


class RememberBatchRequest(CamelModel):
    messages: list[BatchMessage] = Field(default_factory=list)
    items: list[BatchMessage] = Field(default_factory=list)
    scopes: list[list[str]] | list[str] | str | None = None
    session_id: str | None = None
    extract: ExtractionStrategy | str = ExtractionStrategy.WHOLE_CONVERSATION


class RecallRequest(CamelModel):
    query: str
    k: int = Field(default=10, ge=1, le=50)
    mode: SearchMode | str = SearchMode.HYBRID
    lens: list[list[str]] | list[str] | str | None = None
    labels: list[str] = Field(default_factory=list)
    scope_view: ScopeView | str = ScopeView.STRICT
    include: list[str] | None = None
    as_of: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    include_duplicates: bool = False
    source: str | None = None


class ContextRequest(CamelModel):
    query: str
    k: int = Field(default=10, ge=1, le=50)
    lens: list[list[str]] | list[str] | str | None = None
    labels: list[str] = Field(default_factory=list)
    as_of: datetime | None = None


class ForgetRequest(CamelModel):
    query: str
    purge: bool = False
    dry_run: bool = False
    scopes: list[list[str]] | list[str] | str | None = None


class ExtractedEntity(CamelModel):
    name: str
    entity_type: str = "concept"
    summary: str | None = None


class ExtractedFactOp(CamelModel):
    op: FactOpType | str = FactOpType.ADD
    target_attribute_id: str | None = None
    entity: str
    key: str
    value: str
    statement: str
    memory_category: MemoryCategory | str = MemoryCategory.KNOWLEDGE
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    confidence: float = 1.0
    source_quote: str | None = None


class ExtractedRelation(CamelModel):
    source: str
    predicate: str
    target: str
    memory_category: str = "knowledge"
    valid_from: datetime | None = None
    valid_until: datetime | None = None


class ExtractedInstruction(CamelModel):
    op: FactOpType | str = FactOpType.ADD
    target_instruction_id: str | None = None
    rule: str


class ExtractedUncertainty(CamelModel):
    description: str
    entity_name: str | None = None


class LLMExtractionPayload(CamelModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    fact_operations: list[ExtractedFactOp] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    instructions: list[ExtractedInstruction] = Field(default_factory=list)
    uncertainties: list[ExtractedUncertainty] = Field(default_factory=list)


class ExtractionDiff(CamelModel):
    entities: list[dict[str, Any]] = Field(default_factory=list)
    attributes: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    instructions: list[dict[str, Any]] = Field(default_factory=list)
    uncertainties: list[dict[str, Any]] = Field(default_factory=list)
    superseded: list[str] = Field(default_factory=list)


class RememberResponse(CamelModel):
    mode: str
    session_id: str
    turn_id: str
    chunk_id: str | None = None
    extraction: ExtractionDiff | None = None
    preview: ExtractionDiff | None = None
    trace_id: str


class RememberBatchResponse(CamelModel):
    session_id: str
    turn_ids: list[str]
    extractions: list[ExtractionDiff]


class RecallHit(CamelModel):
    id: str
    text: str
    statement: str
    score: float
    source: HitSource | str = HitSource.ATTRIBUTE
    memory_category: str | None = None
    entity_name: str | None = None
    key: str | None = None
    value: str | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    occurred_at: datetime | None = None
    source_span: list[int] | None = None


class TraceInfo(CamelModel):
    trace_id: str


class RecallResponse(CamelModel):
    tier: str = "hybrid"
    hits: list[RecallHit]
    context_hits: list[RecallHit] = Field(default_factory=list)
    query_ms: int
    trace: TraceInfo
    trace_id: str


class ContextQueryResponse(CamelModel):
    context: str
    tier: str = "hybrid"
    query_ms: int
    trace_id: str


class ForgetResponse(CamelModel):
    deleted: int
    trace_id: str = ""
    dry_run: bool | None = None
    matches: list[dict[str, Any]] | None = None


class HealthResponse(CamelModel):
    status: str = "ok"
    version: str = "0.3.0"
    service: str = "cortex"


class LookupRequest(CamelModel):
    query: str = ""
    subject: str | None = None
    entity_type: str | None = None
    include: list[str] | None = None
    fact_limit: int = 50
    relation_limit: int = 50
    uncertainty_limit: int = 20


class LookupResponse(CamelModel):
    resolution: dict[str, Any] = Field(default_factory=dict)
    entities: list[dict[str, Any]] = Field(default_factory=list)
    facts: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty: list[dict[str, Any]] = Field(default_factory=list)
    coverage: dict[str, int] = Field(default_factory=dict)


class ReflectRequest(CamelModel):
    query: str
    persist: bool = False
    lens: list[list[str]] | list[str] | str | None = None


class ReflectResponse(CamelModel):
    reflection: str
    evidence_refs: list[str] = Field(default_factory=list)
    persisted_attributes: list[dict[str, Any]] = Field(default_factory=list)
    trace_id: str


class ChatRequest(CamelModel):
    message: str
    scopes: list[list[str]] | list[str] | str | None = None
    session_id: str | None = None
    stream: bool = False


class ChatResponse(CamelModel):
    reply: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str
    trace_id: str


class ConsolidateRequest(CamelModel):
    scopes: list[list[str]] | list[str] | str | None = None
    dry_run: bool = False


class ConsolidateResponse(CamelModel):
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    merged_count: int = 0
    trace_id: str


class ResolveUncertaintyRequest(CamelModel):
    accepted_value: str
    note: str | None = None


class SessionCreateRequest(CamelModel):
    scopes: list[list[str]] | list[str] | str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
