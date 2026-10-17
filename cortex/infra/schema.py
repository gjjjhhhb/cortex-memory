"""SurrealQL schema DDL generator with dynamic vector dimension."""


def get_schema_ddl(dimension: int = 1536) -> str:
    return """
-- 1. Episodic layer
DEFINE TABLE IF NOT EXISTS session SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS scope ON TABLE session TYPE string DEFAULT 'global';
DEFINE FIELD IF NOT EXISTS created_at ON TABLE session TYPE any DEFAULT time::now();

DEFINE TABLE IF NOT EXISTS turn SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS session ON TABLE turn TYPE any;
DEFINE FIELD IF NOT EXISTS role ON TABLE turn TYPE string ASSERT $value IN ['user', 'assistant', 'system'];
DEFINE FIELD IF NOT EXISTS content ON TABLE turn TYPE string;
DEFINE FIELD IF NOT EXISTS created_at ON TABLE turn TYPE any DEFAULT time::now();

-- 2. Entities
DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS name ON TABLE entity TYPE string;
DEFINE FIELD IF NOT EXISTS entity_type ON TABLE entity TYPE string DEFAULT 'concept';
DEFINE FIELD IF NOT EXISTS aliases ON TABLE entity TYPE array<string> DEFAULT [];
DEFINE FIELD IF NOT EXISTS summary ON TABLE entity TYPE option<string>;
DEFINE FIELD IF NOT EXISTS scope ON TABLE entity TYPE string DEFAULT 'global';
DEFINE FIELD IF NOT EXISTS embedding ON TABLE entity TYPE option<array<float>>;

DEFINE INDEX IF NOT EXISTS idx_entity_name ON TABLE entity FIELDS name;

-- 3. Attributes (Facts with Tri-temporal fields)
DEFINE TABLE IF NOT EXISTS attribute SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS entity ON TABLE attribute TYPE any;
DEFINE FIELD IF NOT EXISTS key ON TABLE attribute TYPE string;
DEFINE FIELD IF NOT EXISTS value ON TABLE attribute TYPE string;
DEFINE FIELD IF NOT EXISTS statement ON TABLE attribute TYPE string;
DEFINE FIELD IF NOT EXISTS memory_category ON TABLE attribute TYPE string 
    ASSERT $value IN ['identity', 'knowledge', 'context'];
DEFINE FIELD IF NOT EXISTS valid_from ON TABLE attribute TYPE any DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS valid_until ON TABLE attribute TYPE any;
DEFINE FIELD IF NOT EXISTS created_at ON TABLE attribute TYPE any DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS superseded_by ON TABLE attribute TYPE any;
DEFINE FIELD IF NOT EXISTS source_turn ON TABLE attribute TYPE any;
DEFINE FIELD IF NOT EXISTS source_span ON TABLE attribute TYPE option<array<int>>;
DEFINE FIELD IF NOT EXISTS confidence ON TABLE attribute TYPE float DEFAULT 1.0;
DEFINE FIELD IF NOT EXISTS scope ON TABLE attribute TYPE string DEFAULT 'global';
DEFINE FIELD IF NOT EXISTS labels ON TABLE attribute TYPE array<string> DEFAULT [];
DEFINE FIELD IF NOT EXISTS embedding ON TABLE attribute TYPE array<float>;

-- 4. Relations
DEFINE TABLE IF NOT EXISTS relation SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS source_entity ON TABLE relation TYPE string;
DEFINE FIELD IF NOT EXISTS target_entity ON TABLE relation TYPE string;
DEFINE FIELD IF NOT EXISTS predicate ON TABLE relation TYPE string;
DEFINE FIELD IF NOT EXISTS memory_category ON TABLE relation TYPE string DEFAULT 'knowledge';
DEFINE FIELD IF NOT EXISTS valid_from ON TABLE relation TYPE any DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS valid_until ON TABLE relation TYPE any;
DEFINE FIELD IF NOT EXISTS created_at ON TABLE relation TYPE any DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS scope ON TABLE relation TYPE string DEFAULT 'global';
DEFINE INDEX IF NOT EXISTS idx_relation_source ON TABLE relation FIELDS source_entity;
DEFINE INDEX IF NOT EXISTS idx_relation_target ON TABLE relation FIELDS target_entity;

-- 5. Behavioral Instructions
DEFINE TABLE IF NOT EXISTS instruction SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS rule ON TABLE instruction TYPE string;
DEFINE FIELD IF NOT EXISTS scope ON TABLE instruction TYPE string DEFAULT 'global';
DEFINE FIELD IF NOT EXISTS active ON TABLE instruction TYPE bool DEFAULT true;
DEFINE FIELD IF NOT EXISTS created_at ON TABLE instruction TYPE any DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS source_turn ON TABLE instruction TYPE any;

-- 6. Uncertainties
DEFINE TABLE IF NOT EXISTS uncertainty SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS description ON TABLE uncertainty TYPE string;
DEFINE FIELD IF NOT EXISTS entity_name ON TABLE uncertainty TYPE option<string>;
DEFINE FIELD IF NOT EXISTS resolved ON TABLE uncertainty TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS created_at ON TABLE uncertainty TYPE any DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS scope ON TABLE uncertainty TYPE string DEFAULT 'global';

-- 7. Memory Chunks (infer: none)
DEFINE TABLE IF NOT EXISTS memory_chunk SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS text ON TABLE memory_chunk TYPE string;
DEFINE FIELD IF NOT EXISTS scope ON TABLE memory_chunk TYPE string DEFAULT 'global';
DEFINE FIELD IF NOT EXISTS labels ON TABLE memory_chunk TYPE array<string> DEFAULT [];
DEFINE FIELD IF NOT EXISTS embedding ON TABLE memory_chunk TYPE array<float>;
DEFINE FIELD IF NOT EXISTS created_at ON TABLE memory_chunk TYPE any DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS source_turn ON TABLE memory_chunk TYPE any;

-- 8. Traces
DEFINE TABLE IF NOT EXISTS trace SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS query ON TABLE trace TYPE string;
DEFINE FIELD IF NOT EXISTS mode ON TABLE trace TYPE string;
DEFINE FIELD IF NOT EXISTS tier ON TABLE trace TYPE string;
DEFINE FIELD IF NOT EXISTS query_ms ON TABLE trace TYPE int;
DEFINE FIELD IF NOT EXISTS hits_count ON TABLE trace TYPE int;
DEFINE FIELD IF NOT EXISTS created_at ON TABLE trace TYPE any DEFAULT time::now();
"""
