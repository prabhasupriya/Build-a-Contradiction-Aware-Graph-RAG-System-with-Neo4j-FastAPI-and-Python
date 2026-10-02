"""All Cypher is parameterized ($param) - never string-interpolated."""

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT attribute_key IF NOT EXISTS FOR (a:Attribute) REQUIRE a.key IS UNIQUE",
    "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (f:Fact) REQUIRE f.id IS UNIQUE",
]

CLEAR_GRAPH = "MATCH (n) WHERE n:Entity OR n:Attribute OR n:Fact DETACH DELETE n"

# Attribute is scoped PER ENTITY via `key` - merging on name alone would fuse "api_timeout" across entities.
INGEST_FACT = """
MERGE (e:Entity {name: $entity})
MERGE (a:Attribute {key: $attr_key})
  ON CREATE SET a.name = $attribute, a.entity = $entity
MERGE (e)-[:HAS_ATTRIBUTE]->(a)
MERGE (f:Fact {id: $fact_id})
  SET f.value = $value, f.source = $source, f.timestamp = $timestamp,
      f.confidence = $confidence, f.context_text = $context_text
MERGE (a)-[:STATED_BY]->(f)
"""

FETCH_FACTS = """
MATCH (e:Entity {name: $entity})-[:HAS_ATTRIBUTE]->(a:Attribute {name: $attribute})-[:STATED_BY]->(f:Fact)
RETURN f.value AS value, f.source AS source, f.timestamp AS timestamp, f.confidence AS confidence
ORDER BY f.timestamp
"""

CATALOG = "MATCH (e:Entity)-[:HAS_ATTRIBUTE]->(a:Attribute) RETURN e.name AS entity, a.name AS attribute"
COUNT_FACTS = "MATCH (f:Fact) RETURN count(f) AS fact_count"
