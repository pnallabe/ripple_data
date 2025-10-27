// Neo4j schema & import script for Stock Dependency Graph

// 1. Constraints
CREATE CONSTRAINT company_ticker IF NOT EXISTS
FOR (c:Company)
REQUIRE c.ticker IS UNIQUE;

CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.sector);
CREATE INDEX IF NOT EXISTS FOR (c:Company) ON (c.name);

// 2. Sample node & relationship creation
MERGE (jpm:Company {ticker:'JPM', name:'JPMorgan Chase & Co.'})
MERGE (bac:Company {ticker:'BAC', name:'Bank of America Corp.'})
MERGE (jpm)-[:CORRELATED_WITH {weight:0.68, source:'rolling_corr_180d', source_date: date('2025-10-01')}]->(bac);

// 3. Bulk CSV import template (place edges.csv in import directory)
LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MERGE (s:Company {ticker: row.source_ticker})
MERGE (t:Company {ticker: row.target_ticker})
ON CREATE SET s.created = date(), t.created = date()
MERGE (s)-[r:REL {relation_type: row.relation_type}]->(t)
ON CREATE SET r.weight = toFloat(row.weight), r.source = row.source, r.source_date = row.source_date, r.meta = apoc.convert.fromJsonMap(row.meta_json)
ON MATCH SET r.weight = toFloat(row.weight), r.source = row.source, r.source_date = row.source_date;