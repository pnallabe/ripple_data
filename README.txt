Files included:
- SDD_Stock_Ripple_Platform.md        : System Design Document (overview, architecture, modeling)
- postgres_schema_ripple.sql           : PostgreSQL DDL for companies, prices, events, edges_staging
- neo4j_import_ripple.cypher          : Neo4j constraints, sample relationships, CSV import template
- ripple_simulation.py                : Python prototype to extract graph from Neo4j and run ripple simulation

How to use:
1. Download and inspect the files.
2. Run postgres_schema_ripple.sql in your Postgres instance to create the schema.
3. Configure Neo4j and place edges.csv into the import directory, then run neo4j_import_ripple.cypher.
4. Update credentials in ripple_simulation.py and run to test propagation.