
# ripple_simulation.py
# Extracts graph from Neo4j and runs a matrix-based ripple propagation simulation.
# Requirements:
#   pip install neo4j pandas numpy

from neo4j import GraphDatabase
import numpy as np
import pandas as pd

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"

def fetch_edges(driver):
    q = \"\"\"MATCH (a:Company)-[r]->(b:Company)
    RETURN a.ticker AS src, b.ticker AS tgt, coalesce(r.weight, 0.0) AS weight
    \"\"\"
    with driver.session() as session:
        res = session.run(q)
        rows = [dict(row) for row in res]
    return pd.DataFrame(rows)

def build_adjacency(edges_df):
    nodes = pd.Index(pd.unique(edges_df[['src','tgt']].values.ravel()))
    node_index = {node:i for i,node in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n,n), dtype=float)
    for _, r in edges_df.iterrows():
        i = node_index[r['src']]
        j = node_index[r['tgt']]
        A[j,i] = float(r['weight'])
    # normalize columns
    col_sums = A.sum(axis=0)
    col_sums[col_sums==0] = 1.0
    A = A / col_sums
    return A, nodes, node_index

def simulate_ripple(A, nodes, seed_ticker, node_index, seed_pct=-0.05, alpha=0.85, tol=1e-6, max_steps=100):
    n = A.shape[0]
    delta = np.zeros(n)
    delta[node_index[seed_ticker]] = seed_pct
    delta_t = delta.copy()
    for step in range(max_steps):
        delta_next = alpha * A.dot(delta_t)
        if np.linalg.norm(delta_next - delta_t) < tol:
            break
        delta_t = delta_next
    return pd.DataFrame({"ticker": nodes, "impact": delta_t}).sort_values("impact")

def main():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    edges_df = fetch_edges(driver)
    if edges_df.empty:
        print("No edges found. Ensure Neo4j has Company nodes and relationships.")
        return
    A, nodes, node_index = build_adjacency(edges_df)
    seed = "JPM"
    if seed not in node_index:
        seed = nodes[0]
        print(f"Seed {seed} not found. Using {seed} instead.")
    res = simulate_ripple(A, nodes, seed, node_index, seed_pct=-0.05, alpha=0.85)
    print(res.head(30))

if __name__ == '__main__':
    main()
