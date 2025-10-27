"""Graph construction and adjacency matrix operations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
from datetime import datetime
import logging

from src.database import neo4j_manager, pg_manager
from config.settings import config

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builder for constructing stock dependency graphs."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.GraphBuilder")
    
    def build_adjacency_matrix(self, 
                             tickers: Optional[List[str]] = None,
                             relation_types: Optional[List[str]] = None,
                             min_weight: float = 0.0) -> Tuple[np.ndarray, List[str]]:
        """Build adjacency matrix from Neo4j graph data."""
        try:
            # Get graph data from Neo4j
            edges_df = self._get_graph_edges(tickers, relation_types, min_weight)
            
            if edges_df.empty:
                self.logger.warning("No edges found for adjacency matrix construction")
                return np.array([]), []
            
            # Get unique tickers
            all_tickers = sorted(set(edges_df['source'].unique()) | set(edges_df['target'].unique()))
            
            # Create ticker to index mapping
            ticker_to_idx = {ticker: idx for idx, ticker in enumerate(all_tickers)}
            n = len(all_tickers)
            
            # Initialize adjacency matrix
            adj_matrix = np.zeros((n, n), dtype=float)
            
            # Fill adjacency matrix
            for _, edge in edges_df.iterrows():
                source_idx = ticker_to_idx[edge['source']]
                target_idx = ticker_to_idx[edge['target']]
                weight = float(edge['weight'])
                
                # Set adjacency value (target <- source influence)
                adj_matrix[target_idx, source_idx] = weight
            
            # Normalize columns (each column sums to 1 for propagation)
            col_sums = adj_matrix.sum(axis=0)
            col_sums[col_sums == 0] = 1.0  # Avoid division by zero
            adj_matrix = adj_matrix / col_sums
            
            return adj_matrix, all_tickers
            
        except Exception as e:
            self.logger.error(f"Error building adjacency matrix: {e}")
            return np.array([]), []
    
    def build_networkx_graph(self, 
                           tickers: Optional[List[str]] = None,
                           relation_types: Optional[List[str]] = None,
                           min_weight: float = 0.0) -> nx.DiGraph:
        """Build NetworkX directed graph from database."""
        try:
            # Get edges data
            edges_df = self._get_graph_edges(tickers, relation_types, min_weight)
            
            # Create directed graph
            G = nx.DiGraph()
            
            if edges_df.empty:
                return G
            
            # Add nodes with attributes
            node_attrs = self._get_node_attributes(tickers)
            for ticker, attrs in node_attrs.items():
                G.add_node(ticker, **attrs)
            
            # Add edges with weights
            for _, edge in edges_df.iterrows():
                G.add_edge(
                    edge['source'], 
                    edge['target'],
                    weight=edge['weight'],
                    relation_type=edge['relation_type'],
                    **edge.get('properties', {})
                )
            
            return G
            
        except Exception as e:
            self.logger.error(f"Error building NetworkX graph: {e}")
            return nx.DiGraph()
    
    def update_graph_from_correlations(self, correlations_df: pd.DataFrame) -> Dict:
        """Update Neo4j graph with correlation data."""
        try:
            if correlations_df.empty:
                return {'relationships_created': 0}
            
            relationships_created = 0
            
            for _, corr in correlations_df.iterrows():
                # Create relationship in Neo4j
                result = neo4j_manager.create_relationship(
                    source_ticker=corr['source_ticker'],
                    target_ticker=corr['target_ticker'],
                    rel_type=corr.get('relation_type', 'CORRELATED_WITH'),
                    weight=float(corr['correlation']),
                    correlation_type=corr.get('correlation_type', 'pearson'),
                    window_days=corr.get('window_days'),
                    source_date=corr.get('date', datetime.now().date()).isoformat(),
                    source='correlation_analysis'
                )
                
                if result.get('relationships_created', 0) > 0:
                    relationships_created += 1
            
            self.logger.info(f"Created {relationships_created} correlation relationships")
            return {'relationships_created': relationships_created}
            
        except Exception as e:
            self.logger.error(f"Error updating graph from correlations: {e}")
            return {'relationships_created': 0}
    
    def update_graph_from_ownership(self, ownership_df: pd.DataFrame) -> Dict:
        """Update Neo4j graph with ownership data."""
        try:
            if ownership_df.empty:
                return {'relationships_created': 0}
            
            relationships_created = 0
            
            for _, ownership in ownership_df.iterrows():
                # Create ownership relationship
                result = neo4j_manager.create_relationship(
                    source_ticker=ownership['source_ticker'],
                    target_ticker=ownership['target_ticker'],
                    rel_type='OWNS',
                    weight=float(ownership.get('weight', 0.0)),
                    shares=ownership.get('shares'),
                    market_value=ownership.get('market_value'),
                    ownership_percent=ownership.get('ownership_percent'),
                    filing_date=ownership.get('filing_date'),
                    source=ownership.get('source', 'edgar')
                )
                
                if result.get('relationships_created', 0) > 0:
                    relationships_created += 1
            
            self.logger.info(f"Created {relationships_created} ownership relationships")
            return {'relationships_created': relationships_created}
            
        except Exception as e:
            self.logger.error(f"Error updating graph from ownership: {e}")
            return {'relationships_created': 0}
    
    def compute_centrality_metrics(self, 
                                 tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """Compute various centrality metrics for the graph."""
        try:
            # Build NetworkX graph
            G = self.build_networkx_graph(tickers)
            
            if G.number_of_nodes() == 0:
                return pd.DataFrame()
            
            # Compute centrality metrics
            centralities = []
            
            # Degree centrality
            in_degree_cent = nx.in_degree_centrality(G)
            out_degree_cent = nx.out_degree_centrality(G)
            
            # Betweenness centrality
            betweenness_cent = nx.betweenness_centrality(G)
            
            # Eigenvector centrality (if graph is strongly connected)
            try:
                eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
            except:
                eigenvector_cent = {node: 0.0 for node in G.nodes()}
            
            # PageRank
            pagerank_cent = nx.pagerank(G, alpha=0.85)
            
            # Closeness centrality
            closeness_cent = nx.closeness_centrality(G)
            
            # Compile results
            for node in G.nodes():
                centralities.append({
                    'ticker': node,
                    'in_degree_centrality': in_degree_cent.get(node, 0.0),
                    'out_degree_centrality': out_degree_cent.get(node, 0.0),
                    'betweenness_centrality': betweenness_cent.get(node, 0.0),
                    'eigenvector_centrality': eigenvector_cent.get(node, 0.0),
                    'pagerank': pagerank_cent.get(node, 0.0),
                    'closeness_centrality': closeness_cent.get(node, 0.0)
                })
            
            return pd.DataFrame(centralities)
            
        except Exception as e:
            self.logger.error(f"Error computing centrality metrics: {e}")
            return pd.DataFrame()
    
    def detect_communities(self, 
                         tickers: Optional[List[str]] = None,
                         method: str = 'louvain') -> pd.DataFrame:
        """Detect communities in the stock graph."""
        try:
            # Build NetworkX graph
            G = self.build_networkx_graph(tickers)
            
            if G.number_of_nodes() == 0:
                return pd.DataFrame()
            
            # Convert to undirected for community detection
            G_undirected = G.to_undirected()
            
            communities = []
            
            if method == 'louvain':
                # Use NetworkX community detection (simplified)
                # In production, you'd use python-louvain library
                import networkx.algorithms.community as nx_comm
                community_sets = nx_comm.greedy_modularity_communities(G_undirected)
                
                for i, community in enumerate(community_sets):
                    for ticker in community:
                        communities.append({
                            'ticker': ticker,
                            'community_id': i,
                            'method': method,
                            'community_size': len(community)
                        })
            
            return pd.DataFrame(communities)
            
        except Exception as e:
            self.logger.error(f"Error detecting communities: {e}")
            return pd.DataFrame()
    
    def get_graph_statistics(self, tickers: Optional[List[str]] = None) -> Dict:
        """Get basic statistics about the graph."""
        try:
            G = self.build_networkx_graph(tickers)
            
            if G.number_of_nodes() == 0:
                return {}
            
            stats = {
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'density': nx.density(G),
                'is_connected': nx.is_strongly_connected(G),
                'num_strongly_connected_components': nx.number_strongly_connected_components(G),
                'num_weakly_connected_components': nx.number_weakly_connected_components(G),
                'average_clustering': nx.average_clustering(G.to_undirected()),
            }
            
            # Additional metrics
            if G.number_of_nodes() > 1:
                try:
                    stats['diameter'] = nx.diameter(G.to_undirected())
                except:
                    stats['diameter'] = None
                
                stats['average_shortest_path_length'] = nx.average_shortest_path_length(G.to_undirected())
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error computing graph statistics: {e}")
            return {}
    
    def _get_graph_edges(self, 
                        tickers: Optional[List[str]] = None,
                        relation_types: Optional[List[str]] = None,
                        min_weight: float = 0.0) -> pd.DataFrame:
        """Get edges from Neo4j with optional filtering."""
        try:
            # Build Cypher query
            conditions = []
            if tickers:
                ticker_list = "', '".join(tickers)
                conditions.append(f"s.ticker IN ['{ticker_list}'] AND t.ticker IN ['{ticker_list}']")
            
            if relation_types:
                rel_conditions = " OR ".join([f"type(r) = '{rt}'" for rt in relation_types])
                conditions.append(f"({rel_conditions})")
            
            if min_weight > 0:
                conditions.append(f"coalesce(r.weight, 0.0) >= {min_weight}")
            
            where_clause = " AND ".join(conditions) if conditions else "true"
            
            query = f"""
            MATCH (s:Company)-[r]->(t:Company)
            WHERE {where_clause}
            RETURN s.ticker AS source, t.ticker AS target, 
                   coalesce(r.weight, 0.0) AS weight,
                   type(r) AS relation_type,
                   properties(r) AS properties
            """
            
            return pd.DataFrame(neo4j_manager.execute_query(query))
            
        except Exception as e:
            self.logger.error(f"Error getting graph edges: {e}")
            return pd.DataFrame()
    
    def _get_node_attributes(self, tickers: Optional[List[str]] = None) -> Dict:
        """Get node attributes from PostgreSQL."""
        try:
            if tickers:
                ticker_filter = f"WHERE ticker = ANY(ARRAY{tickers})"
            else:
                ticker_filter = ""
            
            query = f"""
            SELECT ticker, name, sector, industry, market_cap, country
            FROM companies
            {ticker_filter}
            """
            
            companies_df = pg_manager.read_dataframe(query)
            
            # Convert to dictionary format
            node_attrs = {}
            for _, company in companies_df.iterrows():
                node_attrs[company['ticker']] = {
                    'name': company.get('name', ''),
                    'sector': company.get('sector', ''),
                    'industry': company.get('industry', ''),
                    'market_cap': company.get('market_cap'),
                    'country': company.get('country', '')
                }
            
            return node_attrs
            
        except Exception as e:
            self.logger.error(f"Error getting node attributes: {e}")
            return {}
    
    def export_graph_to_formats(self, 
                              output_path: str,
                              tickers: Optional[List[str]] = None,
                              formats: List[str] = ['gexf', 'graphml']) -> Dict[str, str]:
        """Export graph to various formats for external analysis."""
        try:
            G = self.build_networkx_graph(tickers)
            
            if G.number_of_nodes() == 0:
                return {}
            
            exported_files = {}
            
            for fmt in formats:
                filename = f"{output_path}.{fmt}"
                
                if fmt == 'gexf':
                    nx.write_gexf(G, filename)
                elif fmt == 'graphml':
                    nx.write_graphml(G, filename)
                elif fmt == 'gml':
                    nx.write_gml(G, filename)
                elif fmt == 'json':
                    from networkx.readwrite import json_graph
                    import json
                    data = json_graph.node_link_data(G)
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                
                exported_files[fmt] = filename
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Error exporting graph: {e}")
            return {}