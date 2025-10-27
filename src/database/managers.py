"""Database connection managers for PostgreSQL and Neo4j."""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager, contextmanager
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from neo4j import GraphDatabase, Driver, Session
import redis
from config.settings import config

logger = logging.getLogger(__name__)


class PostgreSQLManager:
    """PostgreSQL connection manager with connection pooling."""
    
    def __init__(self, min_conn: int = 1, max_conn: int = 10):
        self.connection_pool: Optional[SimpleConnectionPool] = None
        self.min_conn = min_conn
        self.max_conn = max_conn
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            self.connection_pool = SimpleConnectionPool(
                self.min_conn,
                self.max_conn,
                host=config.database.postgres_host,
                port=config.database.postgres_port,
                database=config.database.postgres_db,
                user=config.database.postgres_user,
                password=config.database.postgres_password
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL connection pool: {e}")
            logger.warning("PostgreSQL operations will not be available")
            self.connection_pool = None
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        if not self.connection_pool:
            raise RuntimeError("PostgreSQL connection pool not initialized")
        
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute a SELECT query and return results."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
    
    def execute_non_query(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an INSERT/UPDATE/DELETE query."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount
    
    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute a query with multiple parameter sets."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
    
    def read_dataframe(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute query and return pandas DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append', batch_size: int = 1000) -> int:
        """Insert DataFrame into PostgreSQL table in batches."""
        if df.empty:
            logger.warning("Attempting to insert empty DataFrame")
            return 0
            
        # Use SQLAlchemy engine for pandas compatibility
        from sqlalchemy import create_engine
        
        engine_url = config.postgres_url
        engine = create_engine(engine_url)
        
        try:
            total_rows = 0
            # Insert in batches to avoid parameter limits
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                rows_inserted = batch_df.to_sql(table_name, engine, if_exists=if_exists, index=False, method='multi')
                total_rows += len(batch_df)
                logger.info(f"Inserted batch {i//batch_size + 1}: {len(batch_df)} rows")
                # Only use 'append' for subsequent batches
                if_exists = 'append'
            
            logger.info(f"Successfully inserted {total_rows} total rows into {table_name}")
            return total_rows
        except Exception as e:
            logger.error(f"Failed to insert DataFrame: {e}")
            raise


class Neo4jManager:
    """Neo4j connection manager."""
    
    def __init__(self):
        self.driver: Optional[Driver] = None
        self._initialize_driver()
    
    def _initialize_driver(self):
        """Initialize Neo4j driver."""
        try:
            self.driver = GraphDatabase.driver(
                config.database.neo4j_uri,
                auth=(config.database.neo4j_user, config.database.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j driver initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Neo4j driver: {e}")
            logger.warning("Neo4j operations will not be available")
            self.driver = None
    
    @contextmanager
    def get_session(self, database: Optional[str] = None):
        """Get a Neo4j session."""
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        
        session = None
        try:
            session = self.driver.session(database=database or config.database.neo4j_database)
            yield session
        except Exception as e:
            logger.error(f"Neo4j session error: {e}")
            raise
        finally:
            if session:
                session.close()
    
    def execute_query(self, query: str, parameters: Optional[Dict] = None, database: Optional[str] = None) -> List[Dict]:
        """Execute a Cypher query and return results."""
        with self.get_session(database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def execute_write_query(self, query: str, parameters: Optional[Dict] = None, database: Optional[str] = None) -> Dict:
        """Execute a write query and return summary."""
        with self.get_session(database) as session:
            result = session.run(query, parameters or {})
            summary = result.consume()
            return {
                'nodes_created': summary.counters.nodes_created,
                'nodes_deleted': summary.counters.nodes_deleted,
                'relationships_created': summary.counters.relationships_created,
                'relationships_deleted': summary.counters.relationships_deleted,
                'properties_set': summary.counters.properties_set
            }
    
    def create_company_node(self, ticker: str, name: str, **properties) -> Dict:
        """Create or update a company node."""
        query = """
        MERGE (c:Company {ticker: $ticker})
        SET c.name = $name, c.updated_at = timestamp()
        SET c += $properties
        RETURN c
        """
        return self.execute_write_query(query, {
            'ticker': ticker,
            'name': name,
            'properties': properties
        })
    
    def create_relationship(self, source_ticker: str, target_ticker: str, 
                          rel_type: str, weight: float, **properties) -> Dict:
        """Create or update a relationship between companies."""
        query = f"""
        MATCH (s:Company {{ticker: $source_ticker}})
        MATCH (t:Company {{ticker: $target_ticker}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r.weight = $weight, r.updated_at = timestamp()
        SET r += $properties
        RETURN r
        """
        return self.execute_write_query(query, {
            'source_ticker': source_ticker,
            'target_ticker': target_ticker,
            'weight': weight,
            'properties': properties
        })
    
    def get_adjacency_matrix(self) -> pd.DataFrame:
        """Get adjacency matrix representation of the graph."""
        query = """
        MATCH (s:Company)-[r]->(t:Company)
        RETURN s.ticker AS source, t.ticker AS target, 
               coalesce(r.weight, 0.0) AS weight
        """
        results = self.execute_query(query)
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        # Create adjacency matrix
        tickers = sorted(set(df['source'].unique()) | set(df['target'].unique()))
        adj_matrix = pd.DataFrame(0.0, index=tickers, columns=tickers)
        
        for _, row in df.iterrows():
            adj_matrix.loc[row['target'], row['source']] = row['weight']
        
        return adj_matrix
    
    def close(self):
        """Close the driver."""
        if self.driver:
            self.driver.close()


class RedisManager:
    """Redis connection manager for caching."""
    
    def __init__(self):
        self.client: Optional[redis.Redis] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Redis client."""
        try:
            self.client = redis.Redis(
                host=config.database.redis_host,
                port=config.database.redis_port,
                db=config.database.redis_db,
                password=config.database.redis_password,
                decode_responses=True
            )
            # Test connection
            self.client.ping()
            logger.info("Redis client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis client: {e}")
            logger.warning("Redis caching will not be available")
            self.client = None
    
    def get(self, key: str) -> Optional[str]:
        """Get value by key."""
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set key-value pair with optional expiration."""
        try:
            return self.client.set(key, value, ex=ex)
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key."""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False


# Global database managers
pg_manager = PostgreSQLManager()
neo4j_manager = Neo4jManager()
redis_manager = RedisManager()