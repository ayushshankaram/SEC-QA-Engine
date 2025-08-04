"""
Neo4j Graph RAG Database Client for SEC Filings
Advanced vector storage with graph relationships, supporting multi-embedding ensemble.
"""

import os
import asyncio
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple, Union
import json

import numpy as np
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Advanced Neo4j client for SEC filings with Graph RAG capabilities.
    Combines vector similarity search with graph relationships.
    """
    
    def __init__(self, uri: str = None, username: str = None, password: str = None, database: str = None):
        """Initialize Neo4j client with vector capabilities."""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j") 
        self.password = password or os.getenv("NEO4J_PASSWORD", "secfilings123")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Connection settings
        self.max_connection_lifetime = int(os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", 3600))
        self.max_connection_pool_size = int(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", 50))
        self.connection_acquisition_timeout = int(os.getenv("NEO4J_CONNECTION_ACQUISITION_TIMEOUT", 60))
        
        self.driver = None
        
        # Vector index configurations
        self.vector_indexes = {
            'voyage_embedding': {
                'dimension': 1536,
                'similarity': 'cosine'
            },
            'fine5_embedding': {
                'dimension': 1024, 
                'similarity': 'cosine'
            },
            'sparse_embedding': {
                'dimension': 50000,
                'similarity': 'cosine'
            },
            'ensemble_embedding': {
                'dimension': 1536,
                'similarity': 'cosine'
            }
        }
        
        # Statistics tracking
        self.stats = {
            'chunks_stored': 0,
            'queries_processed': 0,
            'companies_stored': 0,
            'filings_stored': 0,
            'last_update': None
        }
        
        logger.info(f"Neo4j Graph RAG client initialized for {self.uri}")
    
    def connect(self):
        """Establish connection to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=self.max_connection_lifetime,
                max_connection_pool_size=self.max_connection_pool_size,
                connection_acquisition_timeout=self.connection_acquisition_timeout
            )
            
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            
            logger.info("Neo4j connection established successfully")
            
            # Initialize database schema
            self._initialize_schema()
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def _initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT company_ticker IF NOT EXISTS FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
                    "CREATE CONSTRAINT filing_id IF NOT EXISTS FOR (f:Filing) REQUIRE f.filing_id IS UNIQUE",
                    "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
                    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (ch:Chunk) REQUIRE ch.chunk_id IS UNIQUE",
                    "CREATE CONSTRAINT fact_id IF NOT EXISTS FOR (xf:XBRLFact) REQUIRE xf.fact_id IS UNIQUE",
                    "CREATE CONSTRAINT executive_id IF NOT EXISTS FOR (e:Executive) REQUIRE e.executive_id IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                        logger.debug(f"Constraint created/verified")
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Constraint creation issue: {e}")
                
                # Create vector indexes
                self._create_vector_indexes(session)
                
                # Create regular indexes
                indexes = [
                    "CREATE INDEX company_name_idx IF NOT EXISTS FOR (c:Company) ON (c.company_name)",
                    "CREATE INDEX filing_date_idx IF NOT EXISTS FOR (f:Filing) ON (f.filing_date)",
                    "CREATE INDEX filing_type_idx IF NOT EXISTS FOR (f:Filing) ON (f.form_type)",
                    "CREATE INDEX section_name_idx IF NOT EXISTS FOR (s:Section) ON (s.section_name)",
                    "CREATE INDEX chunk_content_idx IF NOT EXISTS FOR (ch:Chunk) ON (ch.content)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                        logger.debug(f"Index created/verified")
                    except Exception as e:
                        if "already exists" not in str(e).lower():
                            logger.warning(f"Index creation issue: {e}")
                
            logger.info("Neo4j schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
    
    def _create_vector_indexes(self, session):
        """Create vector indexes for all embedding types."""
        vector_index_queries = [
            # Voyage Finance-2 embeddings
            """
            CREATE VECTOR INDEX section_voyage_idx IF NOT EXISTS
            FOR (s:Section) ON (s.voyage_embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            
            # Fin-E5 embeddings  
            """
            CREATE VECTOR INDEX section_fine5_idx IF NOT EXISTS
            FOR (s:Section) ON (s.fine5_embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1024,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            
            # Sparse embeddings (TF-IDF)
            """
            CREATE VECTOR INDEX section_sparse_idx IF NOT EXISTS
            FOR (s:Section) ON (s.sparse_embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 50000,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            
            # Ensemble embeddings
            """
            CREATE VECTOR INDEX section_ensemble_idx IF NOT EXISTS
            FOR (s:Section) ON (s.ensemble_embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            
            # Chunk vector indexes
            """
            CREATE VECTOR INDEX chunk_voyage_idx IF NOT EXISTS
            FOR (ch:Chunk) ON (ch.voyage_embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            
            """
            CREATE VECTOR INDEX chunk_ensemble_idx IF NOT EXISTS
            FOR (ch:Chunk) ON (ch.ensemble_embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 1536,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
        ]
        
        for query in vector_index_queries:
            try:
                session.run(query)
                logger.debug(f"Vector index created/verified")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Vector index creation issue: {e}")
    
    def create_company(self, ticker: str, company_name: str, industry: str = None, 
                      sector: str = None, description: str = None) -> str:
        """Create or update a company node."""
        if not self.driver:
            self.connect()
        
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MERGE (c:Company {ticker: $ticker})
                SET c.company_name = $company_name,
                    c.industry = $industry,
                    c.sector = $sector,
                    c.description = $description,
                    c.updated_at = datetime()
                ON CREATE SET c.created_at = datetime()
                RETURN c.ticker
                """
                
                result = session.run(query, 
                    ticker=ticker,
                    company_name=company_name,
                    industry=industry,
                    sector=sector,
                    description=description
                )
                
                record = result.single()
                if record:
                    self.stats['companies_stored'] += 1
                    logger.debug(f"Company created/updated: {ticker}")
                    return record["c.ticker"]
                    
        except Exception as e:
            logger.error(f"Error creating company {ticker}: {e}")
            return None
    
    def create_filing(self, filing_id: str, company_ticker: str, form_type: str,
                     filing_date: str, document_url: str = None, 
                     metadata: Dict = None) -> str:
        """Create a SEC filing node."""
        if not self.driver:
            self.connect()
        
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (c:Company {ticker: $company_ticker})
                MERGE (f:Filing {filing_id: $filing_id})
                SET f.form_type = $form_type,
                    f.filing_date = date($filing_date),
                    f.document_url = $document_url,
                    f.metadata = $metadata,
                    f.updated_at = datetime()
                ON CREATE SET f.created_at = datetime()
                MERGE (c)-[:FILED]->(f)
                RETURN f.filing_id
                """
                
                result = session.run(query,
                    filing_id=filing_id,
                    company_ticker=company_ticker,
                    form_type=form_type,
                    filing_date=filing_date,
                    document_url=document_url,
                    metadata=json.dumps(metadata) if metadata else None
                )
                
                record = result.single()
                if record:
                    self.stats['filings_stored'] += 1
                    logger.debug(f"Filing created: {filing_id}")
                    return record["f.filing_id"]
                    
        except Exception as e:
            logger.error(f"Error creating filing {filing_id}: {e}")
            return None

import os
import logging
from typing import List, Dict, Any, Optional, Union
from neo4j import GraphDatabase, Record
from neo4j.exceptions import Neo4jError
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j client for SEC filings knowledge graph operations."""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None, database: str = None):
        """Initialize Neo4j client with connection parameters."""
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database or os.getenv("NEO4J_DATABASE", "sec_filings")
        
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password),
                max_connection_lifetime=3600
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1 as test")
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Neo4jError as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Record]:
        """Execute a Cypher query and return results."""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters or {})
                return list(result)
        except Neo4jError as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    def create_indexes(self):
        """Create necessary indexes for performance."""
        indexes = [
            "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON c.ticker",
            "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON c.name",
            "CREATE INDEX filing_type IF NOT EXISTS FOR (f:Filing) ON f.filing_type",
            "CREATE INDEX filing_date IF NOT EXISTS FOR (f:Filing) ON f.filing_date",
            "CREATE INDEX section_type IF NOT EXISTS FOR (s:Section) ON s.section_type",
            "CREATE INDEX fact_concept IF NOT EXISTS FOR (f:Fact) ON f.concept",
            "CREATE INDEX executive_name IF NOT EXISTS FOR (e:Executive) ON e.name",
            "CREATE TEXT INDEX document_content IF NOT EXISTS FOR (d:Document) ON d.content"
        ]
        
        for index_query in indexes:
            try:
                self.execute_query(index_query)
                logger.info(f"Created index: {index_query.split('FOR')[0].split('INDEX')[1].strip()}")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")
    
    # Company Operations
    def create_company(self, ticker: str, name: str, sector: str = None, 
                      industry: str = None, **kwargs) -> str:
        """Create a company node."""
        query = """
        MERGE (c:Company {ticker: $ticker})
        SET c.name = $name,
            c.sector = $sector,
            c.industry = $industry,
            c.created_at = datetime(),
            c.updated_at = datetime()
        RETURN c.ticker as ticker
        """
        
        parameters = {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "industry": industry
        }
        parameters.update(kwargs)
        
        result = self.execute_query(query, parameters)
        return result[0]["ticker"] if result else None
    
    def get_company(self, ticker: str) -> Optional[Dict]:
        """Get company information by ticker."""
        query = """
        MATCH (c:Company {ticker: $ticker})
        RETURN c {.*} as company
        """
        
        result = self.execute_query(query, {"ticker": ticker})
        return dict(result[0]["company"]) if result else None
    
    # Filing Operations
    def create_filing(self, company_ticker: str, filing_type: str, filing_date: str,
                     accession_number: str, url: str = None, **kwargs) -> str:
        """Create a filing node and link to company."""
        query = """
        MATCH (c:Company {ticker: $company_ticker})
        MERGE (f:Filing {accession_number: $accession_number})
        SET f.filing_type = $filing_type,
            f.filing_date = date($filing_date),
            f.url = $url,
            f.created_at = datetime(),
            f.updated_at = datetime()
        MERGE (c)-[:FILED]->(f)
        RETURN f.accession_number as accession_number
        """
        
        parameters = {
            "company_ticker": company_ticker,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "accession_number": accession_number,
            "url": url
        }
        parameters.update(kwargs)
        
        result = self.execute_query(query, parameters)
        return result[0]["accession_number"] if result else None
    
    # Section Operations
    def create_section(self, accession_number: str, section_type: str, 
                      content: str, embeddings: List[float] = None, **kwargs) -> int:
        """Create a section node and link to filing."""
        query = """
        MATCH (f:Filing {accession_number: $accession_number})
        CREATE (s:Section)
        SET s.section_type = $section_type,
            s.content = $content,
            s.embeddings = $embeddings,
            s.created_at = datetime(),
            s.id = id(s)
        MERGE (f)-[:CONTAINS_SECTION]->(s)
        RETURN s.id as section_id
        """
        
        parameters = {
            "accession_number": accession_number,
            "section_type": section_type,
            "content": content,
            "embeddings": embeddings
        }
        parameters.update(kwargs)
        
        result = self.execute_query(query, parameters)
        return result[0]["section_id"] if result else None
    
    # XBRL Facts Operations
    def create_xbrl_fact(self, accession_number: str, concept: str, value: float,
                        unit: str, period: str, **kwargs) -> int:
        """Create an XBRL fact node and link to filing."""
        query = """
        MATCH (f:Filing {accession_number: $accession_number})
        CREATE (fact:Fact)
        SET fact.concept = $concept,
            fact.value = $value,
            fact.unit = $unit,
            fact.period = $period,
            fact.created_at = datetime(),
            fact.id = id(fact)
        MERGE (f)-[:REPORTS_FACT]->(fact)
        RETURN fact.id as fact_id
        """
        
        parameters = {
            "accession_number": accession_number,
            "concept": concept,
            "value": value,
            "unit": unit,
            "period": period
        }
        parameters.update(kwargs)
        
        result = self.execute_query(query, parameters)
        return result[0]["fact_id"] if result else None
    
    # Executive Operations
    def create_executive(self, name: str, title: str, company_ticker: str = None) -> int:
        """Create an executive node."""
        query = """
        CREATE (e:Executive)
        SET e.name = $name,
            e.title = $title,
            e.created_at = datetime(),
            e.id = id(e)
        """
        
        if company_ticker:
            query += """
            WITH e
            MATCH (c:Company {ticker: $company_ticker})
            MERGE (e)-[:WORKS_FOR]->(c)
            """
        
        query += " RETURN e.id as executive_id"
        
        parameters = {
            "name": name,
            "title": title,
            "company_ticker": company_ticker
        }
        
        result = self.execute_query(query, parameters)
        return result[0]["executive_id"] if result else None
    
    # Trading Operations
    def create_trading_transaction(self, executive_id: int, company_ticker: str,
                                 transaction_date: str, shares: float, price: float,
                                 transaction_type: str) -> int:
        """Create a trading transaction relationship."""
        query = """
        MATCH (e:Executive) WHERE id(e) = $executive_id
        MATCH (c:Company {ticker: $company_ticker})
        MERGE (e)-[t:TRADED {
            date: date($transaction_date),
            shares: $shares,
            price: $price,
            transaction_type: $transaction_type,
            created_at: datetime()
        }]->(c)
        RETURN id(t) as transaction_id
        """
        
        parameters = {
            "executive_id": executive_id,
            "company_ticker": company_ticker,
            "transaction_date": transaction_date,
            "shares": shares,
            "price": price,
            "transaction_type": transaction_type
        }
        
        result = self.execute_query(query, parameters)
        return result[0]["transaction_id"] if result else None
    
    # Search Operations
    def semantic_search_sections(self, query_embedding: List[float], 
                               company_tickers: List[str] = None,
                               filing_types: List[str] = None,
                               limit: int = 10) -> List[Dict]:
        """Perform semantic search on document sections."""
        cypher_query = """
        MATCH (c:Company)-[:FILED]->(f:Filing)-[:CONTAINS_SECTION]->(s:Section)
        WHERE s.embeddings IS NOT NULL
        """
        
        parameters = {"query_embedding": query_embedding, "limit": limit}
        
        if company_tickers:
            cypher_query += " AND c.ticker IN $company_tickers"
            parameters["company_tickers"] = company_tickers
        
        if filing_types:
            cypher_query += " AND f.filing_type IN $filing_types"
            parameters["filing_types"] = filing_types
        
        cypher_query += """
        WITH s, c, f,
             reduce(dot = 0.0, i IN range(0, size($query_embedding)-1) | 
                dot + $query_embedding[i] * s.embeddings[i]) as similarity
        ORDER BY similarity DESC
        LIMIT $limit
        RETURN s.content as content,
               s.section_type as section_type,
               c.ticker as ticker,
               c.name as company_name,
               f.filing_type as filing_type,
               f.filing_date as filing_date,
               f.accession_number as accession_number,
               similarity
        """
        
        results = self.execute_query(cypher_query, parameters)
        return [dict(record) for record in results]
    
    def text_search_sections(self, search_text: str, 
                           company_tickers: List[str] = None,
                           filing_types: List[str] = None,
                           limit: int = 10) -> List[Dict]:
        """Perform text search on document sections."""
        cypher_query = """
        CALL db.index.fulltext.queryNodes("document_content", $search_text)
        YIELD node as s, score
        MATCH (c:Company)-[:FILED]->(f:Filing)-[:CONTAINS_SECTION]->(s)
        """
        
        parameters = {"search_text": search_text, "limit": limit}
        
        if company_tickers:
            cypher_query += " WHERE c.ticker IN $company_tickers"
            parameters["company_tickers"] = company_tickers
        
        if filing_types:
            if company_tickers:
                cypher_query += " AND f.filing_type IN $filing_types"
            else:
                cypher_query += " WHERE f.filing_type IN $filing_types"
            parameters["filing_types"] = filing_types
        
        cypher_query += """
        RETURN s.content as content,
               s.section_type as section_type,
               c.ticker as ticker,
               c.name as company_name,
               f.filing_type as filing_type,
               f.filing_date as filing_date,
               f.accession_number as accession_number,
               score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        results = self.execute_query(cypher_query, parameters)
        return [dict(record) for record in results]
    
    # Analytics Operations
    def get_filing_counts_by_company(self) -> List[Dict]:
        """Get filing counts grouped by company."""
        query = """
        MATCH (c:Company)-[:FILED]->(f:Filing)
        RETURN c.ticker as ticker,
               c.name as company_name,
               c.sector as sector,
               count(f) as filing_count
        ORDER BY filing_count DESC
        """
        
        results = self.execute_query(query)
        return [dict(record) for record in results]
    
    def get_system_stats(self) -> Dict:
        """Get overall system statistics."""
        query = """
        MATCH (c:Company) WITH count(c) as companies
        MATCH (f:Filing) WITH companies, count(f) as filings
        MATCH (s:Section) WITH companies, filings, count(s) as sections
        MATCH (fact:Fact) WITH companies, filings, sections, count(fact) as facts
        MATCH (e:Executive) WITH companies, filings, sections, facts, count(e) as executives
        MATCH ()-[r]->() WITH companies, filings, sections, facts, executives, count(r) as relationships
        RETURN companies, filings, sections, facts, executives, relationships
        """
        
        result = self.execute_query(query)
        return dict(result[0]) if result else {}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the database."""
        try:
            stats = self.get_system_stats()
            
            # Check if indexes exist
            index_query = "SHOW INDEXES YIELD name"
            indexes = [record["name"] for record in self.execute_query(index_query)]
            
            return {
                "status": "healthy",
                "connection": "active",
                "database": self.database,
                "statistics": stats,
                "indexes": len(indexes),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def clear_database(self, confirm: bool = False):
        """Clear all data from the database. Use with caution!"""
        if not confirm:
            raise ValueError("Must confirm database clearing by setting confirm=True")
        
        query = "MATCH (n) DETACH DELETE n"
        self.execute_query(query)
        logger.warning("Database cleared - all data deleted!")


# Context manager support
class Neo4jClient(Neo4jClient):
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
