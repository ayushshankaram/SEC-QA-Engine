#!/usr/bin/env python3
"""
Simple Neo4j ingestion script with direct credentials
"""

import sys
import os
sys.path.append('/Users/ayushshankaram/Desktop/QAEngine/src')

import logging
from neo4j import GraphDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_and_store_data():
    """Test Neo4j connection and store sample data"""
    
    logger.info("🔍 Testing Neo4j connection and storing sample data...")
    
    try:
        # Connect directly with known credentials
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'secfilings123'))
        
        with driver.session() as session:
            # Check current state
            result = session.run('MATCH (n) RETURN count(n) as total_nodes')
            total_nodes = result.single()['total_nodes']
            logger.info(f"Current nodes in database: {total_nodes}")
            
            # Create a simple test company with embedding
            test_embedding = [0.1] * 1024  # Simple test embedding
            
            query = """
            MERGE (c:Company {ticker: $ticker})
            SET c.name = $name,
                c.cik = $cik,
                c.sector = $sector
            RETURN c
            """
            
            result = session.run(query, {
                'ticker': 'TEST',
                'name': 'Test Company Inc.',
                'cik': '1234567890',
                'sector': 'Technology'
            })
            company = result.single()['c']
            logger.info(f"✅ Created test company: {company['name']}")
            
            # Create a test section with embedding
            section_query = """
            MATCH (c:Company {ticker: $ticker})
            CREATE (s:Section {
                section_type: $section_type,
                content: $content,
                embedding: $embedding,
                chunk_index: $chunk_index
            })
            CREATE (c)-[:HAS_SECTION]->(s)
            RETURN s
            """
            
            result = session.run(section_query, {
                'ticker': 'TEST',
                'section_type': 'Business Overview',
                'content': 'This is a test section with sample content for testing the embedding storage.',
                'embedding': test_embedding,
                'chunk_index': 0
            })
            section = result.single()['s']
            logger.info(f"✅ Created test section with embedding")
            
            # Verify embeddings are stored
            result = session.run('MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) as embedding_count')
            embedding_count = result.single()['embedding_count']
            logger.info(f"Nodes with embeddings: {embedding_count}")
            
            # Check total nodes now
            result = session.run('MATCH (n) RETURN count(n) as total_nodes')
            total_nodes = result.single()['total_nodes']
            logger.info(f"Total nodes after test: {total_nodes}")
            
        driver.close()
        logger.info("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_and_store_data()
    sys.exit(0 if success else 1)
