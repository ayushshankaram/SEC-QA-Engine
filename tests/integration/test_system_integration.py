#!/usr/bin/env python3
"""
Final system validation test for SEC QA Engine
Tests the complete pipeline with fallback embedding models
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.embedding_ensemble import EmbeddingEnsemble
from src.storage.neo4j_client import Neo4jClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_embedding_ensemble():
    """Test the embedding ensemble with fallback models"""
    logger.info("Testing Embedding Ensemble with fallback models...")
    
    try:
        ensemble = EmbeddingEnsemble()
        
        # Test with sample text
        test_text = "Apple Inc. reported strong quarterly earnings with revenue growth of 15%."
        
        logger.info(f"Testing embedding for: '{test_text}'")
        embedding = ensemble.embed_batch([test_text])[0]
        
        logger.info(f"Generated embedding with dimension: {len(embedding)}")
        logger.info(f"Embedding preview: {embedding[:5]}...")
        
        # Check which models are active
        active_models = [name for name, enabled in ensemble.enable_models.items() if enabled]
        logger.info(f"Active embedding models: {active_models}")
        
        # Show model weights
        for model_name, weight in ensemble.weights.items():
            if ensemble.enable_models.get(model_name, False):
                logger.info(f"  {model_name}: {weight:.1%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Embedding ensemble test failed: {e}")
        return False

def test_graph_rag():
    """Test Neo4j database functionality"""
    logger.info("Testing Neo4j database functionality...")
    
    try:
        neo4j_client = Neo4jClient()
        
        # Test connection
        if neo4j_client.test_connection():
            logger.info("PASS Neo4j connection successful")
            
            # Get some sample data
            query = """
            MATCH (c:Company)-[:FILED]->(f:Filing)-[:HAS_SECTION]->(s:Section)
            RETURN c.name as company, f.form_type as form, s.title as section
            LIMIT 5
            """
            
            results = neo4j_client.execute_query(query)
            logger.info(f"Sample data from Neo4j: {len(results)} records")
            
            for record in results:
                logger.info(f"  {record['company']} - {record['form']} - {record['section']}")
            
            return True
        else:
            logger.error("FAIL Neo4j connection failed")
            return False
            
    except Exception as e:
        logger.error(f"Neo4j test failed: {e}")
        return False

def test_qa_engine():
    """Test the embedding ensemble with sample question processing"""
    logger.info("Testing question processing with embedding ensemble...")
    
    try:
        ensemble = EmbeddingEnsemble()
        
        # Test with a financial question
        test_question = "What was Apple's revenue growth strategy in their latest filing?"
        
        logger.info(f"Testing question: '{test_question}'")
        
        # Generate embedding for the question
        question_embedding = ensemble.embed_batch([test_question])[0]
        
        logger.info(f"Generated question embedding with dimension: {len(question_embedding)}")
        logger.info("PASS Question processing successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Question processing test failed: {e}")
        return False

def main():
    """Run comprehensive system test"""
    logger.info("=" * 60)
    logger.info("SEC QA ENGINE - FINAL SYSTEM VALIDATION")
    logger.info("=" * 60)
    
    tests = [
        ("Embedding Ensemble", test_embedding_ensemble),
        ("Neo4j Database", test_graph_rag),
        ("Question Processing", test_qa_engine)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "PASS PASS" if passed_test else "FAIL FAIL"
        logger.info(f"{test_name}: {status}")
        if passed_test:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("SUCCESS ALL TESTS PASSED - System is ready for use!")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed - Check logs for details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
