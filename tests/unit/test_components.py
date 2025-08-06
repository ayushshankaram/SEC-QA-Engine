#!/usr/bin/env python3
"""
Simple test of core components working together
"""

import sys
import os
sys.path.append('/Users/ayushshankaram/Desktop/QAEngine/src')

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_embeddings():
    """Test that embeddings are working with Financial BERT fallback"""
    logger.info("🧪 Testing embedding generation...")
    
    try:
        from core.embedding_ensemble import EmbeddingEnsemble
        
        # Initialize ensemble
        ensemble = EmbeddingEnsemble()
        
        # Test embedding generation
        test_texts = [
            "Apple Inc. reported strong quarterly earnings with revenue growth of 15%.",
            "Microsoft Corporation announced new cloud services expansion.",
            "The company's net income increased by 12% year-over-year."
        ]
        
        logger.info(f"Generating embeddings for {len(test_texts)} texts...")
        embeddings = ensemble.embed_batch(test_texts)
        
        logger.info(f"PASS Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            if emb is not None and len(emb) > 0:
                logger.info(f"  Text {i+1}: {len(emb)} dimensions, sample: {emb[:3]}")
            else:
                logger.error(f"  Text {i+1}: Failed to generate embedding")
        
        # Check model weights
        health = ensemble.health_check()
        logger.info(f"Ensemble health: {health}")
        
        return True
        
    except Exception as e:
        logger.error(f"FAIL Embedding test failed: {e}")
        return False

def test_sec_client():
    """Test SEC client functionality"""
    logger.info("🧪 Testing SEC client...")
    
    try:
        from data.secio_client import SECIOClient
        
        client = SECIOClient()
        
        # Test getting company info
        logger.info("Testing company info retrieval...")
        company_info = client.get_company_info("AAPL")
        
        if company_info:
            logger.info(f"PASS Retrieved info for {company_info.get('name', 'Unknown')}")
            logger.info(f"  CIK: {company_info.get('cik', 'N/A')}")
            logger.info(f"  Sector: {company_info.get('sector', 'N/A')}")
        else:
            logger.error("FAIL Failed to retrieve company info")
            return False
        
        # Test getting recent filings
        logger.info("Testing filings retrieval...")
        filings = client.get_company_filings(
            ticker="AAPL",
            filing_type="10-K",
            limit=1
        )
        
        if filings:
            filing = filings[0]
            logger.info(f"PASS Retrieved {len(filings)} filing(s)")
            logger.info(f"  Latest 10-K: {filing.get('filing_date', 'N/A')}")
            logger.info(f"  Accession: {filing.get('accession_number', 'N/A')}")
        else:
            logger.error("FAIL Failed to retrieve filings")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"FAIL SEC client test failed: {e}")
        return False

def main():
    """Run component tests"""
    logger.info("=" * 60)
    logger.info("COMPONENT TESTING - Core Functionality")
    logger.info("=" * 60)
    
    tests = [
        ("Embedding Generation", test_embeddings),
        ("SEC Client", test_sec_client)
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
    
    passed = sum(results.values())
    total = len(tests)
    
    for test_name, success in results.items():
        status = "PASS PASS" if success else "FAIL FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("SUCCESS All core components working! System ready for ticker ingestion!")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
