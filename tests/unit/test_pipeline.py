#!/usr/bin/env python3
"""
Test SEC Pipeline with a few tickers to verify the system works end-to-end
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.enhanced_sec_pipeline import EnhancedSECPipeline, get_default_companies_by_sector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_single_ticker(pipeline, ticker):
    """Test pipeline with a single ticker"""
    logger.info(f"🧪 Testing {ticker}...")
    
    try:
        result = pipeline.ingest_company_filings(
            ticker=ticker,
            filing_types=["10-K", "10-Q"],  # Just test these two types
            max_filings=3,  # Keep it small for testing
            start_date="2023-01-01"  # Recent filings
        )
        
        if result["success"]:
            logger.info(f"PASS {ticker} successful: {result['filings_processed']}/{result['total_filings']} filings")
            return True
        else:
            logger.error(f"FAIL {ticker} failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"FAIL {ticker} crashed: {e}")
        return False

def test_pipeline():
    """Test the pipeline with a few tickers"""
    logger.info("=" * 60)
    logger.info("SEC PIPELINE TEST - TESTING INDIVIDUAL TICKERS")
    logger.info("=" * 60)
    
    # Test with a couple tickers first
    test_tickers = ["AAPL", "MSFT"]
    
    try:
        # Initialize pipeline
        logger.info("Initializing SEC pipeline...")
        pipeline = EnhancedSECPipeline()
        
        # Test health check (skip Neo4j for now)
        logger.info("Testing embedding ensemble...")
        health = pipeline.embedding_ensemble.health_check()
        logger.info(f"Embedding health: {health.get('status', 'unknown')}")
        
        # Test individual tickers
        results = {}
        for ticker in test_tickers:
            results[ticker] = test_single_ticker(pipeline, ticker)
        
        # Summary
        successful = sum(results.values())
        logger.info(f"\nRESULTS Results: {successful}/{len(test_tickers)} tickers successful")
        
        for ticker, success in results.items():
            status = "PASS PASS" if success else "FAIL FAIL"
            logger.info(f"  {ticker}: {status}")
        
        if successful == len(test_tickers):
            logger.info("SUCCESS All test tickers passed! Ready to ingest all 15 tickers.")
            return True
        else:
            logger.warning("⚠️ Some tickers failed. Check logs for details.")
            return False
            
    except Exception as e:
        logger.error(f"Pipeline test crashed: {e}")
        return False

def ingest_all_companies():
    """Ingest all 15 companies from the assignment"""
    logger.info("=" * 60)
    logger.info("SEC PIPELINE - INGESTING ALL 15 COMPANIES")
    logger.info("=" * 60)
    
    try:
        # Initialize pipeline
        pipeline = EnhancedSECPipeline()
        
        # Get all companies from assignment
        companies = get_default_companies_by_sector()
        logger.info(f"Companies to process: {companies}")
        
        # Bulk ingest with optimized settings
        result = pipeline.bulk_ingest_companies(
            companies=companies,
            filing_counts={
                "10-K": 1,    # 1 annual report per company
                "10-Q": 2,    # 2 quarterly reports per company
                "8-K": 1,     # 1 current report per company
                "DEF 14A": 1  # 1 proxy statement per company
            },
            date_range=("2023-01-01", "2024-12-31")  # Recent filings
        )
        
        if result["success"]:
            logger.info("SUCCESS Bulk ingestion completed successfully!")
            logger.info(f"Companies processed: {result['companies_successful']}/{result['companies_total']}")
            logger.info(f"Total filings processed: {result['filings_processed']}")
            logger.info(f"Processing time: {result['processing_time']:.2f} seconds")
            
            # Show detailed results
            for sector, sector_results in result["results_by_sector"].items():
                successful_in_sector = len([r for r in sector_results if r.get("success", False)])
                logger.info(f"  {sector}: {successful_in_sector}/{len(sector_results)} companies")
            
            return True
        else:
            logger.error("FAIL Bulk ingestion failed")
            return False
            
    except Exception as e:
        logger.error(f"Bulk ingestion crashed: {e}")
        return False

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SEC Pipeline")
    parser.add_argument("--mode", choices=["test", "ingest", "both"], default="both",
                       help="Mode: test individual tickers, ingest all, or both")
    
    args = parser.parse_args()
    
    if args.mode in ["test", "both"]:
        logger.info("Starting ticker tests...")
        test_success = test_pipeline()
        
        if not test_success and args.mode == "both":
            logger.error("Test failed, skipping bulk ingestion")
            return False
    
    if args.mode in ["ingest", "both"]:
        logger.info("Starting bulk ingestion...")
        ingest_success = ingest_all_companies()
        return ingest_success
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
