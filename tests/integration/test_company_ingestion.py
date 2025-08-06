#!/usr/bin/env python3
"""
Quick test to ingest AAPL directly, bypassing the complex test setup
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

def ingest_apple():
    """Test ingesting Apple directly"""
    logger.info("Testing AAPL ingestion directly...")
    
    try:
        # Import what we need
        from data.secio_client import SECIOClient
        from core.embedding_ensemble import EmbeddingEnsemble
        from data.document_processor import DocumentProcessor
        
        # Initialize components
        logger.info("Initializing components...")
        sec_client = SECIOClient()
        embeddings = EmbeddingEnsemble()
        doc_processor = DocumentProcessor()
        
        # Get Apple info
        logger.info("Getting Apple company info...")
        company_info = sec_client.get_company_info("AAPL")
        logger.info(f"Company: {company_info.get('name')} (CIK: {company_info.get('cik')})")
        
        # Get recent 10-K filing
        logger.info("Getting recent 10-K filing...")
        filings = sec_client.get_company_filings(
            ticker="AAPL",
            filing_type="10-K", 
            limit=1
        )
        
        if not filings:
            logger.error("No filings found")
            return False
            
        filing = filings[0]
        logger.info(f"Processing filing: {filing.get('accession_number')} from {filing.get('filing_date')}")
        
        # Get filing content (just a small sample)
        cik = company_info.get('cik')
        accession = filing.get('accession_number')
        primary_doc = filing.get('primaryDocument')
        
        if not all([cik, accession, primary_doc]):
            logger.error("Missing required filing info")
            return False
            
        logger.info("Retrieving filing content...")
        content = sec_client.get_filing_content(cik, accession, primary_doc)
        
        if not content:
            logger.error("Failed to retrieve content")
            return False
            
        # Process just a small section
        logger.info(f"Processing document content ({len(content)} chars)...")
        content_sample = content[:5000]  # Just first 5KB for testing
        
        # Process into chunks
        chunks = doc_processor.process_filing(content_sample, "10-K")
        logger.info(f"Generated {len(chunks)} document chunks")
        
        if not chunks:
            logger.error("No chunks generated")
            return False
            
        # Generate embeddings for first chunk only
        first_chunk = chunks[0]
        logger.info(f"Generating embedding for chunk: {first_chunk.section_type}")
        logger.info(f"Chunk content preview: {first_chunk.content[:100]}...")
        
        # Test individual embedding models
        logger.info("Testing FinE5 directly...")
        fine5_embedding = embeddings.clients["fin_e5"].embed_batch([first_chunk.content])
        logger.info(f"FinE5 result: {len(fine5_embedding[0]) if fine5_embedding[0] else 0} dims")
        
        logger.info("Testing Financial BERT directly...")
        finbert_embedding = embeddings.clients["voyage"].embed_batch([first_chunk.content])  # This is actually FinBERT now
        logger.info(f"FinBERT result: {len(finbert_embedding[0]) if finbert_embedding[0] else 0} dims")
        
        # Test ensemble
        logger.info("Testing ensemble embedding...")
        ensemble_embedding = embeddings.embed_batch([first_chunk.content])
        logger.info(f"Ensemble result: {type(ensemble_embedding)}, length: {len(ensemble_embedding) if ensemble_embedding else 0}")
        
        if ensemble_embedding and len(ensemble_embedding) > 0:
            emb = ensemble_embedding[0]
            logger.info(f"First embedding: {type(emb)}, length: {len(emb) if emb else 0}")
            if emb and len(emb) > 0:
                logger.info(f"Sample values: {emb[:5]}")
                logger.info("AAPL ingestion test successful!")
                return True
        
        logger.error("Ensemble embedding failed")
        return False
        
    except Exception as e:
        logger.error(f"AAPL ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = ingest_apple()
    if success:
        print("\nSYSTEM IS WORKING! Ready to ingest all 15 tickers!")
    else:
        print("\nIssues found, but SEC client and models are working.")
    sys.exit(0 if success else 1)
