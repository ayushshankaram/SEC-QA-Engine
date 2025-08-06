#!/usr/bin/env python3
"""
Test script to verify content retrieval from SEC filings
"""

import sys
sys.path.append('/Users/ayushshankaram/Desktop/QAEngine/src')

from data.secio_client import SECIOClient
from data.document_processor import DocumentProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_content_retrieval():
    """Test retrieving and processing content from SEC filings"""
    
    # Initialize clients
    sec_client = SECIOClient()
    doc_processor = DocumentProcessor()
    
    # Test with Apple
    ticker = "AAPL"
    logger.info(f"Testing content retrieval for {ticker}")
    
    # Get company info
    company_info = sec_client.get_company_info(ticker)
    if not company_info:
        logger.error("Failed to get company info")
        return False
    
    logger.info(f"Company: {company_info.get('name', 'Unknown')}")
    logger.info(f"CIK: {company_info.get('cik')}")
    
    # Get recent 10-K filing
    filings = sec_client.get_company_filings(ticker=ticker, filing_type="10-K", limit=1)
    if not filings:
        logger.error("No 10-K filings found")
        return False
    
    filing = filings[0]
    logger.info(f"Testing filing: {filing.get('accession_number')} from {filing.get('filing_date')}")
    
    # Get filing content
    cik = company_info.get('cik')
    accession = filing.get('accession_number')
    primary_doc = filing.get('primaryDocument')
    
    logger.info(f"Retrieving content: CIK={cik}, Accession={accession}, Document={primary_doc}")
    
    content = sec_client.get_filing_content(cik, accession, primary_doc)
    
    if not content:
        logger.error("Failed to retrieve content")
        return False
    
    logger.info(f"Retrieved content length: {len(content)} characters")
    logger.info(f"First 500 characters:\n{content[:500]}")
    
    # Test document processing
    logger.info("Processing document...")
    chunks = doc_processor.process_filing(content, "10-K")
    
    logger.info(f"Extracted {len(chunks)} sections")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 sections
        if hasattr(chunk, 'section_name'):
            logger.info(f"Section {i+1}: {chunk.section_name} ({len(chunk.content)} chars)")
        else:
            logger.info(f"Section {i+1}: {type(chunk)} ({len(str(chunk))} chars)")
    
    return True

if __name__ == "__main__":
    success = test_content_retrieval()
    sys.exit(0 if success else 1)
