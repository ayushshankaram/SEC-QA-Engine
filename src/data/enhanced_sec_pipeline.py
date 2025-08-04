"""
Enhanced SEC Pipeline
End-to-end pipeline for ingesting, processing, and storing SEC filing data.
"""

import os
import sys
import logging
import asyncio
import threading
import time
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Import pipeline components
from storage.neo4j_client import Neo4jClient
from data.secio_client import SECIOClient
from data.document_processor import DocumentProcessor
from core.embedding_ensemble import EmbeddingEnsemble
from data.forms_345_processor import Forms345Processor
from core.neo4j_retrieval_engine import Neo4jRetrievalEngine

logger = logging.getLogger(__name__)


class EnhancedSECPipeline:
    """Comprehensive pipeline for SEC filing data ingestion and processing."""
    
    def __init__(self, secio_client: SECIOClient = None,
                 document_processor: DocumentProcessor = None,
                 embedding_ensemble: EmbeddingEnsemble = None,
                 neo4j_client: Neo4jClient = None,
                 max_workers: int = 3):
        """Initialize the SEC pipeline with all components."""
        
        # Initialize clients
        self.secio_client = secio_client or SECIOClient()
        self.document_processor = document_processor or DocumentProcessor()
        self.embedding_ensemble = embedding_ensemble or EmbeddingEnsemble()
        self.neo4j_client = neo4j_client or Neo4jClient()
        
        # Processing configuration
        self.max_workers = max_workers
        self.batch_size = 5  # Process 5 filings at a time
        self.max_filing_size_mb = int(os.getenv("MAX_FILING_SIZE_MB", 50))
        
        # Threading
        self.processing_lock = threading.Lock()
        
        # Statistics tracking
        self.stats = {
            "companies_processed": 0,
            "filings_processed": 0,
            "sections_created": 0,
            "facts_processed": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
        
        # Token usage tracking
        self.token_usage = {
            "voyage_tokens_used": 0,
            "total_tokens_estimated": 0,
            "token_limit": int(os.getenv("VOYAGE_TOKEN_LIMIT", 1000000))
        }
        
        logger.info("Enhanced SEC pipeline initialized")
    
    def ingest_company_filings(self, ticker: str, filing_types: List[str] = None,
                             max_filings: int = 10, start_date: str = None,
                             end_date: str = None) -> Dict[str, Any]:
        """Ingest filings for a single company."""
        
        logger.info(f"Starting ingestion for company: {ticker}")
        
        filing_types = filing_types or ["10-K", "10-Q", "8-K", "DEF 14A"]
        
        try:
            # Get company information
            company_info = self.secio_client.get_company_info(ticker)
            if not company_info:
                logger.error(f"Could not retrieve company info for {ticker}")
                return {"success": False, "error": f"Company {ticker} not found"}
            
            # Create/update company in Neo4j
            company_result = self.neo4j_client.create_company(
                ticker=ticker,
                name=company_info.get("name", ""),
                sector=company_info.get("sector", ""),
                industry=company_info.get("industry", "")
            )
            
            if not company_result:
                logger.error(f"Failed to create company record for {ticker}")
                return {"success": False, "error": "Failed to create company record"}
            
            # Get filings for each type
            all_filings = []
            # Calculate limit per filing type, ensuring at least 1 per type
            limit_per_type = max(1, max_filings // len(filing_types))
            for filing_type in filing_types:
                logger.info(f"Fetching {filing_type} filings for {ticker}")
                
                filings = self.secio_client.get_company_filings(
                    ticker=ticker,
                    filing_type=filing_type,
                    start_date=start_date,
                    end_date=end_date,
                    limit=limit_per_type
                )
                
                all_filings.extend(filings)
            
            # Sort by date (most recent first) and limit
            all_filings.sort(key=lambda x: x.get("filing_date", ""), reverse=True)
            all_filings = all_filings[:max_filings]
            
            logger.info(f"Processing {len(all_filings)} filings for {ticker}")
            
            # Process filings
            results = []
            cik = company_info.get('cik', '')
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit filing processing tasks
                future_to_filing = {
                    executor.submit(self._process_single_filing, ticker, filing, cik): filing
                    for filing in all_filings
                }
                
                # Collect results
                for future in as_completed(future_to_filing):
                    filing = future_to_filing[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result["success"]:
                            self.stats["filings_processed"] += 1
                        else:
                            self.stats["errors"] += 1
                            
                    except Exception as e:
                        logger.error(f"Filing processing failed: {e}")
                        self.stats["errors"] += 1
                        results.append({
                            "success": False,
                            "accession_number": filing.get("accession_number", "unknown"),
                            "error": str(e)
                        })
            
            self.stats["companies_processed"] += 1
            
            successful_filings = [r for r in results if r["success"]]
            logger.info(f"Company {ticker} ingestion completed: {len(successful_filings)}/{len(all_filings)} successful")
            
            return {
                "success": True,
                "ticker": ticker,
                "filings_processed": len(successful_filings),
                "total_filings": len(all_filings),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Company ingestion failed for {ticker}: {e}")
            self.stats["errors"] += 1
            return {"success": False, "ticker": ticker, "error": str(e)}
    
    def _process_single_filing(self, ticker: str, filing_info: Dict, cik: str) -> Dict[str, Any]:
        """Process a single SEC filing through the complete pipeline."""
        
        accession_number = filing_info.get("accession_number")
        filing_type = filing_info.get("filing_type")
        primary_document = filing_info.get("primaryDocument")
        
        logger.debug(f"Processing {filing_type} filing: {accession_number}")
        
        try:
            # Check token usage before processing
            if self._check_token_limit():
                logger.warning(f"Approaching token limit, skipping filing {accession_number}")
                return {
                    "success": False,
                    "accession_number": accession_number,
                    "error": "Token limit approached"
                }
            
            # Create filing record in Neo4j
            filing_result = self.neo4j_client.create_filing(
                company_ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_info.get("filing_date"),
                accession_number=accession_number,
                url=filing_info.get("url", "")
            )
            
            if not filing_result:
                return {
                    "success": False,
                    "accession_number": accession_number,
                    "error": "Failed to create filing record"
                }
            
            # Get filing content
            if primary_document and cik:
                content = self.secio_client.get_filing_content(cik, accession_number, primary_document)
            else:
                content = ""
                logger.warning(f"Missing CIK or primary document for filing {accession_number}")
                
            if not content:
                return {
                    "success": False,
                    "accession_number": accession_number,
                    "error": "Failed to retrieve filing content"
                }
            
            # Check content size
            content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
            if content_size_mb > self.max_filing_size_mb:
                logger.warning(f"Filing {accession_number} too large: {content_size_mb:.1f}MB")
                # Truncate content
                max_chars = self.max_filing_size_mb * 1024 * 1024
                content = content[:max_chars]
            
            # Process document into sections
            document_chunks = self.document_processor.process_filing(
                content, filing_type
            )
            
            if not document_chunks:
                return {
                    "success": False,
                    "accession_number": accession_number,
                    "error": "Document processing failed"
                }
            
            # Generate embeddings for chunks
            chunk_texts = [chunk.content for chunk in document_chunks]
            embeddings = self.embedding_ensemble.embed_batch(chunk_texts)
            
            # Create section records with embeddings
            sections_created = 0
            for chunk, embedding in zip(document_chunks, embeddings):
                try:
                    section_id = self.neo4j_client.create_section(
                        accession_number=accession_number,
                        section_type=chunk.section_type,
                        content=chunk.content,
                        embeddings=embedding,
                        chunk_id=chunk.chunk_id,
                        chunk_index=chunk.metadata.get("chunk_index", 0),
                        character_count=chunk.metadata.get("character_count", 0)
                    )
                    
                    if section_id:
                        sections_created += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to create section: {e}")
            
            self.stats["sections_created"] += sections_created
            
            # Process XBRL data if available
            facts_processed = 0
            if filing_type in ["10-K", "10-Q"]:
                facts_processed = self._process_xbrl_data(cik, accession_number)
            
            # Handle Forms 3,4,5 for insider trading
            if filing_type in ["3", "4", "5"]:
                self._process_insider_trading_form(ticker, filing_info)
            
            # Update token usage estimation
            self._update_token_usage(chunk_texts, embeddings)
            
            logger.info(f"Filing {accession_number} processed successfully: "
                       f"{sections_created} sections, {facts_processed} facts")
            
            return {
                "success": True,
                "accession_number": accession_number,
                "filing_type": filing_type,
                "sections_created": sections_created,
                "facts_processed": facts_processed,
                "content_size_mb": content_size_mb
            }
            
        except Exception as e:
            logger.error(f"Filing processing failed for {accession_number}: {e}")
            return {
                "success": False,
                "accession_number": accession_number,
                "error": str(e)
            }
    
    def _process_xbrl_data(self, cik: str, accession_number: str) -> int:
        """Process XBRL financial data from a filing."""
        
        try:
            xbrl_data = self.secio_client.get_xbrl_data(cik, accession_number)
            if not xbrl_data:
                return 0
            
            # Process XBRL facts
            facts = self.document_processor.process_xbrl_data(xbrl_data)
            
            facts_processed = 0
            for fact in facts:
                try:
                    fact_id = self.neo4j_client.create_xbrl_fact(
                        accession_number=accession_number,
                        concept=fact.get("concept"),
                        value=fact.get("value"),
                        unit=fact.get("unit"),
                        period=fact.get("period")
                    )
                    
                    if fact_id:
                        facts_processed += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to create XBRL fact: {e}")
            
            self.stats["facts_processed"] += facts_processed
            return facts_processed
            
        except Exception as e:
            logger.warning(f"XBRL processing failed for {accession_number}: {e}")
            return 0
    
    def _process_insider_trading_form(self, ticker: str, filing_info: Dict):
        """Process insider trading forms (3, 4, 5)."""
        
        try:
            # This would need specialized parsing for insider trading forms
            # For now, we'll create a placeholder implementation
            
            # Get insider transactions from SEC.io API
            transactions = self.secio_client.get_insider_transactions(
                ticker=ticker,
                limit=10
            )
            
            for transaction in transactions:
                # Create executive record
                executive_name = transaction.get("executive_name", "Unknown")
                executive_title = transaction.get("executive_title", "")
                
                executive_id = self.neo4j_client.create_executive(
                    name=executive_name,
                    title=executive_title,
                    company_ticker=ticker
                )
                
                if executive_id:
                    # Create trading transaction
                    self.neo4j_client.create_trading_transaction(
                        executive_id=executive_id,
                        company_ticker=ticker,
                        transaction_date=transaction.get("transaction_date"),
                        shares=transaction.get("shares", 0),
                        price=transaction.get("price", 0),
                        transaction_type=transaction.get("transaction_type", "Unknown")
                    )
            
        except Exception as e:
            logger.warning(f"Insider trading processing failed: {e}")
    
    def _check_token_limit(self) -> bool:
        """Check if we're approaching the token limit."""
        usage_stats = self.embedding_ensemble.get_usage_stats()
        
        # Get Voyage token usage
        voyage_stats = usage_stats.get("individual_models", {}).get("voyage", {})
        tokens_used = voyage_stats.get("tokens_used", 0)
        
        # Check if we're using more than 90% of the limit
        usage_percentage = tokens_used / self.token_usage["token_limit"]
        return usage_percentage > 0.9
    
    def _update_token_usage(self, texts: List[str], embeddings: List):
        """Update token usage estimates."""
        # Estimate tokens used (rough approximation)
        total_chars = sum(len(text) for text in texts if text)
        estimated_tokens = total_chars // 4  # Rough estimate
        
        self.token_usage["total_tokens_estimated"] += estimated_tokens
    
    def bulk_ingest_companies(self, companies: Dict[str, List[str]], 
                            filing_counts: Dict[str, int] = None,
                            date_range: Tuple[str, str] = None) -> Dict[str, Any]:
        """
        Bulk ingest multiple companies across sectors.
        
        Args:
            companies: Dict mapping sector names to lists of tickers
            filing_counts: Dict mapping filing types to max counts per company
            date_range: Optional date range for filings (start_date, end_date)
        """
        
        self.stats["start_time"] = datetime.now()
        logger.info("Starting bulk company ingestion")
        
        filing_counts = filing_counts or {
            "10-K": 1,    # 1 recent annual report per company
            "10-Q": 3,    # 3 recent quarterly reports per company
            "8-K": 2,     # 2 recent current reports per company
            "DEF 14A": 1  # 1 recent proxy statement per company
        }
        
        all_results = {}
        total_companies = sum(len(tickers) for tickers in companies.values())
        processed_companies = 0
        
        for sector, tickers in companies.items():
            logger.info(f"Processing {sector} sector: {len(tickers)} companies")
            sector_results = []
            
            for ticker in tickers:
                # Check token usage before each company
                if self._check_token_limit():
                    logger.warning("Token limit approached, stopping bulk ingestion")
                    break
                
                logger.info(f"Processing company {processed_companies + 1}/{total_companies}: {ticker}")
                
                # Determine filing types and counts for this company
                filing_types = list(filing_counts.keys())
                max_filings = sum(filing_counts.values())
                
                # Ingest company filings
                result = self.ingest_company_filings(
                    ticker=ticker,
                    filing_types=filing_types,
                    max_filings=max_filings,
                    start_date=date_range[0] if date_range else None,
                    end_date=date_range[1] if date_range else None
                )
                
                sector_results.append(result)
                processed_companies += 1
                
                # Add small delay to be respectful to APIs
                time.sleep(1)
            
            all_results[sector] = sector_results
        
        self.stats["end_time"] = datetime.now()
        
        # Calculate summary statistics
        successful_companies = sum(
            len([r for r in results if r.get("success", False)])
            for results in all_results.values()
        )
        
        total_filings_processed = sum(
            sum(r.get("filings_processed", 0) for r in results)
            for results in all_results.values()
        )
        
        logger.info(f"Bulk ingestion completed: {successful_companies}/{total_companies} companies, "
                   f"{total_filings_processed} filings processed")
        
        return {
            "success": True,
            "companies_successful": successful_companies,
            "companies_total": total_companies,
            "filings_processed": total_filings_processed,
            "processing_time": (self.stats["end_time"] - self.stats["start_time"]).total_seconds(),
            "results_by_sector": all_results,
            "statistics": self.get_processing_stats(),
            "token_usage": self.get_token_usage()
        }
    
    def fit_sparse_embeddings(self, sample_size: int = 100):
        """Fit the sparse embeddings model using existing content in Neo4j."""
        
        logger.info(f"Fitting sparse embeddings model with sample size: {sample_size}")
        
        # Get sample content from Neo4j
        query = """
        MATCH (s:Section)
        WHERE s.content IS NOT NULL AND length(s.content) > 100
        RETURN s.content as content
        LIMIT $limit
        """
        
        try:
            results = self.neo4j_client.execute_query(query, {"limit": sample_size})
            documents = [record["content"] for record in results]
            
            if not documents:
                logger.warning("No documents found for fitting sparse embeddings")
                return False
            
            # Fit the sparse model
            self.embedding_ensemble.fit_sparse_model(documents)
            
            logger.info(f"Sparse embeddings model fitted on {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to fit sparse embeddings: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        
        stats = self.stats.copy()
        
        if stats["start_time"] and stats["end_time"]:
            duration = stats["end_time"] - stats["start_time"]
            stats["duration_seconds"] = duration.total_seconds()
            stats["duration_formatted"] = str(duration)
            
            # Calculate processing rates
            if duration.total_seconds() > 0:
                stats["filings_per_minute"] = (stats["filings_processed"] * 60) / duration.total_seconds()
                stats["sections_per_minute"] = (stats["sections_created"] * 60) / duration.total_seconds()
        
        return stats
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        
        # Get actual usage from embedding ensemble
        ensemble_stats = self.embedding_ensemble.get_usage_stats()
        
        self.token_usage.update({
            "voyage_actual_usage": ensemble_stats.get("individual_models", {}).get("voyage", {}),
            "usage_percentage": (self.token_usage.get("voyage_tokens_used", 0) / 
                               self.token_usage["token_limit"]) * 100
        })
        
        return self.token_usage
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the entire pipeline."""
        
        try:
            # Check all components
            secio_health = self.secio_client.health_check()
            neo4j_health = self.neo4j_client.health_check()
            embedding_health = self.embedding_ensemble.health_check()
            
            # Overall status
            all_healthy = all(
                health.get("status") == "healthy"
                for health in [secio_health, neo4j_health, embedding_health]
            )
            
            return {
                "status": "healthy" if all_healthy else "degraded",
                "components": {
                    "secio_client": secio_health,
                    "neo4j_client": neo4j_health,
                    "embedding_ensemble": embedding_health
                },
                "processing_stats": self.get_processing_stats(),
                "token_usage": self.get_token_usage(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def close(self):
        """Close pipeline and cleanup resources."""
        
        try:
            self.secio_client.close()
            self.neo4j_client.close()
            self.embedding_ensemble.close()
            logger.info("Enhanced SEC pipeline closed")
        except Exception as e:
            logger.warning(f"Error closing pipeline: {e}")


# Helper functions
def get_default_companies_by_sector() -> Dict[str, List[str]]:
    """Get the default company list for the assignment."""
    return {
        "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "Finance": ["JPM", "BAC", "WFC", "GS"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV"],
        "Energy": ["XOM", "CVX", "COP"]
    }


def get_optimized_filing_counts() -> Dict[str, int]:
    """Get optimized filing counts for token management."""
    return {
        "10-K": 1,     # Annual reports - most comprehensive
        "10-Q": 3,     # Quarterly reports - recent trends
        "8-K": 2,      # Current reports - significant events
        "DEF 14A": 1   # Proxy statements - executive compensation
    }


# Context manager support
class EnhancedSECPipeline(EnhancedSECPipeline):
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
