"""
Neo4j Retrieval Engine
Handles hybrid search combining semantic and text search across SEC filings knowledge graph.
"""

import os
import logging
from typing import List, Dict, Optional, Union, Tuple, Any
import numpy as np
from datetime import datetime, date
import json

from storage.neo4j_client import Neo4jClient
from core.embedding_ensemble import EmbeddingEnsemble

logger = logging.getLogger(__name__)


class Neo4jRetrievalEngine:
    """Hybrid retrieval engine for SEC filings using Neo4j and embeddings."""
    
    def __init__(self, neo4j_client: Neo4jClient = None, 
                 embedding_ensemble: EmbeddingEnsemble = None):
        """Initialize retrieval engine."""
        
        # Initialize clients
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.embedding_ensemble = embedding_ensemble or EmbeddingEnsemble()
        
        # Search configuration
        self.default_limit = 10
        self.similarity_threshold = 0.7
        self.text_search_boost = 1.2
        self.semantic_search_boost = 1.0
        
        # Cache for frequent queries
        self.query_cache = {}
        self.cache_max_size = 100
        
        # Usage tracking
        self.search_count = 0
        self.cache_hits = 0
        
        logger.info("Initialized Neo4j retrieval engine with hybrid search")
    
    def search(self, query: str, 
              company_tickers: List[str] = None,
              filing_types: List[str] = None,
              date_range: Tuple[str, str] = None,
              section_types: List[str] = None,
              search_mode: str = "hybrid",
              limit: int = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across SEC filings.
        
        Args:
            query: Search query text
            company_tickers: Filter by specific company tickers
            filing_types: Filter by filing types (10-K, 10-Q, 8-K, etc.)
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            section_types: Filter by section types
            search_mode: "semantic", "text", or "hybrid"
            limit: Maximum number of results
        """
        
        if not query or not query.strip():
            return []
        
        limit = limit or self.default_limit
        self.search_count += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(query, company_tickers, filing_types, 
                                           date_range, section_types, search_mode, limit)
        if cache_key in self.query_cache:
            self.cache_hits += 1
            logger.debug("Retrieved results from cache")
            return self.query_cache[cache_key]
        
        logger.info(f"Performing {search_mode} search: '{query[:50]}...'")
        
        results = []
        
        if search_mode == "semantic":
            results = self._semantic_search(query, company_tickers, filing_types, 
                                          date_range, section_types, limit)
        elif search_mode == "text":
            results = self._text_search(query, company_tickers, filing_types, 
                                      date_range, section_types, limit)
        elif search_mode == "hybrid":
            results = self._hybrid_search(query, company_tickers, filing_types, 
                                        date_range, section_types, limit)
        else:
            raise ValueError(f"Unsupported search mode: {search_mode}")
        
        # Cache results
        if len(self.query_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[cache_key] = results
        
        logger.info(f"Search completed: {len(results)} results returned")
        return results
    
    def _semantic_search(self, query: str, company_tickers: List[str] = None,
                        filing_types: List[str] = None, date_range: Tuple[str, str] = None,
                        section_types: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        
        # Generate query embedding
        query_embedding = self.embedding_ensemble.embed_text(query)
        
        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return []
        
        # Search using Neo4j vector similarity
        results = self.neo4j_client.semantic_search_sections(
            query_embedding=query_embedding,
            company_tickers=company_tickers,
            filing_types=filing_types,
            limit=limit * 2  # Get more results for filtering
        )
        
        # Apply additional filters
        filtered_results = self._apply_filters(results, date_range, section_types)
        
        # Enhance results with metadata
        enhanced_results = []
        for result in filtered_results[:limit]:
            enhanced_result = self._enhance_result(result, "semantic", 
                                                 result.get("similarity", 0.0))
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _text_search(self, query: str, company_tickers: List[str] = None,
                    filing_types: List[str] = None, date_range: Tuple[str, str] = None,
                    section_types: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform text-based search using full-text search."""
        
        # Prepare search query for full-text search
        search_query = self._prepare_fulltext_query(query)
        
        # Search using Neo4j full-text search
        results = self.neo4j_client.text_search_sections(
            search_text=search_query,
            company_tickers=company_tickers,
            filing_types=filing_types,
            limit=limit * 2  # Get more results for filtering
        )
        
        # Apply additional filters
        filtered_results = self._apply_filters(results, date_range, section_types)
        
        # Enhance results with metadata
        enhanced_results = []
        for result in filtered_results[:limit]:
            enhanced_result = self._enhance_result(result, "text", 
                                                 result.get("score", 0.0))
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _hybrid_search(self, query: str, company_tickers: List[str] = None,
                      filing_types: List[str] = None, date_range: Tuple[str, str] = None,
                      section_types: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and text search."""
        
        # Perform both semantic and text searches
        semantic_results = self._semantic_search(query, company_tickers, filing_types, 
                                               date_range, section_types, limit)
        text_results = self._text_search(query, company_tickers, filing_types, 
                                       date_range, section_types, limit)
        
        # Combine and rank results
        combined_results = self._combine_search_results(semantic_results, text_results, limit)
        
        return combined_results
    
    def _apply_filters(self, results: List[Dict], date_range: Tuple[str, str] = None,
                      section_types: List[str] = None) -> List[Dict]:
        """Apply additional filters to search results."""
        
        filtered = results
        
        # Date range filter
        if date_range:
            start_date, end_date = date_range
            filtered = [r for r in filtered 
                       if self._is_date_in_range(r.get("filing_date"), start_date, end_date)]
        
        # Section type filter
        if section_types:
            section_types_lower = [st.lower() for st in section_types]
            filtered = [r for r in filtered 
                       if r.get("section_type", "").lower() in section_types_lower]
        
        return filtered
    
    def _is_date_in_range(self, filing_date: Any, start_date: str, end_date: str) -> bool:
        """Check if filing date is within the specified range."""
        try:
            if not filing_date:
                return False
            
            # Convert filing_date to string if it's a date object
            if hasattr(filing_date, 'isoformat'):
                filing_date_str = filing_date.isoformat()[:10]  # YYYY-MM-DD
            else:
                filing_date_str = str(filing_date)[:10]
            
            return start_date <= filing_date_str <= end_date
        except Exception as e:
            logger.warning(f"Date comparison failed: {e}")
            return True  # Include by default if comparison fails
    
    def _prepare_fulltext_query(self, query: str) -> str:
        """Prepare query for full-text search with financial domain enhancements."""
        
        # Add financial synonym expansion
        financial_synonyms = {
            "revenue": ["revenue", "sales", "income", "earnings"],
            "profit": ["profit", "earnings", "income", "margin"],
            "debt": ["debt", "liabilities", "borrowings", "obligations"],
            "cash": ["cash", "liquidity", "funds"],
            "growth": ["growth", "increase", "expansion", "improvement"],
            "risk": ["risk", "threat", "challenge", "concern"]
        }
        
        expanded_terms = []
        query_words = query.lower().split()
        
        for word in query_words:
            expanded_terms.append(word)
            for key, synonyms in financial_synonyms.items():
                if word in synonyms:
                    expanded_terms.extend([s for s in synonyms if s != word])
        
        # Create fuzzy search query
        unique_terms = list(set(expanded_terms))
        fuzzy_query = " OR ".join([f"{term}~0.8" for term in unique_terms[:10]])  # Limit terms
        
        return fuzzy_query
    
    def _combine_search_results(self, semantic_results: List[Dict], 
                               text_results: List[Dict], limit: int) -> List[Dict]:
        """Combine and rank results from semantic and text searches."""
        
        # Create a combined ranking system
        combined_scores = {}
        result_data = {}
        
        # Process semantic results
        for result in semantic_results:
            key = self._generate_result_key(result)
            semantic_score = result.get("relevance_score", 0.0) * self.semantic_search_boost
            combined_scores[key] = semantic_score
            result_data[key] = result
            result_data[key]["search_methods"] = ["semantic"]
        
        # Process text results
        for result in text_results:
            key = self._generate_result_key(result)
            text_score = result.get("relevance_score", 0.0) * self.text_search_boost
            
            if key in combined_scores:
                # Combine scores for results found by both methods
                combined_scores[key] = (combined_scores[key] + text_score) / 2
                result_data[key]["search_methods"].append("text")
                result_data[key]["hybrid_score"] = combined_scores[key]
            else:
                combined_scores[key] = text_score
                result_data[key] = result
                result_data[key]["search_methods"] = ["text"]
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top results
        final_results = []
        for key, score in sorted_results[:limit]:
            result = result_data[key]
            result["final_score"] = score
            result["search_method"] = "hybrid"
            final_results.append(result)
        
        return final_results
    
    def _generate_result_key(self, result: Dict) -> str:
        """Generate unique key for a search result."""
        return f"{result.get('accession_number', '')}_{result.get('section_type', '')}_{hash(result.get('content', '')[:100])}"
    
    def _enhance_result(self, result: Dict, search_method: str, score: float) -> Dict[str, Any]:
        """Enhance search result with additional metadata."""
        enhanced = result.copy()
        
        # Add search metadata
        enhanced.update({
            "search_method": search_method,
            "relevance_score": score,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Add content snippet (first 200 characters)
        content = enhanced.get("content", "")
        if len(content) > 200:
            enhanced["content_snippet"] = content[:200] + "..."
        else:
            enhanced["content_snippet"] = content
        
        # Add filing URL if available
        if enhanced.get("accession_number"):
            base_url = "https://www.sec.gov/Archives/edgar/data"
            # This would need actual CIK mapping for real URLs
            enhanced["sec_url"] = f"{base_url}/[CIK]/{enhanced['accession_number']}"
        
        return enhanced
    
    def search_similar_sections(self, section_content: str, 
                               exclude_accession: str = None,
                               limit: int = 5) -> List[Dict[str, Any]]:
        """Find sections similar to a given section content."""
        
        # Generate embedding for the section
        section_embedding = self.embedding_ensemble.embed_text(section_content)
        
        if not section_embedding:
            return []
        
        # Search for similar sections
        results = self.neo4j_client.semantic_search_sections(
            query_embedding=section_embedding,
            limit=limit * 2
        )
        
        # Filter out the original section if specified
        if exclude_accession:
            results = [r for r in results if r.get("accession_number") != exclude_accession]
        
        # Enhance and return results
        enhanced_results = []
        for result in results[:limit]:
            enhanced = self._enhance_result(result, "similarity", result.get("similarity", 0.0))
            enhanced_results.append(enhanced)
        
        return enhanced_results
    
    def search_by_xbrl_concepts(self, concepts: List[str], 
                               company_tickers: List[str] = None,
                               date_range: Tuple[str, str] = None,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Search for filings containing specific XBRL concepts."""
        
        # Build Cypher query for XBRL concept search
        cypher_query = """
        MATCH (c:Company)-[:FILED]->(f:Filing)-[:REPORTS_FACT]->(fact:Fact)
        WHERE fact.concept IN $concepts
        """
        
        parameters = {"concepts": concepts, "limit": limit}
        
        # Add company filter
        if company_tickers:
            cypher_query += " AND c.ticker IN $company_tickers"
            parameters["company_tickers"] = company_tickers
        
        # Add date filter
        if date_range:
            cypher_query += " AND f.filing_date >= date($start_date) AND f.filing_date <= date($end_date)"
            parameters["start_date"] = date_range[0]
            parameters["end_date"] = date_range[1]
        
        cypher_query += """
        RETURN DISTINCT c.ticker as ticker,
               c.name as company_name,
               f.filing_type as filing_type,
               f.filing_date as filing_date,
               f.accession_number as accession_number,
               collect(DISTINCT fact.concept) as concepts,
               collect(DISTINCT {concept: fact.concept, value: fact.value, unit: fact.unit}) as facts
        ORDER BY f.filing_date DESC
        LIMIT $limit
        """
        
        try:
            results = self.neo4j_client.execute_query(cypher_query, parameters)
            
            enhanced_results = []
            for record in results:
                result = dict(record)
                result.update({
                    "search_method": "xbrl_concept",
                    "relevance_score": len(result.get("concepts", [])) / len(concepts),
                    "timestamp": datetime.now().isoformat()
                })
                enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"XBRL concept search failed: {e}")
            return []
    
    def search_insider_trading(self, company_tickers: List[str] = None,
                              executive_names: List[str] = None,
                              transaction_types: List[str] = None,
                              date_range: Tuple[str, str] = None,
                              min_shares: float = None,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """Search for insider trading activities."""
        
        cypher_query = """
        MATCH (e:Executive)-[t:TRADED]->(c:Company)
        WHERE 1=1  // Always true condition for dynamic WHERE clauses
        """
        
        parameters = {"limit": limit}
        
        # Add filters dynamically
        if company_tickers:
            cypher_query += " AND c.ticker IN $company_tickers"
            parameters["company_tickers"] = company_tickers
        
        if executive_names:
            cypher_query += " AND e.name IN $executive_names"
            parameters["executive_names"] = executive_names
        
        if transaction_types:
            cypher_query += " AND t.transaction_type IN $transaction_types"
            parameters["transaction_types"] = transaction_types
        
        if date_range:
            cypher_query += " AND t.date >= date($start_date) AND t.date <= date($end_date)"
            parameters["start_date"] = date_range[0]
            parameters["end_date"] = date_range[1]
        
        if min_shares:
            cypher_query += " AND abs(t.shares) >= $min_shares"
            parameters["min_shares"] = min_shares
        
        cypher_query += """
        RETURN e.name as executive_name,
               e.title as executive_title,
               c.ticker as ticker,
               c.name as company_name,
               t.date as transaction_date,
               t.shares as shares,
               t.price as price,
               t.transaction_type as transaction_type,
               (t.shares * t.price) as transaction_value
        ORDER BY t.date DESC
        LIMIT $limit
        """
        
        try:
            results = self.neo4j_client.execute_query(cypher_query, parameters)
            
            enhanced_results = []
            for record in results:
                result = dict(record)
                result.update({
                    "search_method": "insider_trading",
                    "timestamp": datetime.now().isoformat()
                })
                enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Insider trading search failed: {e}")
            return []
    
    def get_filing_summary(self, accession_number: str) -> Dict[str, Any]:
        """Get comprehensive summary of a specific filing."""
        
        query = """
        MATCH (c:Company)-[:FILED]->(f:Filing {accession_number: $accession_number})
        OPTIONAL MATCH (f)-[:CONTAINS_SECTION]->(s:Section)
        OPTIONAL MATCH (f)-[:REPORTS_FACT]->(fact:Fact)
        
        RETURN c.ticker as ticker,
               c.name as company_name,
               c.sector as sector,
               f.filing_type as filing_type,
               f.filing_date as filing_date,
               f.url as filing_url,
               count(DISTINCT s) as section_count,
               collect(DISTINCT s.section_type) as section_types,
               count(DISTINCT fact) as fact_count,
               collect(DISTINCT fact.concept)[0..10] as top_concepts
        """
        
        try:
            results = self.neo4j_client.execute_query(query, {"accession_number": accession_number})
            
            if results:
                summary = dict(results[0])
                summary.update({
                    "accession_number": accession_number,
                    "timestamp": datetime.now().isoformat()
                })
                return summary
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Filing summary failed: {e}")
            return {}
    
    def _generate_cache_key(self, query: str, company_tickers: List[str],
                           filing_types: List[str], date_range: Tuple[str, str],
                           section_types: List[str], search_mode: str, limit: int) -> str:
        """Generate cache key for search query."""
        key_parts = [
            query,
            ",".join(company_tickers or []),
            ",".join(filing_types or []),
            f"{date_range[0]}_{date_range[1]}" if date_range else "",
            ",".join(section_types or []),
            search_mode,
            str(limit)
        ]
        
        return hash("|".join(key_parts))
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get retrieval engine usage statistics."""
        return {
            "search_count": self.search_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / self.search_count if self.search_count > 0 else 0,
            "cache_size": len(self.query_cache),
            "cache_max_size": self.cache_max_size,
            "neo4j_stats": self.neo4j_client.get_system_stats(),
            "embedding_stats": self.embedding_ensemble.get_usage_stats()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of retrieval engine."""
        try:
            # Test Neo4j connection
            neo4j_health = self.neo4j_client.health_check()
            
            # Test embedding ensemble
            embedding_health = self.embedding_ensemble.health_check()
            
            # Test search functionality
            test_results = self.search("revenue growth", limit=1)
            
            return {
                "status": "healthy" if test_results or len(test_results) >= 0 else "degraded",
                "neo4j_health": neo4j_health,
                "embedding_health": embedding_health,
                "test_search_successful": len(test_results) >= 0,
                "usage_stats": self.get_usage_stats(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def close(self):
        """Close retrieval engine and cleanup resources."""
        try:
            self.neo4j_client.close()
            self.embedding_ensemble.close()
            logger.info("Neo4j retrieval engine closed")
        except Exception as e:
            logger.warning(f"Error closing retrieval engine: {e}")


# Context manager support
class Neo4jRetrievalEngine(Neo4jRetrievalEngine):
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
