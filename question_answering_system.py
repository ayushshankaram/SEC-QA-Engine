"""
Professional SEC Question Answering System
Sophisticated AI-powered system for analyzing SEC filings and answering
complex financial research questions using multi-model embeddings and GPT-4o.
"""

import os
import sys
import logging
import json
import gzip
import base64
import time
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import re

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import core components
from src.core.embedding_ensemble import EmbeddingEnsemble
from src.storage.neo4j_client import Neo4jClient
from src.core.neo4j_retrieval_engine import Neo4jRetrievalEngine

# Import OpenAI
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production/question_answering.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Structure for query results."""
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error: Optional[str] = None


@dataclass
class QueryContext:
    """Context information for query processing."""
    companies: List[str]
    filing_types: List[str]
    query_type: str  # single, comparison, analysis
    sector_focus: Optional[str] = None
    date_range: Optional[Tuple[str, str]] = None


class ProfessionalSECQASystem:
    """Professional SEC Question Answering System with multi-model capabilities."""
    
    def __init__(self):
        """Initialize the QA system with all components."""
        # Check OpenAI availability
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI is required for question answering. Install with: pip install openai")
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.openai_client = OpenAI(api_key=api_key)
        
        # Initialize core components
        self.embedding_ensemble = None
        self.neo4j_client = None
        self.retrieval_engine = None
        
        # Query processing configuration
        self.max_context_sections = 30
        self.similarity_threshold = 0.7
        self.max_tokens_per_section = 1500
        
        # Company and sector mappings
        self.company_sectors = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", 
            "AMZN": "Technology", "NVDA": "Technology",
            "JPM": "Finance", "BAC": "Finance", "WFC": "Finance", "GS": "Finance",
            "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare",
            "XOM": "Energy", "CVX": "Energy", "COP": "Energy"
        }
        
        # Statistics tracking
        self.stats = {
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "average_response_time": 0.0,
            "successful_queries": 0,
            "failed_queries": 0
        }
        
        logger.info("Professional SEC QA System initialized")
    
    def initialize_system(self) -> bool:
        """Initialize all system components."""
        try:
            logger.info("Initializing SEC QA System components...")
            
            # Initialize embedding ensemble
            self.embedding_ensemble = EmbeddingEnsemble()
            logger.info("Embedding ensemble initialized")
            
            # Initialize Neo4j client
            self.neo4j_client = Neo4jClient()
            self.neo4j_client.connect()
            logger.info("Neo4j connection established")
            
            # Initialize retrieval engine
            self.retrieval_engine = Neo4jRetrievalEngine(
                neo4j_client=self.neo4j_client,
                embedding_ensemble=self.embedding_ensemble
            )
            logger.info("Retrieval engine initialized")
            
            # Check database status
            section_count = self.neo4j_client.get_section_count()
            company_count = self.neo4j_client.get_company_count()
            
            logger.info(f"Database ready: {section_count:,} sections, {company_count} companies")
            logger.info("SEC QA System initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize QA system: {str(e)}")
            return False
    
    def answer_question(self, question: str, context: QueryContext = None) -> QueryResult:
        """Answer a question using the SEC filings database."""
        start_time = time.time()
        
        try:
            logger.info(f"QUESTION: {question}")
            
            # Analyze question and determine context
            if context is None:
                context = self._analyze_question_context(question)
            
            # Log query context
            companies_str = ", ".join(context.companies) if context.companies else "All"
            filing_types_str = ", ".join(context.filing_types) if context.filing_types else "All"
            logger.info(f"Query Context: [{companies_str}] | [{filing_types_str}] | {context.query_type}")
            
            # Retrieve relevant sections
            relevant_sections = self._retrieve_relevant_sections(question, context)
            
            if not relevant_sections:
                return QueryResult(
                    answer="I couldn't find relevant information to answer your question.",
                    sources=[],
                    metadata={"error": "No relevant sections found"},
                    processing_time=time.time() - start_time,
                    success=False,
                    error="No relevant sections found"
                )
            
            logger.info(f"Retrieved {len(relevant_sections)} relevant sections")
            
            # Generate answer using GPT-4o
            answer, sources = self._generate_answer_with_gpt4o(question, relevant_sections, context)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(processing_time, True)
            
            # Log completion
            unique_companies = len(set(section.get('ticker', '') for section in relevant_sections))
            logger.info(f"Generated answer using {len(sources)} sections from {unique_companies} companies")
            
            return QueryResult(
                answer=answer,
                sources=sources,
                metadata={
                    "question": question,
                    "context": context.__dict__,
                    "sections_retrieved": len(relevant_sections),
                    "sections_used": len(sources),
                    "companies_analyzed": unique_companies
                },
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(processing_time, False)
            
            logger.error(f"Error answering question: {str(e)}")
            
            return QueryResult(
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                metadata={"error": str(e)},
                processing_time=processing_time,
                success=False,
                error=str(e)
            )
    
    def _analyze_question_context(self, question: str) -> QueryContext:
        """Analyze question to determine context and scope."""
        question_lower = question.lower()
        
        # Identify mentioned companies
        mentioned_companies = []
        company_patterns = {
            "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "alphabet": "GOOGL",
            "amazon": "AMZN", "nvidia": "NVDA", "jpmorgan": "JPM", "chase": "JPM",
            "bank of america": "BAC", "wells fargo": "WFC", "goldman sachs": "GS",
            "johnson & johnson": "JNJ", "pfizer": "PFE", "unitedhealth": "UNH",
            "exxon": "XOM", "chevron": "CVX", "conocophillips": "COP"
        }
        
        for company_name, ticker in company_patterns.items():
            if company_name in question_lower or ticker.lower() in question_lower:
                mentioned_companies.append(ticker)
        
        # Identify sector keywords
        sector_keywords = {
            "technology": ["tech", "technology", "software", "hardware", "ai", "cloud"],
            "finance": ["bank", "banking", "financial", "finance", "credit", "loan"],
            "healthcare": ["healthcare", "pharma", "pharmaceutical", "medical", "drug"],
            "energy": ["energy", "oil", "gas", "petroleum", "renewable"]
        }
        
        sector_focus = None
        for sector, keywords in sector_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                sector_focus = sector
                break
        
        # If no companies mentioned but sector identified, include all sector companies
        if not mentioned_companies and sector_focus:
            sector_companies = [ticker for ticker, sector in self.company_sectors.items() 
                             if sector.lower() == sector_focus.lower()]
            mentioned_companies = sector_companies[:5]  # Limit to avoid too many
        
        # Identify filing types based on keywords
        filing_keywords = {
            "10-K": ["annual", "yearly", "comprehensive", "10-k"],
            "10-Q": ["quarterly", "quarter", "10-q"],
            "8-K": ["material", "event", "announcement", "8-k"],
            "DEF 14A": ["proxy", "shareholder", "governance", "def 14a"],
            "Forms 3/4/5": ["insider", "trading", "ownership", "forms"]
        }
        
        relevant_filing_types = []
        for filing_type, keywords in filing_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                relevant_filing_types.append(filing_type)
        
        # Determine query type
        comparison_keywords = ["compare", "comparison", "versus", "vs", "difference", "similar"]
        analysis_keywords = ["analyze", "analysis", "trend", "pattern", "insight", "examine"]
        
        if any(keyword in question_lower for keyword in comparison_keywords):
            query_type = "comparison"
        elif any(keyword in question_lower for keyword in analysis_keywords):
            query_type = "analysis"
        else:
            query_type = "single"
        
        return QueryContext(
            companies=mentioned_companies,
            filing_types=relevant_filing_types,
            query_type=query_type,
            sector_focus=sector_focus
        )
    
    def _retrieve_relevant_sections(self, question: str, context: QueryContext) -> List[Dict[str, Any]]:
        """Retrieve relevant sections using semantic search."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_ensemble.embed_document(question)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            logger.info("Ensemble batch embedding completed: 1/1 successful")
            
            # Build search filters
            filters = {}
            if context.companies:
                filters['companies'] = context.companies
            if context.filing_types:
                filters['filing_types'] = context.filing_types
            if context.sector_focus:
                filters['sector'] = context.sector_focus
            
            # Retrieve sections using Neo4j
            sections = self.retrieval_engine.semantic_search(
                query_embedding=query_embedding,
                limit=self.max_context_sections,
                similarity_threshold=self.similarity_threshold,
                filters=filters
            )
            
            return sections
            
        except Exception as e:
            logger.error(f"Error retrieving sections: {str(e)}")
            return []
    
    def _generate_answer_with_gpt4o(self, question: str, sections: List[Dict[str, Any]], 
                                   context: QueryContext) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate answer using GPT-4o with retrieved sections."""
        try:
            # Prepare context for GPT-4o
            context_text, sources = self._prepare_context_for_gpt(sections, question)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(context.query_type)
            
            # Create user prompt
            user_prompt = f"""
Based on the following SEC filing information, please answer this question comprehensively:

QUESTION: {question}

SEC FILING CONTEXT:
{context_text}

Please provide a professional, detailed answer with:
1. Direct response to the question
2. Supporting evidence from the filings
3. Specific data and figures where available
4. Proper citations using [Company-FilingType-Date] format
5. Professional financial analysis tone

Answer:"""
            
            # Call GPT-4o
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error generating GPT-4o response: {str(e)}")
            raise
    
    def _prepare_context_for_gpt(self, sections: List[Dict[str, Any]], 
                                question: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Prepare context text and sources for GPT-4o."""
        context_parts = []
        sources = []
        total_tokens = 0
        max_total_tokens = 12000  # Leave room for question and answer
        
        # Sort sections by relevance score if available
        sections_sorted = sorted(sections, 
                               key=lambda x: x.get('similarity_score', 0), 
                               reverse=True)
        
        for section in sections_sorted:
            # Estimate token count (rough: 4 chars per token)
            section_content = section.get('content', '')
            estimated_tokens = len(section_content) // 4
            
            if total_tokens + estimated_tokens > max_total_tokens:
                break
            
            # Format section for context
            ticker = section.get('ticker', 'Unknown')
            filing_type = section.get('filing_type', 'Unknown')
            filing_date = section.get('filing_date', 'Unknown')
            section_title = section.get('section_title', 'Unknown Section')
            
            section_header = f"\n--- {ticker} {filing_type} ({filing_date}) - {section_title} ---"
            section_text = f"{section_header}\n{section_content}\n"
            
            context_parts.append(section_text)
            total_tokens += estimated_tokens
            
            # Add to sources
            sources.append({
                "company": ticker,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "section_title": section_title,
                "similarity_score": section.get('similarity_score', 0),
                "url": section.get('url', ''),
                "accession_number": section.get('accession_number', '')
            })
        
        context_text = "\n".join(context_parts)
        return context_text, sources
    
    def _create_system_prompt(self, query_type: str) -> str:
        """Create system prompt based on query type."""
        base_prompt = """You are a professional financial analyst with expertise in SEC filings analysis. 
You provide accurate, detailed, and well-sourced responses to financial research questions.

Key guidelines:
- Use only information provided in the SEC filing context
- Cite sources using [Company-FilingType-Date] format
- Provide specific data, figures, and quotes when available
- Maintain a professional, analytical tone
- Focus on factual information from the filings"""
        
        type_specific = {
            "comparison": "\n- Compare and contrast information across companies\n- Highlight similarities and differences\n- Use comparative analysis techniques",
            "analysis": "\n- Provide deep analytical insights\n- Identify trends and patterns\n- Offer professional interpretation of the data",
            "single": "\n- Provide comprehensive answer focused on the specific question\n- Include relevant details and context\n- Support with specific evidence"
        }
        
        return base_prompt + type_specific.get(query_type, type_specific["single"])
    
    def _update_stats(self, processing_time: float, success: bool):
        """Update system statistics."""
        self.stats["queries_processed"] += 1
        self.stats["total_processing_time"] += processing_time
        
        if success:
            self.stats["successful_queries"] += 1
        else:
            self.stats["failed_queries"] += 1
        
        self.stats["average_response_time"] = (
            self.stats["total_processing_time"] / self.stats["queries_processed"]
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        try:
            health_status = {
                "system_initialized": all([
                    self.embedding_ensemble is not None,
                    self.neo4j_client is not None,
                    self.retrieval_engine is not None
                ]),
                "neo4j_connected": False,
                "openai_available": OPENAI_AVAILABLE,
                "database_sections": 0,
                "database_companies": 0,
                "query_stats": self.stats,
                "timestamp": datetime.now().isoformat()
            }
            
            # Check Neo4j connection
            if self.neo4j_client:
                try:
                    health_status["database_sections"] = self.neo4j_client.get_section_count()
                    health_status["database_companies"] = self.neo4j_client.get_company_count()
                    health_status["neo4j_connected"] = True
                except Exception as e:
                    health_status["neo4j_error"] = str(e)
            
            return health_status
            
        except Exception as e:
            return {
                "error": str(e),
                "system_initialized": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def interactive_session(self):
        """Run interactive question-answering session."""
        print("\n" + "="*80)
        print("SEC FILINGS QUESTION ANSWERING SYSTEM")
        print("Professional Financial Research Interface")
        print("="*80)
        print("\nAvailable commands:")
        print("- Ask any financial question about SEC filings")
        print("- Type 'health' for system status")
        print("- Type 'stats' for query statistics")
        print("- Type 'quit' to exit")
        print("\nExample questions:")
        print("- What are Apple's main revenue sources?")
        print("- Compare R&D spending across technology companies")
        print("- What are the risk factors for JPMorgan Chase?")
        print("-" * 80)
        
        while True:
            try:
                question = input("\n📊 Enter your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using the SEC QA System!")
                    break
                
                elif question.lower() == 'health':
                    health = self.health_check()
                    print(f"\n🏥 System Health Check:")
                    for key, value in health.items():
                        print(f"   {key}: {value}")
                    continue
                
                elif question.lower() == 'stats':
                    print(f"\n📈 Query Statistics:")
                    for key, value in self.stats.items():
                        if key == "average_response_time":
                            print(f"   {key}: {value:.2f}s")
                        else:
                            print(f"   {key}: {value}")
                    continue
                
                elif not question:
                    print("Please enter a question.")
                    continue
                
                print(f"\n🔍 Processing question: {question}")
                print("⏳ Retrieving relevant SEC filing information...")
                
                # Answer the question
                result = self.answer_question(question)
                
                if result.success:
                    print(f"\n✅ Answer (processed in {result.processing_time:.1f}s):")
                    print("-" * 60)
                    print(result.answer)
                    
                    if result.sources:
                        print(f"\n📚 Sources ({len(result.sources)} sections):")
                        for i, source in enumerate(result.sources[:5], 1):  # Show top 5
                            print(f"   {i}. {source['company']} {source['filing_type']} ({source['filing_date']}) - {source['section_title']}")
                        
                        if len(result.sources) > 5:
                            print(f"   ... and {len(result.sources) - 5} more sources")
                    
                    print(f"\n📊 Analysis: {result.metadata['sections_retrieved']} sections analyzed, "
                          f"{result.metadata['companies_analyzed']} companies")
                else:
                    print(f"\n❌ Error: {result.error}")
                
            except KeyboardInterrupt:
                print("\n\nExiting SEC QA System...")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")


def main():
    """Main function for question answering system."""
    try:
        # Initialize QA system
        qa_system = ProfessionalSECQASystem()
        
        # Initialize system components
        if not qa_system.initialize_system():
            logger.error("Failed to initialize QA system")
            return False
        
        # Check if running interactively
        if len(sys.argv) > 1:
            # Command line question
            question = " ".join(sys.argv[1:])
            result = qa_system.answer_question(question)
            
            if result.success:
                print(f"\nQuestion: {question}")
                print(f"\nAnswer:\n{result.answer}")
                print(f"\nSources: {len(result.sources)} sections analyzed")
                print(f"Processing time: {result.processing_time:.1f}s")
            else:
                print(f"Error: {result.error}")
        else:
            # Interactive mode
            qa_system.interactive_session()
        
        return True
        
    except Exception as e:
        logger.error(f"Critical error in QA system: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)