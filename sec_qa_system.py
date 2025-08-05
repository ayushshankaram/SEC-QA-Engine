#!/usr/bin/env python3
"""
SEC Filings QA System - Complete Question Answering Implementation
Handles complex financial research questions with source attribution
"""

import sys
import os
sys.path.append('/Users/ayushshankaram/Desktop/QAEngine/src')

import logging
import json
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/ayushshankaram/Desktop/QAEngine/qa_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class QueryContext:
    """Structured query context for SEC research questions"""
    tickers: List[str]
    filing_types: List[str]
    time_periods: List[str]
    question_type: str  # "single", "comparison", "temporal", "cross-sector"
    keywords: List[str]

@dataclass
class SourceAttribution:
    """Source attribution for research answers"""
    ticker: str
    filing_type: str
    filing_date: str
    section_type: str
    content_excerpt: str
    confidence_score: float

class SECQASystem:
    """Comprehensive SEC Filings Question Answering System"""
    
    def __init__(self):
        self.openai_client = None
        self.neo4j_driver = None
        self.embeddings = None
        self.query_history = []
        
        # Sample evaluation questions from assignment
        self.evaluation_questions = [
            "What are the primary revenue drivers for major technology companies, and how have they evolved?",
            "Compare R&D spending trends across companies. What insights about innovation investment strategies?",
            "Identify significant working capital changes for financial services companies and driving factors.",
            "What are the most commonly cited risk factors across industries? How do same-sector companies prioritize differently?",
            "How do companies describe climate-related risks? Notable industry differences?",
            "Analyze recent executive compensation changes. What trends emerge?",
            "What significant insider trading activity occurred? What might this indicate?",
            "How are companies positioning regarding AI and automation? Strategic approaches?",
            "Identify recent M&A activity. What strategic rationale do companies provide?",
            "How do companies describe competitive advantages? What themes emerge?"
        ]
    
    def initialize_system(self) -> bool:
        """Initialize all QA system components"""
        
        try:
            logger.info("🔧 Initializing SEC QA System...")
            
            # Initialize OpenAI
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                logger.error("❌ OPENAI_API_KEY environment variable not set")
                return False
            
            self.openai_client = openai
            
            # Initialize Neo4j
            from neo4j import GraphDatabase
            self.neo4j_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'secfilings123'))
            
            # Test Neo4j connection
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (c:Company) RETURN count(c) as company_count")
                company_count = result.single()['company_count']
                logger.info(f"✅ Connected to Neo4j: {company_count} companies in database")
            
            # Initialize embeddings
            from core.embedding_ensemble import EmbeddingEnsemble
            self.embeddings = EmbeddingEnsemble()
            
            logger.info("✅ SEC QA System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    def parse_query(self, question: str) -> QueryContext:
        """Parse user question to extract context and intent"""
        
        question_lower = question.lower()
        
        # Extract tickers
        tickers = []
        ticker_patterns = [
            'apple', 'aapl', 'microsoft', 'msft', 'google', 'googl', 'amazon', 'amzn', 
            'nvidia', 'nvda', 'jpmorgan', 'jpm', 'bank of america', 'bac', 'wells fargo', 'wfc',
            'goldman sachs', 'gs', 'johnson & johnson', 'jnj', 'pfizer', 'pfe', 
            'unitedhealth', 'unh', 'exxon', 'xom', 'chevron', 'cvx', 'conocophillips', 'cop'
        ]
        
        ticker_map = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'amazon': 'AMZN',
            'nvidia': 'NVDA', 'jpmorgan': 'JPM', 'bank of america': 'BAC', 
            'wells fargo': 'WFC', 'goldman sachs': 'GS', 'johnson & johnson': 'JNJ',
            'pfizer': 'PFE', 'unitedhealth': 'UNH', 'exxon': 'XOM', 'chevron': 'CVX',
            'conocophillips': 'COP'
        }
        
        for pattern in ticker_patterns:
            if pattern in question_lower:
                if pattern in ticker_map:
                    tickers.append(ticker_map[pattern])
                else:
                    tickers.append(pattern.upper())
        
        # If no specific tickers mentioned, determine from context
        if not tickers:
            if any(word in question_lower for word in ['technology', 'tech', 'software']):
                tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
            elif any(word in question_lower for word in ['financial', 'bank', 'finance']):
                tickers = ['JPM', 'BAC', 'WFC', 'GS']
            elif any(word in question_lower for word in ['healthcare', 'pharmaceutical', 'drug']):
                tickers = ['JNJ', 'PFE', 'UNH']
            elif any(word in question_lower for word in ['energy', 'oil', 'gas']):
                tickers = ['XOM', 'CVX', 'COP']
            else:
                # Default to all companies for broad questions
                tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'BAC', 'WFC', 'GS', 'JNJ', 'PFE', 'UNH', 'XOM', 'CVX', 'COP']
        
        # Extract filing types
        filing_types = []
        if any(word in question_lower for word in ['10-k', 'annual', 'yearly']):
            filing_types.append('10-K')
        if any(word in question_lower for word in ['10-q', 'quarterly']):
            filing_types.append('10-Q')
        if any(word in question_lower for word in ['8-k', 'material event', 'acquisition']):
            filing_types.append('8-K')
        if any(word in question_lower for word in ['proxy', 'compensation', 'governance', 'def 14a']):
            filing_types.append('DEF 14A')
        if any(word in question_lower for word in ['insider', 'trading', 'ownership']):
            filing_types.extend(['3', '4', '5'])
        
        if not filing_types:
            filing_types = ['10-K', '10-Q']  # Default to main financial filings
        
        # Extract time periods
        time_periods = []
        if any(year in question_lower for year in ['2024', '2023', '2022', '2021']):
            for year in ['2024', '2023', '2022', '2021']:
                if year in question_lower:
                    time_periods.append(year)
        
        # Determine question type
        if len(tickers) == 1:
            question_type = "single"
        elif any(word in question_lower for word in ['compare', 'versus', 'vs', 'comparison']):
            question_type = "comparison"
        elif any(word in question_lower for word in ['trend', 'over time', 'evolution', 'change']):
            question_type = "temporal"
        elif any(word in question_lower for word in ['industry', 'sector', 'across companies']):
            question_type = "cross-sector"
        else:
            question_type = "single"
        
        # Extract keywords
        keywords = [word for word in question_lower.split() if len(word) > 3 and word not in [
            'what', 'where', 'when', 'which', 'companies', 'company', 'filings', 'information'
        ]]
        
        return QueryContext(
            tickers=tickers,
            filing_types=filing_types,
            time_periods=time_periods,
            question_type=question_type,
            keywords=keywords
        )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import math
            
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            
            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(a * a for a in vec2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except Exception as e:
            logger.debug(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def retrieve_relevant_sections(self, query_context: QueryContext, question: str, limit: int = 20) -> List[Dict]:
        """Retrieve relevant SEC filing sections using compressed embedding search"""
        
        try:
            # Generate question embedding
            question_embedding = self.embeddings.embed_query(question)
            if not question_embedding:
                logger.warning("⚠️ Failed to generate question embedding")
                return []
            
            # Build Neo4j query with filters (without similarity calculation)
            query_parts = []
            params = {'limit': limit}
            
            # Base query - get sections with compressed embeddings
            query_parts.append("MATCH (c:Company)-[:FILED]->(f:Filing)-[:HAS_SECTION]->(s:Section)")
            query_parts.append("WHERE s.embedding_compressed IS NOT NULL")
            
            # Add filters
            conditions = []
            
            if query_context.tickers:
                params['tickers'] = query_context.tickers
                conditions.append("c.ticker IN $tickers")
            
            if query_context.filing_types:
                params['filing_types'] = query_context.filing_types
                conditions.append("f.filing_type IN $filing_types")
            
            if query_context.time_periods:
                year_conditions = []
                for year in query_context.time_periods:
                    year_conditions.append(f"f.filing_date CONTAINS '{year}'")
                if year_conditions:
                    conditions.append(f"({' OR '.join(year_conditions)})")
            
            if conditions:
                query_parts.append("AND " + " AND ".join(conditions))
            
            # Return sections with compressed embeddings
            query_parts.append("""
            RETURN c.ticker AS ticker,
                   c.name AS company_name,
                   c.sector AS sector,
                   f.filing_type AS filing_type,
                   f.filing_date AS filing_date,
                   s.section_type AS section_type,
                   s.content AS content,
                   s.embedding_compressed AS embedding_compressed
            LIMIT $limit * 3
            """)
            
            cypher_query = "\n".join(query_parts)
            
            # Execute query
            with self.neo4j_driver.session() as session:
                result = session.run(cypher_query, params)
                sections = []
                
                for record in result:
                    try:
                        # Decompress and calculate similarity
                        compressed_embedding = record['embedding_compressed']
                        if compressed_embedding:
                            # Decompress the embedding
                            import gzip
                            import base64
                            import json
                            
                            decompressed_data = gzip.decompress(base64.b64decode(compressed_embedding))
                            section_embedding = json.loads(decompressed_data.decode('utf-8'))
                            
                            # Calculate cosine similarity
                            similarity = self._cosine_similarity(question_embedding, section_embedding)
                            
                            if similarity > 0.3:  # Threshold filter
                                sections.append({
                                    'ticker': record['ticker'],
                                    'company_name': record['company_name'],
                                    'sector': record['sector'],
                                    'filing_type': record['filing_type'],
                                    'filing_date': record['filing_date'],
                                    'section_type': record['section_type'],
                                    'content': record['content'],
                                    'similarity': similarity
                                })
                    except Exception as e:
                        logger.debug(f"Error processing section embedding: {e}")
                        continue
                
                # Sort by similarity and return top results
                sections.sort(key=lambda x: x['similarity'], reverse=True)
                logger.info(f"🔍 Retrieved {len(sections)} relevant sections")
                return sections[:limit]
                
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []
    
    def generate_answer(self, question: str, relevant_sections: List[Dict], query_context: QueryContext) -> Dict[str, Any]:
        """Generate comprehensive answer using GPT-4o with source attribution"""
        
        try:
            # Prepare context for GPT-4o
            context_sections = []
            sources = []
            
            for i, section in enumerate(relevant_sections[:15]):  # Limit to top 15 sections
                context_text = f"""
Section {i+1}:
Company: {section['company_name']} ({section['ticker']}) - {section['sector']}
Filing: {section['filing_type']} dated {section['filing_date']}
Section: {section['section_type']}
Content: {section['content'][:2000]}...
Relevance Score: {section['similarity']:.3f}
---
"""
                context_sections.append(context_text)
                
                sources.append(SourceAttribution(
                    ticker=section['ticker'],
                    filing_type=section['filing_type'],
                    filing_date=section['filing_date'],
                    section_type=section['section_type'],
                    content_excerpt=section['content'][:500],
                    confidence_score=section['similarity']
                ))
            
            context = "\n".join(context_sections)
            
            # Create system prompt
            system_prompt = f"""You are an expert financial analyst specializing in SEC filings analysis. You provide comprehensive, accurate answers to financial research questions based on SEC filing data.

Key Requirements:
1. Provide detailed, well-structured answers
2. Always cite specific sources (Company, Filing Type, Date, Section)
3. Compare and contrast information across companies when relevant
4. Identify trends, patterns, and key insights
5. Be specific about numbers, dates, and facts
6. If information is insufficient, clearly state limitations
7. Use professional financial terminology

Query Context:
- Tickers: {', '.join(query_context.tickers)}
- Filing Types: {', '.join(query_context.filing_types)}
- Question Type: {query_context.question_type}
- Time Periods: {', '.join(query_context.time_periods) if query_context.time_periods else 'Not specified'}

Answer Format:
1. Executive Summary (2-3 sentences)
2. Detailed Analysis with specific examples
3. Key Findings and Trends
4. Source Citations
5. Limitations and Additional Context Needed (if any)
"""
            
            user_prompt = f"""
Based on the following SEC filing information, please answer this financial research question:

QUESTION: {question}

SEC FILING CONTEXT:
{context}

Please provide a comprehensive analysis following the required format with proper source attribution.
"""
            
            # Call GPT-4o
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for factual accuracy
                max_tokens=2000
            )
            
            answer = response.choices[0].message.content
            
            # Prepare response
            response_data = {
                "question": question,
                "answer": answer,
                "query_context": query_context,
                "sources": sources,
                "timestamp": datetime.now().isoformat(),
                "sections_analyzed": len(relevant_sections),
                "companies_covered": len(set(s['ticker'] for s in relevant_sections)),
                "filing_types_used": list(set(s['filing_type'] for s in relevant_sections))
            }
            
            logger.info(f"✅ Generated answer using {len(relevant_sections)} sections from {len(set(s['ticker'] for s in relevant_sections))} companies")
            
            return response_data
            
        except Exception as e:
            logger.error(f"❌ Answer generation error: {e}")
            return {
                "question": question,
                "answer": f"Error generating answer: {str(e)}",
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Main method to answer SEC research questions"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"💬 QUESTION: {question}")
        logger.info(f"{'='*80}")
        
        # Parse query
        query_context = self.parse_query(question)
        logger.info(f"🎯 Query Context: {query_context.tickers} | {query_context.filing_types} | {query_context.question_type}")
        
        # Retrieve relevant sections
        relevant_sections = self.retrieve_relevant_sections(query_context, question)
        
        if not relevant_sections:
            return {
                "question": question,
                "answer": "No relevant information found in the SEC filings database.",
                "query_context": query_context,
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
        
        # Generate answer
        response = self.generate_answer(question, relevant_sections, query_context)
        
        # Store in query history
        self.query_history.append(response)
        
        return response
    
    def run_evaluation_questions(self) -> List[Dict[str, Any]]:
        """Run all evaluation questions from the assignment"""
        
        logger.info("\n🎯 RUNNING ASSIGNMENT EVALUATION QUESTIONS")
        logger.info("=" * 80)
        
        results = []
        
        for i, question in enumerate(self.evaluation_questions, 1):
            logger.info(f"\n📊 Question {i}/{len(self.evaluation_questions)}")
            
            result = self.answer_question(question)
            results.append(result)
            
            # Brief pause between questions
            import time
            time.sleep(2)
        
        return results
    
    def generate_evaluation_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("# SEC FILINGS QA SYSTEM - EVALUATION REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n" + "="*80 + "\n")
        
        # Executive Summary
        report.append("## EXECUTIVE SUMMARY")
        total_questions = len(results)
        successful_answers = len([r for r in results if not r.get('error', False)])
        avg_companies = sum(r.get('companies_covered', 0) for r in results) / total_questions if total_questions > 0 else 0
        avg_sections = sum(r.get('sections_analyzed', 0) for r in results) / total_questions if total_questions > 0 else 0
        
        report.append(f"- **Questions Processed**: {total_questions}")
        report.append(f"- **Successful Answers**: {successful_answers}/{total_questions}")
        report.append(f"- **Average Companies per Answer**: {avg_companies:.1f}")
        report.append(f"- **Average Sections Analyzed**: {avg_sections:.1f}")
        report.append("")
        
        # Detailed Q&A
        report.append("## DETAILED QUESTION & ANSWER ANALYSIS")
        report.append("")
        
        for i, result in enumerate(results, 1):
            report.append(f"### Question {i}")
            report.append(f"**Q**: {result['question']}")
            report.append("")
            report.append("**Answer**:")
            report.append(result['answer'])
            report.append("")
            
            if result.get('sources'):
                report.append("**Sources**:")
                for source in result['sources'][:5]:  # Top 5 sources
                    report.append(f"- {source.ticker} {source.filing_type} ({source.filing_date}) - {source.section_type} (Score: {source.confidence_score:.3f})")
                report.append("")
            
            report.append(f"**Analysis Coverage**: {result.get('companies_covered', 0)} companies, {result.get('sections_analyzed', 0)} sections")
            report.append(f"**Filing Types**: {', '.join(result.get('filing_types_used', []))}")
            report.append("")
            report.append("-" * 80)
            report.append("")
        
        return "\n".join(report)

def main():
    """Main execution for SEC QA System"""
    
    # Initialize system
    qa_system = SECQASystem()
    
    if not qa_system.initialize_system():
        print("❌ Failed to initialize QA system")
        return False
    
    print("🎯 SEC FILINGS QA SYSTEM READY")
    print("=" * 60)
    
    # Run evaluation questions
    print("\n🔍 Running Assignment Evaluation Questions...")
    results = qa_system.run_evaluation_questions()
    
    # Generate report
    print("\n📝 Generating Evaluation Report...")
    report = qa_system.generate_evaluation_report(results)
    
    # Save report
    report_file = "/Users/ayushshankaram/Desktop/QAEngine/SEC_QA_Evaluation_Report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Evaluation report saved: {report_file}")
    
    # Interactive mode
    print("\n💬 Interactive QA Mode (type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            result = qa_system.answer_question(question)
            
            print("\n📊 ANSWER:")
            print("-" * 40)
            print(result['answer'])
            
            if result.get('sources'):
                print(f"\n📚 SOURCES ({len(result['sources'])} total):")
                for source in result['sources'][:3]:  # Show top 3
                    print(f"• {source.ticker} {source.filing_type} ({source.filing_date}) - {source.section_type}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Cleanup
    try:
        qa_system.neo4j_driver.close()
    except:
        pass
    
    print("\n👋 SEC QA System session ended")
    return True

if __name__ == "__main__":
    main()
