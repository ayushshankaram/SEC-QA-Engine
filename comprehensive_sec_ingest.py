#!/usr/bin/env python3
"""
Comprehensive SEC Filings QA Agent - Complete Assignment Implementation
Handles all SEC filing types with cross-company relationship analysis
"""

import sys
import os
sys.path.append('/Users/ayushshankaram/Desktop/QAEngine/src')

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/ayushshankaram/Desktop/QAEngine/comprehensive_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveSECIngestion:
    """Complete SEC filings ingestion system for QA agent assignment"""
    
    def __init__(self):
        # Assignment companies across sectors
        self.companies = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "Finance": ["JPM", "BAC", "WFC", "GS"],
            "Healthcare": ["JNJ", "PFE", "UNH"],
            "Energy": ["XOM", "CVX", "COP"]
        }
        
        # All required SEC filing types with enhanced descriptions
        self.filing_types = {
            "10-K": "Annual comprehensive financial and business information",
            "10-Q": "Quarterly financial information and updates", 
            "8-K": "Material events and corporate changes",
            "DEF 14A": "Governance, compensation, and proxy statements",
            "3": "Initial insider trading disclosure forms",
            "4": "Statement of ownership changes by insiders", 
            "5": "Annual insider trading summary reports"
        }
        
        # Filing type priorities for cross-analysis
        self.priority_filings = ["10-K", "10-Q", "8-K", "DEF 14A"]
        self.insider_filings = ["3", "4", "5"]
        
        # Cross-company analysis requirements
        self.cross_analysis = {
            "competitive_metrics": ["revenue", "profit", "market_share", "growth"],
            "risk_factors": ["regulatory", "market", "operational", "financial"],
            "governance_analysis": ["board_composition", "executive_comp", "shareholder_rights"]
        }
        
        self.results = {}
        self.total_processed = 0
        self.start_time = datetime.now()
        
    def initialize_components(self):
        """Initialize all system components"""
        try:
            from data.secio_client import SECIOClient
            from core.embedding_ensemble import EmbeddingEnsemble
            from data.document_processor import DocumentProcessor
            from neo4j import GraphDatabase
            
            logger.info("🔧 Initializing system components...")
            
            self.sec_client = SECIOClient()
            self.embeddings = EmbeddingEnsemble()
            self.doc_processor = DocumentProcessor()
            
            # Connect to Neo4j
            self.neo4j_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'secfilings123'))
            
            # Create database schema
            self._create_comprehensive_schema()
            
            logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    def _create_comprehensive_schema(self):
        """Create comprehensive schema for all SEC filing types"""
        with self.neo4j_driver.session() as session:
            try:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT FOR (c:Company) REQUIRE c.ticker IS UNIQUE",
                    "CREATE CONSTRAINT FOR (f:Filing) REQUIRE f.filing_id IS UNIQUE",
                    "CREATE CONSTRAINT FOR (s:Section) REQUIRE s.section_id IS UNIQUE"
                ]
                
                for constraint in constraints:
                    try:
                        session.run(constraint)
                        logger.info("✅ Schema: CONSTRAINT created")
                    except Exception as e:
                        if "already exists" in str(e) or "equivalent constraint" in str(e):
                            logger.warning(f"⚠️ Schema creation: {e}")
                        else:
                            logger.error(f"❌ Schema error: {e}")
                
                # Create indexes (avoiding embedding index due to size)
                indexes = [
                    "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
                    "CREATE INDEX filing_type IF NOT EXISTS FOR (f:Filing) ON (f.filing_type)",
                    "CREATE INDEX filing_date IF NOT EXISTS FOR (f:Filing) ON (f.filing_date)",
                    "CREATE INDEX section_type IF NOT EXISTS FOR (s:Section) ON (s.section_type)",
                    "CREATE INDEX section_ticker IF NOT EXISTS FOR (s:Section) ON (s.ticker)"
                ]
                
                for index in indexes:
                    try:
                        session.run(index)
                        logger.info("✅ Schema: INDEX created")
                    except Exception as e:
                        if "already exists" in str(e) or "equivalent index" in str(e):
                            logger.warning(f"⚠️ Schema creation: {e}")
                        else:
                            logger.error(f"❌ Schema error: {e}")
                            
            except Exception as e:
                logger.error(f"❌ Schema creation failed: {e}")
                raise
    
    def get_all_tickers(self) -> List[str]:
        """Get flattened list of all tickers"""
        tickers = []
        for sector_tickers in self.companies.values():
            tickers.extend(sector_tickers)
        return tickers
    
    def process_company_comprehensive(self, ticker: str, sector: str) -> Dict[str, Any]:
        """Process all filing types for a single company"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🏢 PROCESSING {ticker} ({sector})")
        logger.info(f"{'='*60}")
        
        company_result = {
            "ticker": ticker,
            "sector": sector, 
            "filings": {},
            "total_sections": 0,
            "success": False,
            "errors": []
        }
        
        try:
            # Get company information
            company_info = self.sec_client.get_company_info(ticker)
            if not company_info:
                company_result["errors"].append("No company info found")
                return company_result
            
            logger.info(f"📊 Company: {company_info.get('name', 'Unknown')}")
            
            # Create company node in Neo4j
            self._create_company_node(ticker, company_info, sector)
            
            # Process each filing type
            total_sections = 0
            for filing_type, description in self.filing_types.items():
                logger.info(f"\n📄 Processing {filing_type}: {description}")
                
                filing_result = self._process_filing_type(ticker, filing_type, company_info)
                company_result["filings"][filing_type] = filing_result
                total_sections += filing_result.get("sections_count", 0)
                
                # Small delay between filing types
                time.sleep(0.5)
            
            company_result["total_sections"] = total_sections
            company_result["success"] = total_sections > 0
            
            # Create cross-company relationships
            self._create_sector_relationships(ticker, sector)
            
            # Create cross-filing relationships for comprehensive analysis
            self._create_cross_filing_relationships(ticker, company_result["filings"])
            
            logger.info(f"✅ {ticker} complete: {total_sections} total sections across all filings")
            
        except Exception as e:
            logger.error(f"❌ {ticker} processing failed: {e}")
            company_result["errors"].append(str(e))
        
        return company_result
    
    def _create_company_node(self, ticker: str, company_info: Dict, sector: str):
        """Create company node with comprehensive metadata"""
        
        with self.neo4j_driver.session() as session:
            query = """
            MERGE (c:Company {ticker: $ticker})
            SET c.name = $name,
                c.cik = $cik,
                c.sector = $sector,
                c.industry = $industry,
                c.sic = $sic,
                c.created_at = datetime(),
                c.last_updated = datetime()
            RETURN c
            """
            
            session.run(query, {
                'ticker': ticker,
                'name': company_info.get('name', ticker),
                'cik': company_info.get('cik'),
                'sector': sector,
                'industry': company_info.get('industry', 'Unknown'),
                'sic': company_info.get('sic', 'Unknown')
            })
    
    def _process_filing_type(self, ticker: str, filing_type: str, company_info: Dict) -> Dict[str, Any]:
        """Process specific filing type for a company"""
        
        result = {
            "filing_type": filing_type,
            "filings_found": 0,
            "sections_count": 0,
            "success": False
        }
        
        try:
            # Get filings of this type with dynamic limits based on importance
            if filing_type in self.priority_filings:
                limit = 5  # More filings for critical types
                process_count = 3  # Process more filings
            elif filing_type in self.insider_filings:
                limit = 10  # Insider filings are smaller, get more
                process_count = 5
            else:
                limit = 3
                process_count = 2
            
            filings = self.sec_client.get_company_filings(
                ticker=ticker,
                filing_type=filing_type,
                limit=limit
            )
            
            if not filings:
                logger.info(f"  ⚠️ No {filing_type} filings found")
                return result
            
            result["filings_found"] = len(filings)
            logger.info(f"  📁 Found {len(filings)} {filing_type} filings")
            
            # Process each filing with dynamic processing based on type
            total_sections = 0
            for i, filing in enumerate(filings[:process_count]):
                logger.info(f"    📄 Filing {i+1}: {filing.get('filing_date', 'Unknown date')} ({filing_type})")
                
                sections = self._process_single_filing(ticker, filing, filing_type, company_info)
                total_sections += sections
                
                # Dynamic delays based on filing complexity
                if filing_type in ["10-K", "DEF 14A"]:
                    time.sleep(1)  # Longer delay for complex filings
                else:
                    time.sleep(0.3)  # Shorter delay for simpler filings
            
            result["sections_count"] = total_sections
            result["success"] = total_sections > 0
            
        except Exception as e:
            logger.error(f"  ❌ {filing_type} processing error: {e}")
            result["error"] = str(e)
        
        return result
    
    def _process_single_filing(self, ticker: str, filing: Dict, filing_type: str, company_info: Dict) -> int:
        """Process a single SEC filing"""
        
        try:
            cik = company_info.get('cik')
            accession = filing.get('accession_number')
            primary_doc = filing.get('primaryDocument')
            
            if not all([cik, accession, primary_doc]):
                logger.warning(f"    ⚠️ Incomplete filing info")
                return 0
            
            # Get filing content
            content = self.sec_client.get_filing_content(cik, accession, primary_doc)
            if not content:
                logger.warning(f"    ⚠️ Failed to retrieve content")
                return 0
            
            # Limit content size based on filing type
            content_size_mb = len(content.encode('utf-8')) / (1024 * 1024)
            if filing_type in ["10-K", "DEF 14A"]:
                size_limit = 8  # Larger limit for comprehensive filings
            elif filing_type in ["10-Q", "8-K"]:
                size_limit = 5  # Medium limit
            else:
                size_limit = 2  # Smaller limit for insider filings
                
            if content_size_mb > size_limit:
                content = content[:size_limit * 1024 * 1024]
                logger.info(f"    📏 Truncated from {content_size_mb:.1f}MB to {size_limit}MB")
            
            # Process document into sections
            chunks = self.doc_processor.process_filing(content, filing_type)
            
            if not chunks:
                # Create fallback section for important filings
                chunks = [{
                    'content': content[:3000],
                    'section_type': f'{filing_type} Overview',
                    'chunk_id': f"{ticker}_{filing_type}_{accession}_overview"
                }]
            
            logger.info(f"    📋 Extracted {len(chunks)} sections")
            
            # Generate embeddings with appropriate section limits by filing type
            chunk_texts = []
            valid_chunks = []
            
            # Dynamic section limits based on filing type complexity
            if filing_type in ["10-K"]:
                section_limit = 8  # More sections for annual reports
            elif filing_type in ["10-Q", "DEF 14A"]:
                section_limit = 6  # Medium sections for quarterly/proxy
            elif filing_type in ["8-K"]:
                section_limit = 4  # Fewer sections for events
            else:
                section_limit = 3  # Minimal sections for insider filings
            
            for chunk in chunks[:section_limit]:
                if hasattr(chunk, 'content'):
                    content_text = chunk.content
                elif isinstance(chunk, dict):
                    content_text = chunk.get('content', '')
                else:
                    content_text = str(chunk)
                
                if content_text and content_text.strip():
                    chunk_texts.append(content_text)
                    valid_chunks.append(chunk)
            
            if not chunk_texts:
                return 0
            
            # Generate embeddings
            embeddings_list = self.embeddings.embed_batch(chunk_texts)
            if not embeddings_list:
                return 0
            
            # Store in Neo4j
            return self._store_filing_data(ticker, filing, filing_type, valid_chunks, chunk_texts, embeddings_list)
            
        except Exception as e:
            logger.error(f"    ❌ Filing processing error: {e}")
            return 0
    
    def _store_filing_data(self, ticker: str, filing: Dict, filing_type: str, 
                          chunks: List, chunk_texts: List[str], embeddings_list: List) -> int:
        """Store filing data in Neo4j with comprehensive relationships"""
        
        try:
            with self.neo4j_driver.session() as session:
                # Create filing node with proper SEC URL
                filing_query = """
                MATCH (c:Company {ticker: $ticker})
                MERGE (f:Filing {filing_id: $filing_id})
                SET f.accession_number = $accession_number,
                    f.filing_type = $filing_type,
                    f.filing_date = $filing_date,
                    f.period_end_date = $period_end_date,
                    f.sec_url = $sec_url,
                    f.created_at = datetime()
                MERGE (c)-[:FILED]->(f)
                RETURN f
                """
                
                filing_id = f"{ticker}_{filing.get('accession_number', 'unknown')}"
                
                # Create proper SEC filing URL (not the XML data URL)
                accession_clean = filing.get('accession_number', '').replace('-', '')
                if len(accession_clean) >= 10:
                    sec_url = f"https://www.sec.gov/Archives/edgar/data/{filing.get('cik', 'unknown')}/{accession_clean}/{filing.get('accession_number', 'unknown')}-index.html"
                else:
                    sec_url = f"https://www.sec.gov/edgar/search/#/filings"
                
                session.run(filing_query, {
                    'ticker': ticker,
                    'filing_id': filing_id,
                    'accession_number': filing.get('accession_number'),
                    'filing_type': filing_type,
                    'filing_date': filing.get('filing_date'),
                    'period_end_date': filing.get('period_end_date'),
                    'sec_url': sec_url
                })
                
                # Store sections with embeddings
                stored_sections = 0
                for i, (chunk, chunk_text, embedding) in enumerate(zip(chunks, chunk_texts, embeddings_list)):
                    if embedding is None:
                        continue
                    
                    section_type = 'General'
                    if hasattr(chunk, 'section_type'):
                        section_type = chunk.section_type
                    elif isinstance(chunk, dict):
                        section_type = chunk.get('section_type', 'General')
                    
                    # Store embedding as compressed string to avoid Neo4j size limits
                    import json
                    import gzip
                    import base64
                    
                    # Compress embedding vector
                    embedding_json = json.dumps(embedding).encode('utf-8')
                    compressed_embedding = gzip.compress(embedding_json)
                    embedding_b64 = base64.b64encode(compressed_embedding).decode('utf-8')
                    
                    section_query = """
                    MATCH (f:Filing {filing_id: $filing_id})
                    CREATE (s:Section {
                        section_id: $section_id,
                        section_type: $section_type,
                        content: $content,
                        embedding_compressed: $embedding_compressed,
                        embedding_size: $embedding_size,
                        chunk_index: $chunk_index,
                        filing_type: $filing_type,
                        ticker: $ticker,
                        created_at: datetime()
                    })
                    CREATE (f)-[:HAS_SECTION]->(s)
                    RETURN s
                    """
                    
                    section_id = f"{filing_id}_section_{i}"
                    session.run(section_query, {
                        'filing_id': filing_id,
                        'section_id': section_id,
                        'section_type': section_type,
                        'content': chunk_text[:4000],  # Limit content
                        'embedding_compressed': embedding_b64,
                        'embedding_size': len(embedding) if embedding else 0,
                        'chunk_index': i,
                        'filing_type': filing_type,
                        'ticker': ticker
                    })
                    stored_sections += 1
                
                logger.info(f"      💾 Stored {stored_sections} sections")
                return stored_sections
                
        except Exception as e:
            logger.error(f"      ❌ Storage error: {e}")
            return 0
    
    def _create_sector_relationships(self, ticker: str, sector: str):
        """Create relationships between companies in the same sector"""
        
        try:
            with self.neo4j_driver.session() as session:
                # Find other companies in same sector
                query = """
                MATCH (c1:Company {ticker: $ticker})
                MATCH (c2:Company {sector: $sector})
                WHERE c1.ticker <> c2.ticker
                MERGE (c1)-[:SAME_SECTOR]->(c2)
                MERGE (c2)-[:SAME_SECTOR]->(c1)
                """
                
                session.run(query, {'ticker': ticker, 'sector': sector})
                
                # Create competitive relationships for same sector
                competitive_query = """
                MATCH (c1:Company {ticker: $ticker, sector: $sector})
                MATCH (c2:Company {sector: $sector})
                WHERE c1.ticker <> c2.ticker
                MERGE (c1)-[:COMPETITOR]->(c2)
                """
                
                session.run(competitive_query, {'ticker': ticker, 'sector': sector})
                
        except Exception as e:
            logger.warning(f"⚠️ Relationship creation error: {e}")
    
    def _create_cross_filing_relationships(self, ticker: str, filings_data: Dict):
        """Create relationships between different filing types for comprehensive analysis"""
        
        try:
            with self.neo4j_driver.session() as session:
                
                # Link 10-K to 10-Q filings (annual to quarterly relationship)
                if filings_data.get("10-K", {}).get("success") and filings_data.get("10-Q", {}).get("success"):
                    annual_quarterly_query = """
                    MATCH (f1:Filing {filing_type: '10-K'})-[:FILED]-(c:Company {ticker: $ticker})
                    MATCH (f2:Filing {filing_type: '10-Q'})-[:FILED]-(c)
                    MERGE (f1)-[:ANNUAL_TO_QUARTERLY]->(f2)
                    """
                    session.run(annual_quarterly_query, {'ticker': ticker})
                
                # Link 8-K to other filings (material events context)
                if filings_data.get("8-K", {}).get("success"):
                    events_query = """
                    MATCH (f1:Filing {filing_type: '8-K'})-[:FILED]-(c:Company {ticker: $ticker})
                    MATCH (f2:Filing)-[:FILED]-(c)
                    WHERE f2.filing_type IN ['10-K', '10-Q', 'DEF 14A'] AND f1.filing_id <> f2.filing_id
                    MERGE (f1)-[:MATERIAL_EVENT_CONTEXT]->(f2)
                    """
                    session.run(events_query, {'ticker': ticker})
                
                # Link DEF 14A to other filings (governance context)
                if filings_data.get("DEF 14A", {}).get("success"):
                    governance_query = """
                    MATCH (f1:Filing {filing_type: 'DEF 14A'})-[:FILED]-(c:Company {ticker: $ticker})
                    MATCH (f2:Filing)-[:FILED]-(c)
                    WHERE f2.filing_type IN ['10-K', '10-Q'] AND f1.filing_id <> f2.filing_id
                    MERGE (f1)-[:GOVERNANCE_CONTEXT]->(f2)
                    """
                    session.run(governance_query, {'ticker': ticker})
                
                # Link insider trading forms to other filings
                insider_types = ["3", "4", "5"]
                for insider_type in insider_types:
                    if filings_data.get(insider_type, {}).get("success"):
                        insider_query = """
                        MATCH (f1:Filing {filing_type: $insider_type})-[:FILED]-(c:Company {ticker: $ticker})
                        MATCH (f2:Filing)-[:FILED]-(c)
                        WHERE f2.filing_type IN ['10-K', '10-Q', 'DEF 14A'] AND f1.filing_id <> f2.filing_id
                        MERGE (f1)-[:INSIDER_TRADING_CONTEXT]->(f2)
                        """
                        session.run(insider_query, {'ticker': ticker, 'insider_type': insider_type})
                
                logger.info(f"      🔗 Created cross-filing relationships for {ticker}")
                
        except Exception as e:
            logger.warning(f"⚠️ Cross-filing relationship creation error: {e}")
    
    def run_comprehensive_ingestion(self):
        """Run complete SEC data ingestion for all companies and filing types"""
        
        all_tickers = self.get_all_tickers()
        
        logger.info("🚀 COMPREHENSIVE SEC FILINGS QA AGENT INGESTION")
        logger.info("=" * 80)
        logger.info(f"📊 Companies: {len(all_tickers)} across {len(self.companies)} sectors")
        logger.info(f"📄 Filing types: {list(self.filing_types.keys())}")
        logger.info(f"⏰ Started: {self.start_time}")
        logger.info("=" * 80)
        
        if not self.initialize_components():
            return False
        
        # Process each company
        for sector, tickers in self.companies.items():
            logger.info(f"\n🏭 SECTOR: {sector}")
            logger.info("-" * 40)
            
            for ticker in tickers:
                company_result = self.process_company_comprehensive(ticker, sector)
                self.results[ticker] = company_result
                
                if company_result["success"]:
                    self.total_processed += 1
                
                # Delay between companies
                time.sleep(2)
        
        # Generate final report with cross-filing analytics
        self._generate_final_report()
        
        # Create comprehensive cross-company analytics
        self._generate_cross_company_analytics()
        
        # Close connections
        try:
            self.neo4j_driver.close()
        except:
            pass
        
        return self.total_processed >= len(all_tickers) * 0.7  # 70% success rate
    
    def _generate_final_report(self):
        """Generate comprehensive ingestion report"""
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        total_companies = len(self.get_all_tickers())
        successful_companies = len([r for r in self.results.values() if r.get("success", False)])
        
        total_sections = sum(r.get("total_sections", 0) for r in self.results.values())
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPREHENSIVE INGESTION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"🏢 Companies processed: {successful_companies}/{total_companies}")
        logger.info(f"📄 Total sections with embeddings: {total_sections}")
        logger.info(f"⏱️  Processing time: {duration}")
        logger.info(f"⚡ Average per company: {duration.total_seconds()/total_companies:.1f}s")
        
        # Detailed breakdown by sector
        logger.info("\n📊 SECTOR BREAKDOWN:")
        for sector, tickers in self.companies.items():
            sector_success = len([t for t in tickers if self.results.get(t, {}).get("success", False)])
            sector_sections = sum(self.results.get(t, {}).get("total_sections", 0) for t in tickers)
            logger.info(f"  {sector}: {sector_success}/{len(tickers)} companies, {sector_sections} sections")
        
        # Filing type breakdown
        logger.info("\n📄 FILING TYPE ANALYSIS:")
        for filing_type in self.filing_types.keys():
            type_sections = sum(
                r.get("filings", {}).get(filing_type, {}).get("sections_count", 0) 
                for r in self.results.values()
            )
            logger.info(f"  {filing_type}: {type_sections} sections across all companies")
        
        # Success details
        logger.info("\n✅ SUCCESSFUL COMPANIES:")
        for ticker, result in self.results.items():
            if result.get("success"):
                logger.info(f"  {ticker} ({result['sector']}): {result['total_sections']} sections")
        
        # Errors
        logger.info("\n❌ ISSUES ENCOUNTERED:")
        for ticker, result in self.results.items():
            if not result.get("success") or result.get("errors"):
                errors = result.get("errors", ["Unknown error"])
                logger.info(f"  {ticker}: {', '.join(errors)}")
        
        if successful_companies >= total_companies * 0.7:
            logger.info(f"\n🎉 ASSIGNMENT READY!")
            logger.info("✅ Comprehensive SEC data ingested successfully")
            logger.info("🔍 System ready for financial research questions")
            logger.info("📊 Cross-company relationships established")
            logger.info("🔗 Multi-filing type analysis enabled")
        else:
            logger.info(f"\n⚠️ PARTIAL SUCCESS")
            logger.info("Some companies failed - check logs for details")

    def _generate_cross_company_analytics(self):
        """Generate cross-company and cross-filing analytics summary"""
        
        logger.info("\n" + "🔗" * 80)
        logger.info("CROSS-COMPANY & CROSS-FILING ANALYTICS")
        logger.info("🔗" * 80)
        
        try:
            with self.neo4j_driver.session() as session:
                
                # Sector coverage analysis
                sector_coverage_query = """
                MATCH (c:Company)
                RETURN c.sector as sector, count(c) as companies
                ORDER BY companies DESC
                """
                
                sector_results = session.run(sector_coverage_query)
                logger.info("📊 SECTOR COVERAGE:")
                for record in sector_results:
                    logger.info(f"  {record['sector']}: {record['companies']} companies")
                
                # Filing type coverage across companies
                filing_coverage_query = """
                MATCH (f:Filing)
                RETURN f.filing_type as filing_type, count(DISTINCT f.ticker) as companies_with_filing
                ORDER BY companies_with_filing DESC
                """
                
                filing_results = session.run(filing_coverage_query)
                logger.info("\n📄 FILING TYPE COVERAGE:")
                for record in filing_results:
                    logger.info(f"  {record['filing_type']}: {record['companies_with_filing']} companies")
                
                # Cross-filing relationships
                relationships_query = """
                MATCH ()-[r:ANNUAL_TO_QUARTERLY|MATERIAL_EVENT_CONTEXT|GOVERNANCE_CONTEXT|INSIDER_TRADING_CONTEXT]->()
                RETURN type(r) as relationship_type, count(r) as relationship_count
                ORDER BY relationship_count DESC
                """
                
                rel_results = session.run(relationships_query)
                logger.info("\n🔗 CROSS-FILING RELATIONSHIPS:")
                for record in rel_results:
                    logger.info(f"  {record['relationship_type']}: {record['relationship_count']} relationships")
                
                # Competitive relationships
                competitive_query = """
                MATCH ()-[r:COMPETITOR|SAME_SECTOR]->()
                RETURN type(r) as relationship_type, count(r) as relationship_count
                """
                
                comp_results = session.run(competitive_query)
                logger.info("\n🏢 COMPETITIVE RELATIONSHIPS:")
                for record in comp_results:
                    logger.info(f"  {record['relationship_type']}: {record['relationship_count']} relationships")
                
                # Total sections with embeddings by filing type
                sections_query = """
                MATCH (s:Section)
                WHERE s.embedding_compressed IS NOT NULL
                RETURN s.filing_type as filing_type, count(s) as sections_with_embeddings
                ORDER BY sections_with_embeddings DESC
                """
                
                sections_results = session.run(sections_query)
                logger.info("\n📊 EMBEDDED SECTIONS BY FILING TYPE:")
                for record in sections_results:
                    logger.info(f"  {record['filing_type']}: {record['sections_with_embeddings']} sections")
                
        except Exception as e:
            logger.error(f"❌ Cross-company analytics error: {e}")
        
        logger.info("\n✅ COMPREHENSIVE SYSTEM READY FOR:")
        logger.info("  🔍 Multi-company financial analysis")
        logger.info("  📊 Cross-sectoral competitive research") 
        logger.info("  📄 Multi-filing type insights")
        logger.info("  🤖 Advanced GPT-4o question answering")
        logger.info("🔗" * 80)

def main():
    """Main execution function"""
    
    ingestion = ComprehensiveSECIngestion()
    success = ingestion.run_comprehensive_ingestion()
    
    print(f"\n{'='*80}")
    if success:
        print("🎉 SEC FILINGS QA AGENT - INGESTION COMPLETE!")
        print("📊 Ready for comprehensive financial research questions")
        print("🔗 Cross-company relationships established in Neo4j")
        print("📄 Multiple SEC filing types processed")
    else:
        print("⚠️ Ingestion completed with some issues")
        print("Check logs for detailed information")
    print(f"{'='*80}")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
