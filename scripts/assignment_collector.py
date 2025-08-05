#!/usr/bin/env python3
"""
Assignment-Specific SEC Data Collection
Comprehensive data ingestion to answer all 10 evaluation questions.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import click
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.assignment_strategy import (
    TARGET_COMPANIES, 
    RESEARCH_FILING_TYPES, 
    TIME_PERIOD,
    FILING_LIMITS,
    EVALUATION_MAPPING
)

# Use the existing CLI framework
from scripts.enhanced_ingest_cli import EnhancedSECPipeline

console = Console()
logger = logging.getLogger(__name__)


class AssignmentDataCollector:
    """Specialized collector for assignment requirements."""
    
    def __init__(self):
        """Initialize the assignment data collector."""
        self.pipeline = None
        self.collected_data = {
            "companies": set(),
            "filing_types": set(),
            "total_filings": 0,
            "evaluation_coverage": {}
        }
    
    async def initialize(self):
        """Initialize the SEC pipeline."""
        console.print("🚀 [bold]Initializing SEC Filings QA Assignment Data Collection[/bold]")
        
        self.pipeline = EnhancedSECPipeline()
        await self.pipeline.initialize()
        
        console.print("✅ Pipeline initialized successfully")
    
    def analyze_coverage_requirements(self) -> Dict:
        """Analyze what data is needed for each evaluation question."""
        coverage_plan = {}
        
        for question_id, requirements in EVALUATION_MAPPING.items():
            companies = requirements["required_companies"]
            if companies == "ALL":
                companies = list(TARGET_COMPANIES.keys())
            
            coverage_plan[question_id] = {
                "description": requirements["description"],
                "companies": companies,
                "filing_types": requirements["required_filings"],
                "sections": requirements["focus_sections"],
                "estimated_filings": len(companies) * len(requirements["required_filings"])
            }
        
        return coverage_plan
    
    def display_collection_plan(self):
        """Display the data collection plan."""
        console.print("\n📋 [bold]SEC Data Collection Plan for Assignment[/bold]")
        
        # Companies table
        companies_table = Table(title="Target Companies (15 companies across 5 sectors)")
        companies_table.add_column("Ticker", style="cyan")
        companies_table.add_column("Company Name", style="white")
        companies_table.add_column("Sector", style="yellow")
        
        sector_mapping = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "NVDA": "Technology",
            "JPM": "Financial", "BAC": "Financial", "GS": "Financial",
            "JNJ": "Healthcare", "PFE": "Healthcare", "UNH": "Healthcare", 
            "XOM": "Energy", "CVX": "Energy",
            "WMT": "Consumer", "AMZN": "Consumer", "TSLA": "Consumer"
        }
        
        for ticker, name in TARGET_COMPANIES.items():
            companies_table.add_row(ticker, name, sector_mapping.get(ticker, "Other"))
        
        console.print(companies_table)
        
        # Filing types table
        filings_table = Table(title="Filing Types for Investment Research")
        filings_table.add_column("Filing Type", style="cyan")
        filings_table.add_column("Purpose", style="white")
        filings_table.add_column("Limit per Company", style="yellow")
        
        filing_purposes = {
            "10-K": "Annual comprehensive business overview",
            "10-Q": "Quarterly financial performance",
            "8-K": "Material events and corporate changes",
            "DEF 14A": "Executive compensation and governance",
            "3": "Initial insider ownership",
            "4": "Changes in insider ownership", 
            "5": "Annual insider ownership",
            "S-1": "New securities registration",
            "S-3": "Seasoned issuer registration",
            "20-F": "Foreign company annual report",
            "13F": "Institutional holdings"
        }
        
        for filing_type in RESEARCH_FILING_TYPES:
            purpose = filing_purposes.get(filing_type, "Investment research")
            limit = FILING_LIMITS.get(filing_type, 10)
            filings_table.add_row(filing_type, purpose, str(limit))
        
        console.print(filings_table)
        
        # Coverage analysis
        coverage_plan = self.analyze_coverage_requirements()
        
        console.print("\n🎯 [bold]Evaluation Questions Coverage[/bold]")
        coverage_table = Table()
        coverage_table.add_column("Question", style="cyan")
        coverage_table.add_column("Description", style="white", max_width=40)
        coverage_table.add_column("Companies", style="yellow")
        coverage_table.add_column("Filings", style="green")
        coverage_table.add_column("Est. Documents", style="red")
        
        total_estimated = 0
        for question_id, plan in coverage_plan.items():
            q_num = question_id.split('_')[1]
            companies_str = f"{len(plan['companies'])} companies"
            filings_str = ", ".join(plan["filing_types"])
            estimated = plan["estimated_filings"]
            total_estimated += estimated
            
            coverage_table.add_row(
                f"Q{q_num}",
                plan["description"][:40] + "..." if len(plan["description"]) > 40 else plan["description"],
                companies_str,
                filings_str,
                str(estimated)
            )
        
        coverage_table.add_row(
            "[bold]TOTAL[/bold]", 
            "[bold]All Questions[/bold]", 
            "[bold]15 companies[/bold]", 
            "[bold]11 filing types[/bold]", 
            f"[bold]~{total_estimated}[/bold]"
        )
        
        console.print(coverage_table)
    
    async def collect_assignment_data(self, max_companies: int = None, test_mode: bool = False):
        """Collect all data needed for the assignment."""
        if test_mode:
            console.print("🧪 [yellow]Running in TEST MODE - limited data collection[/yellow]")
            companies = list(TARGET_COMPANIES.keys())[:3]  # Only 3 companies
            filing_types = ["10-K", "10-Q"]  # Only main filings
            max_filings = 2  # Only 2 filings per type
        else:
            companies = list(TARGET_COMPANIES.keys())
            if max_companies:
                companies = companies[:max_companies]
            filing_types = RESEARCH_FILING_TYPES
            max_filings = 20  # Reasonable limit per company
        
        console.print(f"\n🏗️ [bold]Starting Data Collection[/bold]")
        console.print(f"Companies: {len(companies)}")
        console.print(f"Filing types: {len(filing_types)}")
        console.print(f"Time period: {TIME_PERIOD['start_date']} to {TIME_PERIOD['end_date']}")
        
        with Progress() as progress:
            main_task = progress.add_task("[green]Overall Progress", total=len(companies))
            
            results = {}
            
            for i, ticker in enumerate(companies):
                company_name = TARGET_COMPANIES[ticker]
                
                progress.update(main_task, description=f"[green]Processing {ticker} ({company_name})")
                
                try:
                    result = await self.pipeline.ingest_company(
                        ticker=ticker,
                        filing_types=filing_types,
                        max_filings=max_filings,
                        start_date=TIME_PERIOD["start_date"],
                        end_date=TIME_PERIOD["end_date"]
                    )
                    
                    results[ticker] = result
                    
                    # Update collection stats
                    self.collected_data["companies"].add(ticker)
                    if result.get("success", False):
                        for filing in result.get("filings", []):
                            self.collected_data["filing_types"].add(filing.get("filing_type"))
                            self.collected_data["total_filings"] += 1
                    
                    console.print(f"✅ {ticker}: {result.get('successful_filings', 0)} filings processed")
                    
                except Exception as e:
                    console.print(f"❌ {ticker}: Error - {str(e)}")
                    results[ticker] = {"success": False, "error": str(e)}
                
                progress.advance(main_task)
        
        return results
    
    def analyze_collection_results(self, results: Dict):
        """Analyze and display collection results."""
        console.print("\n📊 [bold]Data Collection Results[/bold]")
        
        # Summary statistics
        total_companies = len(results)
        successful_companies = sum(1 for r in results.values() if r.get("success", False))
        total_filings = sum(r.get("successful_filings", 0) for r in results.values())
        
        summary_table = Table(title="Collection Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")
        summary_table.add_column("Target", style="green")
        summary_table.add_column("Status", style="white")
        
        summary_table.add_row("Companies Processed", str(successful_companies), "10-15", 
                             "✅" if successful_companies >= 10 else "⚠️")
        summary_table.add_row("Total Filings", str(total_filings), "200+", 
                             "✅" if total_filings >= 200 else "⚠️")
        summary_table.add_row("Filing Types", str(len(self.collected_data["filing_types"])), "8+",
                             "✅" if len(self.collected_data["filing_types"]) >= 8 else "⚠️")
        
        console.print(summary_table)
        
        # Detailed results by company
        details_table = Table(title="Results by Company")
        details_table.add_column("Ticker", style="cyan")
        details_table.add_column("Company", style="white", max_width=30)
        details_table.add_column("Status", style="green")
        details_table.add_column("Filings", style="yellow")
        details_table.add_column("Notes", style="red")
        
        for ticker, result in results.items():
            company_name = TARGET_COMPANIES[ticker]
            status = "✅ Success" if result.get("success", False) else "❌ Failed"
            filings = str(result.get("successful_filings", 0))
            notes = result.get("error", "")[:30] if result.get("error") else ""
            
            details_table.add_row(ticker, company_name, status, filings, notes)
        
        console.print(details_table)
        
        # Evaluation readiness
        self.evaluate_assignment_readiness()
    
    def evaluate_assignment_readiness(self):
        """Evaluate if collected data can answer evaluation questions."""
        console.print("\n🎯 [bold]Assignment Readiness Assessment[/bold]")
        
        readiness_table = Table(title="Evaluation Questions Readiness")
        readiness_table.add_column("Question", style="cyan")
        readiness_table.add_column("Description", style="white", max_width=40)
        readiness_table.add_column("Data Status", style="yellow")
        readiness_table.add_column("Readiness", style="green")
        
        for question_id, requirements in EVALUATION_MAPPING.items():
            q_num = question_id.split('_')[1]
            description = requirements["description"]
            
            # Check if we have required companies
            required_companies = requirements["required_companies"]
            if required_companies == "ALL":
                required_companies = list(TARGET_COMPANIES.keys())
            
            companies_coverage = len(set(required_companies) & self.collected_data["companies"])
            companies_needed = len(required_companies) if isinstance(required_companies, list) else len(TARGET_COMPANIES)
            
            # Check if we have required filing types  
            filing_coverage = len(set(requirements["required_filings"]) & self.collected_data["filing_types"])
            filing_needed = len(requirements["required_filings"])
            
            data_status = f"{companies_coverage}/{companies_needed} companies, {filing_coverage}/{filing_needed} filings"
            
            readiness = "✅ Ready" if (companies_coverage >= companies_needed * 0.8 and 
                                    filing_coverage >= filing_needed) else "⚠️ Partial"
            
            readiness_table.add_row(f"Q{q_num}", description[:40], data_status, readiness)
        
        console.print(readiness_table)
    
    async def close(self):
        """Close the pipeline."""
        if self.pipeline:
            await self.pipeline.close()


@click.group()
def cli():
    """SEC Filings Assignment Data Collector"""
    pass


@cli.command()
@click.option('--test-mode', is_flag=True, help='Run in test mode with limited data')
@click.option('--max-companies', type=int, help='Maximum number of companies to process')
@click.option('--plan-only', is_flag=True, help='Only show the collection plan')
def collect_assignment_data(test_mode, max_companies, plan_only):
    """Collect comprehensive SEC data for assignment evaluation."""
    
    async def run_collection():
        collector = AssignmentDataCollector()
        
        try:
            collector.display_collection_plan()
            
            if plan_only:
                console.print("\n✅ Collection plan displayed. Use --plan-only=false to start collection.")
                return
            
            await collector.initialize()
            results = await collector.collect_assignment_data(max_companies, test_mode)
            collector.analyze_collection_results(results)
            
        except Exception as e:
            console.print(f"❌ Collection failed: {str(e)}")
            logger.exception("Collection error")
        finally:
            await collector.close()
    
    asyncio.run(run_collection())


if __name__ == "__main__":
    cli()
