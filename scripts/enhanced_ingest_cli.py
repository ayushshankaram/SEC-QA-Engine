#!/usr/bin/env python3
"""
Enhanced SEC Filings Ingestion CLI
Complete command-line interface for managing the SEC filings QA system.
"""

import os
import sys
import logging
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import pipeline components
from data.enhanced_sec_pipeline import EnhancedSECPipeline, get_default_companies_by_sector
from core.neo4j_retrieval_engine import Neo4jRetrievalEngine
from storage.neo4j_client import Neo4jClient

console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sec_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(debug):
    """SEC Filings QA Engine - Management CLI"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    rprint(Panel.fit("🏦 SEC Filings QA Engine", style="bold blue"))


@cli.command()
@click.option('--ticker', '-t', required=True, help='Company ticker symbol')
@click.option('--filings', '-f', default='10-K,10-Q,8-K,DEF 14A', 
              help='Comma-separated filing types')
@click.option('--max-count', '-m', default=10, help='Maximum filings per company')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
def ingest_company(ticker, filings, max_count, start_date, end_date):
    """Ingest SEC filings for a single company."""
    
    filing_types = [f.strip() for f in filings.split(',')]
    
    console.print(f"\n🏢 Ingesting filings for [bold]{ticker}[/bold]")
    console.print(f"Filing types: {', '.join(filing_types)}")
    console.print(f"Max filings: {max_count}")
    
    if start_date or end_date:
        console.print(f"Date range: {start_date or 'any'} to {end_date or 'any'}")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Initializing pipeline...", total=None)
            
            # Initialize pipeline
            pipeline = EnhancedSECPipeline()
            
            progress.update(task, description="Processing company filings...")
            
            # Ingest company filings
            result = pipeline.ingest_company_filings(
                ticker=ticker,
                filing_types=filing_types,
                max_filings=max_count,
                start_date=start_date,
                end_date=end_date
            )
            
            progress.update(task, description="Completed", completed=True)
        
        # Display results
        if result["success"]:
            console.print(f"\n✅ [green]Success![/green] Processed {result['filings_processed']}/{result['total_filings']} filings")
            
            # Show detailed results
            table = Table(title=f"Filing Results for {ticker}")
            table.add_column("Accession Number", style="cyan")
            table.add_column("Filing Type", style="magenta")
            table.add_column("Status", style="green")
            table.add_column("Sections", justify="right")
            table.add_column("Facts", justify="right")
            
            for filing_result in result.get("results", []):
                status = "✅ Success" if filing_result["success"] else f"❌ {filing_result.get('error', 'Failed')}"
                sections = str(filing_result.get("sections_created", 0))
                facts = str(filing_result.get("facts_processed", 0))
                
                table.add_row(
                    filing_result.get("accession_number", "Unknown"),
                    filing_result.get("filing_type", "Unknown"),
                    status,
                    sections,
                    facts
                )
            
            console.print(table)
            
        else:
            console.print(f"\n❌ [red]Failed:[/red] {result.get('error', 'Unknown error')}")
            
        pipeline.close()
        
    except Exception as e:
        console.print(f"\n❌ [red]Error:[/red] {str(e)}")
        logger.error(f"Company ingestion failed: {e}")


@cli.command()
@click.option('--sector', help='Specific sector to process (Technology, Finance, Healthcare, Energy)')
@click.option('--companies', help='JSON file with custom company list')
@click.option('--start-date', help='Start date for filings (YYYY-MM-DD)')
@click.option('--end-date', help='End date for filings (YYYY-MM-DD)')
@click.option('--dry-run', is_flag=True, help='Show what would be processed without executing')
def bulk_ingest(sector, companies, start_date, end_date, dry_run):
    """Bulk ingest filings for multiple companies across sectors."""
    
    # Determine company list
    if companies:
        with open(companies, 'r') as f:
            company_dict = json.load(f)
    else:
        company_dict = get_default_companies_by_sector()
    
    # Filter by sector if specified
    if sector:
        if sector in company_dict:
            company_dict = {sector: company_dict[sector]}
        else:
            console.print(f"❌ [red]Error:[/red] Sector '{sector}' not found")
            return
    
    # Calculate totals
    total_companies = sum(len(tickers) for tickers in company_dict.values())
    
    console.print(f"\n🏭 Bulk ingestion plan:")
    
    table = Table()
    table.add_column("Sector", style="cyan")
    table.add_column("Companies", justify="right")
    table.add_column("Tickers", style="dim")
    
    for sector_name, tickers in company_dict.items():
        table.add_row(sector_name, str(len(tickers)), ", ".join(tickers))
    
    console.print(table)
    console.print(f"\nTotal: [bold]{total_companies}[/bold] companies")
    
    if start_date or end_date:
        console.print(f"Date range: {start_date or 'any'} to {end_date or 'any'}")
    
    if dry_run:
        console.print("\n🔍 [yellow]Dry run mode - no changes will be made[/yellow]")
        return
    
    # Confirm before proceeding
    if not click.confirm(f"\nProceed with bulk ingestion of {total_companies} companies?"):
        console.print("Operation cancelled.")
        return
    
    try:
        with Progress(console=console) as progress:
            
            # Initialize pipeline
            task_init = progress.add_task("Initializing pipeline...", total=1)
            pipeline = EnhancedSECPipeline()
            progress.complete_task(task_init)
            
            # Bulk ingestion
            task_bulk = progress.add_task("Processing companies...", total=total_companies)
            
            date_range = None
            if start_date and end_date:
                date_range = (start_date, end_date)
            
            result = pipeline.bulk_ingest_companies(
                companies=company_dict,
                date_range=date_range
            )
            
            progress.complete_task(task_bulk)
        
        # Display comprehensive results
        console.print(f"\n🎉 [green]Bulk ingestion completed![/green]")
        
        # Summary statistics
        stats_table = Table(title="Processing Summary")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        
        stats_table.add_row("Companies Successful", str(result["companies_successful"]))
        stats_table.add_row("Total Companies", str(result["companies_total"]))
        stats_table.add_row("Filings Processed", str(result["filings_processed"]))
        stats_table.add_row("Processing Time", f"{result['processing_time']:.1f} seconds")
        
        console.print(stats_table)
        
        # Token usage
        token_usage = result.get("token_usage", {})
        if token_usage:
            console.print(f"\n📊 Token Usage:")
            console.print(f"Estimated tokens used: {token_usage.get('total_tokens_estimated', 0):,}")
            console.print(f"Usage percentage: {token_usage.get('usage_percentage', 0):.1f}%")
        
        # Sector breakdown
        console.print(f"\n📈 Results by Sector:")
        for sector_name, sector_results in result["results_by_sector"].items():
            successful = len([r for r in sector_results if r.get("success", False)])
            total = len(sector_results)
            console.print(f"  {sector_name}: {successful}/{total} companies")
        
        pipeline.close()
        
    except Exception as e:
        console.print(f"\n❌ [red]Error:[/red] {str(e)}")
        logger.error(f"Bulk ingestion failed: {e}")


@cli.command()
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--companies', '-c', help='Comma-separated company tickers')
@click.option('--filings', '-f', help='Comma-separated filing types')
@click.option('--mode', '-m', default='hybrid', 
              type=click.Choice(['semantic', 'text', 'hybrid']),
              help='Search mode')
@click.option('--limit', '-l', default=10, help='Maximum results')
def search(query, companies, filings, mode, limit):
    """Search SEC filings using the QA engine."""
    
    console.print(f"\n🔍 Searching: [bold]'{query}'[/bold]")
    console.print(f"Mode: {mode}")
    
    company_tickers = None
    if companies:
        company_tickers = [c.strip().upper() for c in companies.split(',')]
        console.print(f"Companies: {', '.join(company_tickers)}")
    
    filing_types = None
    if filings:
        filing_types = [f.strip() for f in filings.split(',')]
        console.print(f"Filing types: {', '.join(filing_types)}")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Initializing search engine...", total=None)
            
            # Initialize retrieval engine
            retrieval_engine = Neo4jRetrievalEngine()
            
            progress.update(task, description="Searching filings...")
            
            # Perform search
            results = retrieval_engine.search(
                query=query,
                company_tickers=company_tickers,
                filing_types=filing_types,
                search_mode=mode,
                limit=limit
            )
            
            progress.update(task, description="Completed", completed=True)
        
        # Display results
        if results:
            console.print(f"\n📋 Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                console.print(f"\n[bold cyan]{i}. {result.get('company_name', 'Unknown Company')} ({result.get('ticker', 'N/A')})[/bold cyan]")
                console.print(f"Filing: {result.get('filing_type', 'Unknown')} - {result.get('filing_date', 'Unknown date')}")
                console.print(f"Section: {result.get('section_type', 'Unknown section')}")
                console.print(f"Relevance: {result.get('relevance_score', 0):.3f}")
                
                # Content snippet
                snippet = result.get('content_snippet', result.get('content', ''))[:200]
                console.print(f"[dim]{snippet}...[/dim]")
                
                if result.get('sec_url'):
                    console.print(f"[link]{result['sec_url']}[/link]")
        else:
            console.print("\n📭 No results found.")
        
        retrieval_engine.close()
        
    except Exception as e:
        console.print(f"\n❌ [red]Error:[/red] {str(e)}")
        logger.error(f"Search failed: {e}")


@cli.command()
def status():
    """Show system status and statistics."""
    
    console.print("\n📊 System Status")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Checking system health...", total=None)
            
            # Initialize clients
            neo4j_client = Neo4jClient()
            pipeline = EnhancedSECPipeline()
            
            progress.update(task, description="Getting statistics...")
            
            # Get system statistics
            neo4j_stats = neo4j_client.get_system_stats()
            filing_counts = neo4j_client.get_filing_counts_by_company()
            health = pipeline.health_check()
            
            progress.update(task, description="Completed", completed=True)
        
        # Display statistics
        stats_table = Table(title="Database Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", justify="right", style="green")
        
        stats_table.add_row("Companies", str(neo4j_stats.get("companies", 0)))
        stats_table.add_row("Filings", str(neo4j_stats.get("filings", 0)))
        stats_table.add_row("Sections", str(neo4j_stats.get("sections", 0)))
        stats_table.add_row("XBRL Facts", str(neo4j_stats.get("facts", 0)))
        stats_table.add_row("Executives", str(neo4j_stats.get("executives", 0)))
        stats_table.add_row("Relationships", str(neo4j_stats.get("relationships", 0)))
        
        console.print(stats_table)
        
        # Health status
        overall_status = health.get("status", "unknown")
        status_color = "green" if overall_status == "healthy" else "red"
        console.print(f"\n🏥 Overall Health: [{status_color}]{overall_status.upper()}[/{status_color}]")
        
        # Component health
        components = health.get("components", {})
        health_table = Table(title="Component Health")
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="green")
        
        for component, comp_health in components.items():
            status = comp_health.get("status", "unknown")
            status_emoji = "✅" if status == "healthy" else "❌"
            health_table.add_row(component, f"{status_emoji} {status}")
        
        console.print(health_table)
        
        # Top companies by filing count
        if filing_counts:
            console.print(f"\n🏆 Top Companies by Filing Count:")
            for i, company in enumerate(filing_counts[:5], 1):
                console.print(f"  {i}. {company['company_name']} ({company['ticker']}): {company['filing_count']} filings")
        
        neo4j_client.close()
        pipeline.close()
        
    except Exception as e:
        console.print(f"\n❌ [red]Error:[/red] {str(e)}")
        logger.error(f"Status check failed: {e}")


@cli.command()
@click.option('--sample-size', '-s', default=100, help='Sample size for fitting sparse model')
def fit_sparse(sample_size):
    """Fit the sparse embeddings model on existing content."""
    
    console.print(f"\n🔧 Fitting sparse embeddings model with {sample_size} documents")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Initializing pipeline...", total=None)
            
            pipeline = EnhancedSECPipeline()
            
            progress.update(task, description="Fitting sparse model...")
            
            success = pipeline.fit_sparse_embeddings(sample_size)
            
            progress.update(task, description="Completed", completed=True)
        
        if success:
            console.print("✅ [green]Sparse embeddings model fitted successfully![/green]")
        else:
            console.print("❌ [red]Failed to fit sparse embeddings model[/red]")
        
        pipeline.close()
        
    except Exception as e:
        console.print(f"\n❌ [red]Error:[/red] {str(e)}")
        logger.error(f"Sparse model fitting failed: {e}")


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear all data?')
def clear_database():
    """Clear all data from the database (DANGEROUS!)."""
    
    console.print("\n🗑️  Clearing database...")
    
    try:
        neo4j_client = Neo4jClient()
        neo4j_client.clear_database(confirm=True)
        neo4j_client.close()
        
        console.print("✅ [green]Database cleared successfully![/green]")
        
    except Exception as e:
        console.print(f"\n❌ [red]Error:[/red] {str(e)}")
        logger.error(f"Database clearing failed: {e}")


@cli.command()
def test_pipeline():
    """Run a test of the complete pipeline with a small dataset."""
    
    console.print("\n🧪 Running pipeline test")
    
    try:
        with Progress(console=console) as progress:
            
            task = progress.add_task("Testing pipeline components...", total=5)
            
            # Test 1: Initialize pipeline
            pipeline = EnhancedSECPipeline()
            progress.advance(task)
            
            # Test 2: Health check
            health = pipeline.health_check()
            progress.advance(task)
            
            # Test 3: Test with one small company
            result = pipeline.ingest_company_filings("AAPL", ["10-Q"], max_filings=1)
            progress.advance(task)
            
            # Test 4: Test search
            retrieval_engine = Neo4jRetrievalEngine()
            search_results = retrieval_engine.search("revenue", limit=1)
            progress.advance(task)
            
            # Test 5: Cleanup
            pipeline.close()
            retrieval_engine.close()
            progress.advance(task)
        
        # Display results
        console.print(f"\n📋 Test Results:")
        
        # Health check results
        overall_health = health.get("status", "unknown")
        health_color = "green" if overall_health == "healthy" else "red"
        console.print(f"System Health: [{health_color}]{overall_health}[/{health_color}]")
        
        # Ingestion results
        if result.get("success"):
            console.print(f"✅ Ingestion Test: {result['filings_processed']} filings processed")
        else:
            console.print(f"❌ Ingestion Test: {result.get('error', 'Failed')}")
        
        # Search results
        if search_results:
            console.print(f"✅ Search Test: {len(search_results)} results found")
        else:
            console.print("⚠️  Search Test: No results (may be normal for new database)")
        
        console.print(f"\n🎉 [green]Pipeline test completed![/green]")
        
    except Exception as e:
        console.print(f"\n❌ [red]Test failed:[/red] {str(e)}")
        logger.error(f"Pipeline test failed: {e}")


if __name__ == '__main__':
    cli()
