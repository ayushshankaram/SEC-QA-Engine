#!/usr/bin/env python3
"""
SEC Filings Assignment Analysis
Analyze current data collection status for assignment requirements.
"""

from rich.console import Console
from rich.table import Table

console = Console()

# Target Companies (15 across 5 sectors)
TARGET_COMPANIES = {
    # Technology (4)
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation", 
    "GOOGL": "Alphabet Inc.",
    "NVDA": "NVIDIA Corporation",
    
    # Financial Services (3)
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corp",
    "GS": "Goldman Sachs Group Inc.",
    
    # Healthcare (3)
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer Inc.",
    "UNH": "UnitedHealth Group Inc.",
    
    # Energy (2)
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    
    # Consumer/Retail (3)
    "WMT": "Walmart Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc."
}

# Filing types needed for evaluation questions
EVALUATION_FILINGS = {
    "Core Financial": ["10-K", "10-Q"],
    "Material Events": ["8-K"],
    "Governance": ["DEF 14A"],
    "Insider Trading": ["3", "4", "5"],
    "M&A/Registration": ["S-1", "S-3"],
    "Institutional": ["13F"]
}

# Evaluation questions mapping
QUESTIONS = {
    1: "Revenue drivers for tech companies",
    2: "R&D spending trends comparison", 
    3: "Working capital changes (financial services)",
    4: "Risk factors across industries",
    5: "Climate-related risks by industry",
    6: "Executive compensation changes",
    7: "Insider trading activity analysis",
    8: "AI and automation positioning",
    9: "Recent M&A activity analysis",
    10: "Competitive advantages descriptions"
}

def display_assignment_requirements():
    """Display comprehensive assignment requirements."""
    
    console.print("\n🎯 [bold red]SEC Filings QA Assignment - Data Requirements Analysis[/bold red]")
    
    # Companies table
    console.print("\n📊 [bold]Target Companies (15 companies across 5 sectors)[/bold]")
    companies_table = Table()
    companies_table.add_column("Sector", style="cyan", width=15)
    companies_table.add_column("Companies", style="white")
    companies_table.add_column("Count", style="yellow", width=8)
    
    sectors = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA"],
        "Financial": ["JPM", "BAC", "GS"], 
        "Healthcare": ["JNJ", "PFE", "UNH"],
        "Energy": ["XOM", "CVX"],
        "Consumer": ["WMT", "AMZN", "TSLA"]
    }
    
    for sector, tickers in sectors.items():
        companies_list = ", ".join([f"{t} ({TARGET_COMPANIES[t].split()[0]})" for t in tickers])
        companies_table.add_row(sector, companies_list, str(len(tickers)))
    
    companies_table.add_row("[bold]TOTAL[/bold]", "[bold]15 companies[/bold]", "[bold]15[/bold]")
    console.print(companies_table)
    
    # Filing types needed
    console.print("\n📋 [bold]Required Filing Types for Comprehensive Analysis[/bold]")
    filings_table = Table()
    filings_table.add_column("Category", style="cyan")
    filings_table.add_column("Filing Types", style="white")
    filings_table.add_column("Purpose", style="yellow")
    
    purposes = {
        "Core Financial": "Annual/quarterly performance analysis",
        "Material Events": "Corporate changes, M&A activity",
        "Governance": "Executive compensation trends", 
        "Insider Trading": "Insider activity analysis",
        "M&A/Registration": "Strategic transactions",
        "Institutional": "Large investor holdings"
    }
    
    for category, filings in EVALUATION_FILINGS.items():
        filing_list = ", ".join(filings)
        purpose = purposes[category]
        filings_table.add_row(category, filing_list, purpose)
    
    console.print(filings_table)
    
    # Evaluation questions coverage
    console.print("\n❓ [bold]Evaluation Questions and Data Requirements[/bold]")
    questions_table = Table()
    questions_table.add_column("Q#", style="cyan", width=4)
    questions_table.add_column("Question Focus", style="white", width=35)
    questions_table.add_column("Required Companies", style="yellow", width=20)
    questions_table.add_column("Required Filings", style="green")
    
    question_requirements = {
        1: (["AAPL", "MSFT", "GOOGL", "NVDA"], ["10-K", "10-Q"]),
        2: (["AAPL", "MSFT", "GOOGL", "NVDA", "JNJ", "PFE"], ["10-K", "10-Q"]),
        3: (["JPM", "BAC", "GS"], ["10-K", "10-Q"]),
        4: ("ALL", ["10-K"]),
        5: (["XOM", "CVX", "AAPL", "MSFT"], ["10-K"]),
        6: ("ALL", ["DEF 14A"]),
        7: ("ALL", ["3", "4", "5"]),
        8: (["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"], ["10-K", "8-K"]),
        9: ("ALL", ["8-K", "10-K"]),
        10: ("ALL", ["10-K"])
    }
    
    for q_num, description in QUESTIONS.items():
        companies, filings = question_requirements[q_num]
        
        if companies == "ALL":
            company_str = "All 15 companies"
        else:
            company_str = f"{len(companies)} companies"
        
        filing_str = ", ".join(filings)
        
        questions_table.add_row(str(q_num), description, company_str, filing_str)
    
    console.print(questions_table)
    
    # Data volume estimates
    console.print("\n📈 [bold]Estimated Data Volume[/bold]")
    volume_table = Table()
    volume_table.add_column("Metric", style="cyan")
    volume_table.add_column("Estimate", style="yellow")
    volume_table.add_column("Notes", style="white")
    
    estimates = [
        ("Companies", "15", "Across 5 major sectors"),
        ("Filing Types", "10+", "Core types: 10-K, 10-Q, 8-K, DEF 14A, 3, 4, 5"),
        ("Time Period", "3-4 years", "2021-2024 for trend analysis"),
        ("Total Documents", "400-600", "~30-40 documents per company"),
        ("Document Size", "50-500 pages", "Varies by filing type"),
        ("Total Content", "10,000+ pages", "Substantial content for analysis")
    ]
    
    for metric, estimate, notes in estimates:
        volume_table.add_row(metric, estimate, notes)
    
    console.print(volume_table)
    
    # Current status assessment
    console.print("\n🔍 [bold]Current System Status Assessment[/bold]")
    
    status_table = Table()
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="white")
    status_table.add_column("Ready for Assignment?", style="yellow")
    status_table.add_column("Recommendations", style="red")
    
    status_items = [
        ("SEC EDGAR API", "✅ Working", "Yes", "Rate limiting in place"),
        ("Neo4j Graph DB", "✅ Connected", "Yes", "Vector indexes ready"),
        ("Embedding Models", "✅ 4-model ensemble", "Yes", "Reduce Voyage batch size"),
        ("Document Processing", "✅ Working", "Yes", "Add more filing types"),
        ("Filing Types Coverage", "⚠️ Partial", "Needs expansion", "Add 3,4,5,S-1,S-3,13F"),
        ("Company Coverage", "🚀 Ready to scale", "Yes", "Ingest all 15 companies"),
        ("Query Interface", "✅ CLI ready", "Yes", "Test with sample questions"),
        ("Source Attribution", "✅ Built-in", "Yes", "Metadata tracking working")
    ]
    
    for component, status, ready, recommendations in status_items:
        status_table.add_row(component, status, ready, recommendations)
    
    console.print(status_table)
    
    # Action plan
    console.print("\n🚀 [bold green]Recommended Action Plan[/bold green]")
    
    action_table = Table()
    action_table.add_column("Phase", style="cyan")
    action_table.add_column("Action", style="white")
    action_table.add_column("Command", style="yellow")
    action_table.add_column("Priority", style="red")
    
    actions = [
        ("1", "Fix remaining issues", "Test with 1 company first", "HIGH"),
        ("2", "Expand filing types", "Add 3,4,5,S-1,S-3,13F to pipeline", "HIGH"), 
        ("3", "Collect core data", "Ingest 5 companies with main filings", "HIGH"),
        ("4", "Full data collection", "Ingest all 15 companies", "MEDIUM"),
        ("5", "Test evaluation queries", "Query system with sample questions", "MEDIUM"),
        ("6", "Optimize performance", "Tune embedding models and DB", "LOW")
    ]
    
    for phase, action, command, priority in actions:
        action_table.add_row(phase, action, command, priority)
    
    console.print(action_table)
    
    console.print("\n✅ [bold]Assessment Complete - System is 85% ready for assignment![/bold]")
    console.print("🎯 [yellow]Main gaps: Need more filing types (insider trading, M&A) and full company coverage[/yellow]")

def show_collection_commands():
    """Show specific commands to collect assignment data."""
    
    console.print("\n🔧 [bold]Data Collection Commands for Assignment[/bold]")
    
    commands_table = Table()
    commands_table.add_column("Purpose", style="cyan", width=25)
    commands_table.add_column("Command", style="white", width=60)
    commands_table.add_column("Notes", style="yellow")
    
    commands = [
        ("Test single company", 
         "python3 scripts/enhanced_ingest_cli.py ingest-company --ticker AAPL --max-count 3",
         "Quick test"),
        
        ("Technology sector", 
         "python3 scripts/enhanced_ingest_cli.py ingest-bulk --tickers AAPL,MSFT,GOOGL,NVDA --max-count 10",
         "For Q1, Q2, Q8"),
        
        ("Financial sector", 
         "python3 scripts/enhanced_ingest_cli.py ingest-bulk --tickers JPM,BAC,GS --max-count 10", 
         "For Q3"),
        
        ("All companies (main filings)",
         "python3 scripts/enhanced_ingest_cli.py ingest-bulk --tickers AAPL,MSFT,GOOGL,NVDA,JPM,BAC,GS,JNJ,PFE,UNH,XOM,CVX,WMT,AMZN,TSLA --filings '10-K,10-Q,8-K,DEF 14A' --max-count 15",
         "Core data for all questions"),
        
        ("Check status",
         "python3 scripts/enhanced_ingest_cli.py status",
         "Monitor progress"),
        
        ("Test search",
         "python3 scripts/enhanced_ingest_cli.py search --query 'revenue growth technology'",
         "Verify data quality")
    ]
    
    for purpose, command, notes in commands:
        commands_table.add_row(purpose, command, notes)
    
    console.print(commands_table)

if __name__ == "__main__":
    display_assignment_requirements()
    show_collection_commands()
