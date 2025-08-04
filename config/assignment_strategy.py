"""
SEC Filings QA Assignment Strategy
Comprehensive configuration for collecting data to answer evaluation questions.
"""

# Target Companies (10-15 across different sectors)
TARGET_COMPANIES = {
    # Technology Sector
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation", 
    "GOOGL": "Alphabet Inc.",
    "NVDA": "NVIDIA Corporation",
    
    # Financial Services
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corp",
    "GS": "Goldman Sachs Group Inc.",
    
    # Healthcare & Pharmaceuticals
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer Inc.",
    "UNH": "UnitedHealth Group Inc.",
    
    # Energy
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    
    # Consumer/Retail
    "WMT": "Walmart Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc."
}

# Comprehensive Filing Types for Investment Research
RESEARCH_FILING_TYPES = [
    # Core Financial Reports
    "10-K",      # Annual reports - comprehensive business overview
    "10-Q",      # Quarterly reports - recent performance
    
    # Material Events & Changes
    "8-K",       # Current reports - material events, M&A, leadership changes
    
    # Governance & Compensation
    "DEF 14A",   # Proxy statements - executive compensation, governance
    "DEFA14A",   # Additional proxy materials
    
    # Insider Trading (for Question 7)
    "3",         # Initial ownership statements
    "4",         # Changes in ownership
    "5",         # Annual ownership statements
    
    # Registration & Offerings
    "S-1",       # Registration statements for new securities
    "S-3",       # Simplified registration for seasoned issuers
    
    # Foreign Companies (if applicable)
    "20-F",      # Annual report for foreign companies
    
    # Investment Companies
    "13F",       # Institutional investment manager holdings
]

# Time Period Strategy (3-4 years for meaningful trend analysis)
TIME_PERIOD = {
    "start_date": "2021-01-01",  # 4 years of data
    "end_date": "2024-12-31",
    "focus_periods": [
        "2021",  # Post-COVID recovery
        "2022",  # Interest rate changes, inflation
        "2023",  # AI boom year
        "2024"   # Current trends
    ]
}

# Filing Limits per Company (to manage data volume)
FILING_LIMITS = {
    "10-K": 4,      # Last 4 annual reports
    "10-Q": 12,     # Last 12 quarterly reports (3 years)
    "8-K": 20,      # Last 20 material events
    "DEF 14A": 4,   # Last 4 proxy statements
    "3": 10,        # Recent insider trading
    "4": 50,        # Recent insider trading changes
    "5": 4,         # Annual insider statements
    "S-1": 5,       # Recent offerings
    "S-3": 5,       # Recent offerings
    "20-F": 4,      # For foreign companies
    "13F": 4        # Recent institutional holdings
}

# Evaluation Questions Mapping to Required Data
EVALUATION_MAPPING = {
    "question_1": {
        "description": "Primary revenue drivers for major technology companies",
        "required_filings": ["10-K", "10-Q"],
        "required_companies": ["AAPL", "MSFT", "GOOGL", "NVDA"],
        "focus_sections": ["Item 7", "Revenue", "Business segments"]
    },
    
    "question_2": {
        "description": "R&D spending trends across companies",
        "required_filings": ["10-K", "10-Q"],
        "required_companies": ["AAPL", "MSFT", "GOOGL", "NVDA", "JNJ", "PFE"],
        "focus_sections": ["Item 8", "Financial statements", "R&D expenses"]
    },
    
    "question_3": {
        "description": "Working capital changes for financial services",
        "required_filings": ["10-K", "10-Q"],
        "required_companies": ["JPM", "BAC", "GS"],
        "focus_sections": ["Item 8", "Balance sheet", "Cash flow"]
    },
    
    "question_4": {
        "description": "Most commonly cited risk factors across industries",
        "required_filings": ["10-K"],
        "required_companies": "ALL",
        "focus_sections": ["Item 1A", "Risk factors"]
    },
    
    "question_5": {
        "description": "Climate-related risks by industry",
        "required_filings": ["10-K", "DEF 14A"],
        "required_companies": ["XOM", "CVX", "AAPL", "MSFT", "WMT"],
        "focus_sections": ["Item 1A", "Environmental", "Sustainability"]
    },
    
    "question_6": {
        "description": "Executive compensation changes",
        "required_filings": ["DEF 14A"],
        "required_companies": "ALL",
        "focus_sections": ["Compensation discussion", "Summary compensation"]
    },
    
    "question_7": {
        "description": "Significant insider trading activity",
        "required_filings": ["3", "4", "5"],
        "required_companies": "ALL",
        "focus_sections": ["All content"]
    },
    
    "question_8": {
        "description": "AI and automation positioning",
        "required_filings": ["10-K", "10-Q", "8-K"],
        "required_companies": ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA"],
        "focus_sections": ["Item 1", "Item 7", "Business description"]
    },
    
    "question_9": {
        "description": "Recent M&A activity",
        "required_filings": ["8-K", "10-K", "10-Q"],
        "required_companies": "ALL",
        "focus_sections": ["Material agreements", "Acquisitions", "Item 2"]
    },
    
    "question_10": {
        "description": "Competitive advantages descriptions",
        "required_filings": ["10-K"],
        "required_companies": "ALL",
        "focus_sections": ["Item 1", "Competition", "Business strategy"]
    }
}

# Data Quality Requirements
DATA_QUALITY_REQUIREMENTS = {
    "minimum_companies": 10,
    "minimum_filing_types": 8,
    "minimum_time_span_years": 3,
    "required_sections": [
        "Risk Factors",
        "Management Discussion",
        "Financial Statements",
        "Business Description",
        "Executive Compensation"
    ],
    "metadata_requirements": [
        "ticker",
        "filing_type", 
        "filing_date",
        "section_name",
        "accession_number",
        "company_name"
    ]
}
