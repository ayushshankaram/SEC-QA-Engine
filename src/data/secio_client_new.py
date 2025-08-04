"""
SEC EDGAR API Client (Compatibility wrapper)
Handles interactions with the official SEC EDGAR API for fetching SEC filing data.
"""

# Import the new SEC EDGAR client and create compatibility wrapper
from .sec_edgar_client import SECEdgarClient

# Alias for backward compatibility
SECIOClient = SECEdgarClient


def get_major_companies_by_sector() -> dict:
    """
    Returns a dictionary of major companies organized by sector for testing.
    """
    return {
        "Technology": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "Finance": ["JPM", "BAC", "WFC", "GS"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV"],
        "Energy": ["XOM", "CVX", "COP"]
    }


def get_default_companies() -> list:
    """Get list of default companies for bulk ingestion."""
    companies = []
    for sector_companies in get_major_companies_by_sector().values():
        companies.extend(sector_companies)
    return companies
