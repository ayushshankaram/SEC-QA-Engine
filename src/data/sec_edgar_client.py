"""
SEC EDGAR API Client
Handles interactions with the official SEC EDGAR API for fetching SEC filing data.
"""

import os
import logging
import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import json

logger = logging.getLogger(__name__)


class SECEdgarClient:
    """Client for interacting with official SEC EDGAR API."""
    
    def __init__(self, base_url: str = None, user_agent: str = None):
        """Initialize SEC EDGAR API client."""
        self.data_base_url = "https://data.sec.gov"
        self.sec_base_url = "https://www.sec.gov"
        
        # SEC requires a User-Agent header
        self.user_agent = user_agent or "SEC Filings QA Engine (contact@example.com)"
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate"
        })
        
        # Rate limiting - SEC allows 10 requests per second
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests (10 req/sec)
    
    def _rate_limit(self):
        """Implement rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _make_request(self, endpoint: str, params: Dict = None, use_data_api: bool = True) -> Dict:
        """Make a rate-limited request to the SEC EDGAR API."""
        self._rate_limit()
        
        base_url = self.data_base_url if use_data_api else self.sec_base_url
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.get(url, params=params or {})
            response.raise_for_status()
            
            logger.debug(f"API request successful: {endpoint}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {endpoint} - {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise
    
    def get_company_tickers(self) -> Dict[str, Dict]:
        """Get mapping of company tickers to CIKs."""
        try:
            data = self._make_request("files/company_tickers.json", use_data_api=False)
            
            # Convert to ticker -> company info mapping
            ticker_map = {}
            for item in data.values():
                if isinstance(item, dict) and 'ticker' in item:
                    ticker = item['ticker'].upper()
                    ticker_map[ticker] = {
                        'cik': str(item['cik_str']).zfill(10),  # Pad to 10 digits
                        'title': item.get('title', ''),
                        'ticker': ticker
                    }
            
            return ticker_map
            
        except Exception as e:
            logger.error(f"Failed to get company tickers: {e}")
            return {}
    
    def get_company_submissions(self, cik: str) -> Dict:
        """Get company submissions by CIK."""
        # Ensure CIK is 10 digits with leading zeros
        cik_padded = str(cik).zfill(10)
        endpoint = f"submissions/CIK{cik_padded}.json"
        
        try:
            return self._make_request(endpoint)
        except Exception as e:
            logger.error(f"Failed to get submissions for CIK {cik_padded}: {e}")
            return {}
    
    def get_company_info(self, ticker: str) -> Optional[Dict]:
        """Get company information by ticker symbol."""
        ticker_map = self.get_company_tickers()
        
        ticker_upper = ticker.upper()
        if ticker_upper not in ticker_map:
            logger.warning(f"Ticker {ticker} not found in SEC database")
            return None
        
        company_info = ticker_map[ticker_upper]
        
        # Get detailed submissions data
        submissions = self.get_company_submissions(company_info['cik'])
        if submissions:
            company_info.update({
                'name': submissions.get('name', company_info.get('title', '')),
                'sic': submissions.get('sic'),
                'sicDescription': submissions.get('sicDescription'),
                'stateOfIncorporation': submissions.get('stateOfIncorporation'),
                'fiscalYearEnd': submissions.get('fiscalYearEnd'),
                'exchanges': submissions.get('exchanges', []),
                'addresses': {
                    'business': submissions.get('addresses', {}).get('business'),
                    'mailing': submissions.get('addresses', {}).get('mailing')
                }
            })
        
        return company_info
    
    def get_company_filings(self, ticker: str, filing_types: List[str] = None,
                           filing_type: str = None, limit: int = 100, 
                           start_date: str = None, end_date: str = None) -> List[Dict]:
        """Get company filings by ticker symbol."""
        # Handle both filing_types and filing_type parameters for compatibility
        if filing_type and not filing_types:
            filing_types = [filing_type]
        
        company_info = self.get_company_info(ticker)
        if not company_info:
            return []
        
        cik = company_info['cik']
        submissions = self.get_company_submissions(cik)
        
        if not submissions or 'filings' not in submissions:
            return []
        
        filings_data = submissions['filings']['recent']
        
        # Convert to list of filing dictionaries
        filings = []
        for i in range(len(filings_data.get('accessionNumber', []))):
            filing = {
                'accessionNumber': filings_data['accessionNumber'][i],
                'filingDate': filings_data['filingDate'][i],
                'reportDate': filings_data.get('reportDate', [None] * len(filings_data['accessionNumber']))[i],
                'acceptanceDateTime': filings_data.get('acceptanceDateTime', [None] * len(filings_data['accessionNumber']))[i],
                'act': filings_data.get('act', [None] * len(filings_data['accessionNumber']))[i],
                'form': filings_data['form'][i],
                'fileNumber': filings_data.get('fileNumber', [None] * len(filings_data['accessionNumber']))[i],
                'filmNumber': filings_data.get('filmNumber', [None] * len(filings_data['accessionNumber']))[i],
                'items': filings_data.get('items', [None] * len(filings_data['accessionNumber']))[i],
                'size': filings_data.get('size', [None] * len(filings_data['accessionNumber']))[i],
                'isXBRL': filings_data.get('isXBRL', [None] * len(filings_data['accessionNumber']))[i],
                'isInlineXBRL': filings_data.get('isInlineXBRL', [None] * len(filings_data['accessionNumber']))[i],
                'primaryDocument': filings_data.get('primaryDocument', [None] * len(filings_data['accessionNumber']))[i],
                'primaryDocDescription': filings_data.get('primaryDocDescription', [None] * len(filings_data['accessionNumber']))[i],
                # Add compatibility fields
                'filing_type': filings_data['form'][i],
                'accession_number': filings_data['accessionNumber'][i],
                'filing_date': filings_data['filingDate'][i]
            }
            filings.append(filing)
        
        # Filter by filing types
        if filing_types:
            filing_types = [ft.upper() for ft in filing_types]
            filings = [f for f in filings if f['form'] in filing_types]
        
        # Filter by date range
        if start_date:
            filings = [f for f in filings if f['filingDate'] >= start_date]
        if end_date:
            filings = [f for f in filings if f['filingDate'] <= end_date]
        
        # Limit results
        return filings[:limit]
    
    def get_filing_document_url(self, cik: str, accession_number: str, 
                               primary_document: str) -> str:
        """Construct URL for filing document."""
        cik_padded = str(cik).zfill(10)
        accession_clean = accession_number.replace('-', '')
        
        return (f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/"
                f"{accession_clean}/{primary_document}")
    
    def get_filing_content(self, accession_number_or_cik: str, accession_number: str = None, 
                          primary_document: str = None) -> str:
        """Get the content of a filing document."""
        # Handle both old and new calling patterns
        if accession_number is None and primary_document is None:
            # Old style: get_filing_content(accession_number)
            accession_number = accession_number_or_cik
            # For now, return empty content - this needs filing metadata
            logger.warning(f"get_filing_content called with only accession number: {accession_number}")
            return ""
        else:
            # New style: get_filing_content(cik, accession_number, primary_document)
            cik = accession_number_or_cik
            
        url = self.get_filing_document_url(cik, accession_number, primary_document)
        
        try:
            self._rate_limit()
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Failed to get filing content from {url}: {e}")
            return ""
    
    def get_company_facts(self, cik: str) -> Dict:
        """Get XBRL facts for a company."""
        cik_padded = str(cik).zfill(10)
        endpoint = f"api/xbrl/companyfacts/CIK{cik_padded}.json"
        
        try:
            return self._make_request(endpoint)
        except Exception as e:
            logger.error(f"Failed to get company facts for CIK {cik_padded}: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if the SEC EDGAR API is accessible."""
        try:
            # Try to fetch the company tickers file as a health check
            self._make_request("files/company_tickers.json", use_data_api=False)
            return True
        except Exception as e:
            logger.error(f"SEC EDGAR API health check failed: {e}")
            return False
    
    def get_xbrl_data(self, cik: str, accession_number: str) -> Optional[Dict[str, Any]]:
        """
        Get XBRL data for a filing.
        
        Args:
            cik: Company CIK
            accession_number: Filing accession number
            
        Returns:
            Dict containing XBRL data or None if not available
        """
        try:
            # Remove dashes from accession number for URL
            clean_accession = accession_number.replace('-', '')
            
            # Try to get XBRL data from data.sec.gov
            xbrl_url = f"{self.data_base_url}/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
            
            self._rate_limit()
            response = self.session.get(xbrl_url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"XBRL data not available for {accession_number}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching XBRL data for {accession_number}: {str(e)}")
            return None

    def close(self):
        """Close the session."""
        if hasattr(self, 'session'):
            self.session.close()


# Maintain backward compatibility
SECIOClient = SECEdgarClient
