"""
Forms 3, 4, 5 Processor
Specialized processor for SEC insider trading forms with structured data extraction.
"""

import os
import logging
import re
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, date
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class InsiderTransaction:
    """Represents a single insider trading transaction."""
    executive_name: str
    executive_title: str
    company_ticker: str
    transaction_date: str
    shares: float
    price: float
    transaction_type: str  # Purchase, Sale, Gift, etc.
    security_type: str = "Common Stock"
    ownership_type: str = "Direct"
    form_type: str = "4"  # 3, 4, or 5
    filing_date: str = None
    accession_number: str = None


class Forms345Processor:
    """Processor for SEC Forms 3, 4, and 5 (insider trading documents)."""
    
    def __init__(self):
        """Initialize Forms 3,4,5 processor."""
        
        # Transaction type mappings
        self.transaction_codes = {
            "P": "Purchase",
            "S": "Sale",
            "A": "Award/Grant",
            "G": "Gift",
            "F": "Payment of Exercise Price",
            "M": "Exercise of Derivative",
            "X": "Exercise of Derivative",
            "D": "Disposition",
            "B": "Purchase (Beneficial)",
            "C": "Conversion",
            "E": "Exercise",
            "H": "Holding",
            "I": "Inheritance",
            "J": "Other Acquisition",
            "K": "Other Disposition",
            "L": "Loan",
            "O": "Other",
            "U": "Transfer",
            "V": "Transaction Voluntarily Reported",
            "W": "Will or Bequest",
            "Z": "Deposit"
        }
        
        # Security type mappings
        self.security_types = {
            "common": "Common Stock",
            "preferred": "Preferred Stock",
            "option": "Stock Option",
            "warrant": "Warrant",
            "convertible": "Convertible Security",
            "right": "Rights",
            "unit": "Unit"
        }
        
        # Executive title standardization
        self.title_mappings = {
            "ceo": "Chief Executive Officer",
            "cfo": "Chief Financial Officer",
            "coo": "Chief Operating Officer",
            "cto": "Chief Technology Officer",
            "president": "President",
            "chairman": "Chairman",
            "director": "Director",
            "vp": "Vice President",
            "svp": "Senior Vice President",
            "evp": "Executive Vice President"
        }
        
        # Processing statistics
        self.processed_forms = 0
        self.extracted_transactions = 0
        self.processing_errors = 0
        
        logger.info("Forms 3,4,5 processor initialized")
    
    def process_form_content(self, content: str, form_type: str, 
                           accession_number: str = None) -> List[InsiderTransaction]:
        """Process the content of a Form 3, 4, or 5."""
        
        transactions = []
        
        try:
            # Determine if content is XML or HTML
            if content.strip().startswith('<?xml') or '<ownershipDocument>' in content:
                transactions = self._process_xml_form(content, form_type, accession_number)
            else:
                transactions = self._process_text_form(content, form_type, accession_number)
            
            self.processed_forms += 1
            self.extracted_transactions += len(transactions)
            
            logger.info(f"Processed Form {form_type}: {len(transactions)} transactions extracted")
            
        except Exception as e:
            logger.error(f"Form processing failed for {accession_number}: {e}")
            self.processing_errors += 1
        
        return transactions
    
    def _process_xml_form(self, xml_content: str, form_type: str, 
                         accession_number: str = None) -> List[InsiderTransaction]:
        """Process XML format insider trading form."""
        
        transactions = []
        
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Extract executive information
            executive_info = self._extract_executive_info_xml(root)
            
            # Extract company information
            company_info = self._extract_company_info_xml(root)
            
            # Extract transactions
            transaction_elements = self._find_xml_elements(root, [
                ".//nonDerivativeTransaction",
                ".//derivativeTransaction",
                ".//transaction"
            ])
            
            for trans_elem in transaction_elements:
                transaction = self._parse_xml_transaction(
                    trans_elem, executive_info, company_info, 
                    form_type, accession_number
                )
                if transaction:
                    transactions.append(transaction)
            
        except ET.ParseError as e:
            logger.warning(f"XML parsing failed, trying as HTML: {e}")
            # Fall back to text processing
            transactions = self._process_text_form(xml_content, form_type, accession_number)
        
        return transactions
    
    def _process_text_form(self, text_content: str, form_type: str,
                          accession_number: str = None) -> List[InsiderTransaction]:
        """Process text/HTML format insider trading form."""
        
        transactions = []
        
        try:
            # Extract executive information
            executive_info = self._extract_executive_info_text(text_content)
            
            # Extract company information
            company_info = self._extract_company_info_text(text_content)
            
            # Extract transactions using pattern matching
            transaction_matches = self._find_transaction_patterns(text_content)
            
            for match in transaction_matches:
                transaction = self._parse_text_transaction(
                    match, executive_info, company_info,
                    form_type, accession_number
                )
                if transaction:
                    transactions.append(transaction)
        
        except Exception as e:
            logger.error(f"Text form processing failed: {e}")
        
        return transactions
    
    def _extract_executive_info_xml(self, root: ET.Element) -> Dict[str, str]:
        """Extract executive information from XML."""
        
        info = {"name": "", "title": ""}
        
        # Look for reporting owner information
        owner_elem = root.find(".//reportingOwner")
        if owner_elem is not None:
            # Name
            name_elem = owner_elem.find(".//rptOwnerName")
            if name_elem is not None:
                info["name"] = name_elem.text or ""
            
            # Title
            title_elem = owner_elem.find(".//officerTitle")
            if title_elem is not None:
                info["title"] = self._standardize_title(title_elem.text or "")
        
        return info
    
    def _extract_company_info_xml(self, root: ET.Element) -> Dict[str, str]:
        """Extract company information from XML."""
        
        info = {"ticker": "", "name": ""}
        
        # Look for issuer information
        issuer_elem = root.find(".//issuer")
        if issuer_elem is not None:
            # Ticker
            ticker_elem = issuer_elem.find(".//issuerTradingSymbol")
            if ticker_elem is not None:
                info["ticker"] = ticker_elem.text or ""
            
            # Name
            name_elem = issuer_elem.find(".//issuerName")
            if name_elem is not None:
                info["name"] = name_elem.text or ""
        
        return info
    
    def _extract_executive_info_text(self, text: str) -> Dict[str, str]:
        """Extract executive information from text content."""
        
        info = {"name": "", "title": ""}
        
        # Patterns for executive name
        name_patterns = [
            r"name[:\s]*([A-Z][a-zA-Z\s,\.]+?)(?:\n|title|officer)",
            r"reporting\s+person[:\s]*([A-Z][a-zA-Z\s,\.]+?)(?:\n|title)",
            r"insider[:\s]*([A-Z][a-zA-Z\s,\.]+?)(?:\n|title)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info["name"] = match.group(1).strip()
                break
        
        # Patterns for title
        title_patterns = [
            r"title[:\s]*([A-Za-z\s,\.]+?)(?:\n|date|transaction)",
            r"officer[:\s]*([A-Za-z\s,\.]+?)(?:\n|date|transaction)",
            r"position[:\s]*([A-Za-z\s,\.]+?)(?:\n|date|transaction)"
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info["title"] = self._standardize_title(match.group(1).strip())
                break
        
        return info
    
    def _extract_company_info_text(self, text: str) -> Dict[str, str]:
        """Extract company information from text content."""
        
        info = {"ticker": "", "name": ""}
        
        # Patterns for ticker
        ticker_patterns = [
            r"ticker[:\s]*([A-Z]{1,5})(?:\s|\n|symbol)",
            r"symbol[:\s]*([A-Z]{1,5})(?:\s|\n|exchange)",
            r"trading\s+symbol[:\s]*([A-Z]{1,5})"
        ]
        
        for pattern in ticker_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info["ticker"] = match.group(1).strip().upper()
                break
        
        # Patterns for company name
        name_patterns = [
            r"company[:\s]*([A-Z][a-zA-Z\s,\.&]+?)(?:\n|ticker|symbol)",
            r"issuer[:\s]*([A-Z][a-zA-Z\s,\.&]+?)(?:\n|ticker|symbol)",
            r"corporation[:\s]*([A-Z][a-zA-Z\s,\.&]+?)(?:\n|ticker)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info["name"] = match.group(1).strip()
                break
        
        return info
    
    def _find_transaction_patterns(self, text: str) -> List[Dict[str, str]]:
        """Find transaction data using pattern matching."""
        
        transactions = []
        
        # Pattern for transaction table rows
        transaction_pattern = r"""
            (?P<date>\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}).*?
            (?P<code>[PSAGFMXDBCEIHJKLOUV]).*?
            (?P<shares>[\d,]+(?:\.\d+)?).*?
            (?P<price>\$?[\d,]+(?:\.\d+)?)
        """
        
        matches = re.finditer(transaction_pattern, text, re.VERBOSE | re.IGNORECASE)
        
        for match in matches:
            transaction_data = {
                "date": match.group("date"),
                "code": match.group("code"),
                "shares": match.group("shares").replace(",", ""),
                "price": match.group("price").replace("$", "").replace(",", "")
            }
            transactions.append(transaction_data)
        
        return transactions
    
    def _parse_xml_transaction(self, trans_elem: ET.Element, 
                              executive_info: Dict, company_info: Dict,
                              form_type: str, accession_number: str) -> Optional[InsiderTransaction]:
        """Parse a transaction from XML element."""
        
        try:
            # Transaction date
            date_elem = trans_elem.find(".//transactionDate/value")
            if date_elem is None:
                date_elem = trans_elem.find(".//transactionDate")
            
            transaction_date = date_elem.text if date_elem is not None else ""
            
            # Transaction code
            code_elem = trans_elem.find(".//transactionCode")
            transaction_code = code_elem.text if code_elem is not None else ""
            
            # Shares
            shares_elem = trans_elem.find(".//transactionShares/value")
            if shares_elem is None:
                shares_elem = trans_elem.find(".//transactionShares")
            
            shares = float(shares_elem.text) if shares_elem is not None else 0.0
            
            # Price
            price_elem = trans_elem.find(".//transactionPricePerShare/value")
            if price_elem is None:
                price_elem = trans_elem.find(".//transactionPricePerShare")
            
            price = float(price_elem.text) if price_elem is not None else 0.0
            
            # Create transaction object
            transaction = InsiderTransaction(
                executive_name=executive_info.get("name", ""),
                executive_title=executive_info.get("title", ""),
                company_ticker=company_info.get("ticker", ""),
                transaction_date=transaction_date,
                shares=shares,
                price=price,
                transaction_type=self.transaction_codes.get(transaction_code, transaction_code),
                form_type=form_type,
                accession_number=accession_number
            )
            
            return transaction
            
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse XML transaction: {e}")
            return None
    
    def _parse_text_transaction(self, transaction_data: Dict, 
                               executive_info: Dict, company_info: Dict,
                               form_type: str, accession_number: str) -> Optional[InsiderTransaction]:
        """Parse a transaction from text data."""
        
        try:
            # Parse date
            date_str = transaction_data.get("date", "")
            if "/" in date_str:
                # Convert MM/DD/YYYY to YYYY-MM-DD
                month, day, year = date_str.split("/")
                transaction_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            else:
                transaction_date = date_str
            
            # Parse shares and price
            shares = float(transaction_data.get("shares", "0"))
            price = float(transaction_data.get("price", "0"))
            
            # Get transaction type
            code = transaction_data.get("code", "")
            transaction_type = self.transaction_codes.get(code, code)
            
            # Create transaction object
            transaction = InsiderTransaction(
                executive_name=executive_info.get("name", ""),
                executive_title=executive_info.get("title", ""),
                company_ticker=company_info.get("ticker", ""),
                transaction_date=transaction_date,
                shares=shares,
                price=price,
                transaction_type=transaction_type,
                form_type=form_type,
                accession_number=accession_number
            )
            
            return transaction
            
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to parse text transaction: {e}")
            return None
    
    def _find_xml_elements(self, root: ET.Element, xpaths: List[str]) -> List[ET.Element]:
        """Find XML elements using multiple XPath expressions."""
        
        elements = []
        for xpath in xpaths:
            found = root.findall(xpath)
            elements.extend(found)
        
        return elements
    
    def _standardize_title(self, title: str) -> str:
        """Standardize executive title."""
        
        if not title:
            return ""
        
        title_lower = title.lower().strip()
        
        # Direct mapping
        if title_lower in self.title_mappings:
            return self.title_mappings[title_lower]
        
        # Partial matching
        for key, standard_title in self.title_mappings.items():
            if key in title_lower:
                return standard_title
        
        # Return original if no mapping found
        return title.strip()
    
    def store_transactions_in_neo4j(self, transactions: List[InsiderTransaction],
                                   neo4j_client) -> int:
        """Store insider transactions in Neo4j database."""
        
        stored_count = 0
        
        for transaction in transactions:
            try:
                # Create executive if not exists
                executive_id = neo4j_client.create_executive(
                    name=transaction.executive_name,
                    title=transaction.executive_title,
                    company_ticker=transaction.company_ticker
                )
                
                if executive_id:
                    # Create trading transaction relationship
                    transaction_id = neo4j_client.create_trading_transaction(
                        executive_id=executive_id,
                        company_ticker=transaction.company_ticker,
                        transaction_date=transaction.transaction_date,
                        shares=transaction.shares,
                        price=transaction.price,
                        transaction_type=transaction.transaction_type
                    )
                    
                    if transaction_id:
                        stored_count += 1
                        
            except Exception as e:
                logger.error(f"Failed to store transaction in Neo4j: {e}")
        
        logger.info(f"Stored {stored_count}/{len(transactions)} transactions in Neo4j")
        return stored_count
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        
        return {
            "processed_forms": self.processed_forms,
            "extracted_transactions": self.extracted_transactions,
            "processing_errors": self.processing_errors,
            "supported_transaction_codes": len(self.transaction_codes),
            "supported_security_types": len(self.security_types),
            "title_mappings": len(self.title_mappings)
        }
    
    def analyze_insider_activity(self, transactions: List[InsiderTransaction]) -> Dict[str, Any]:
        """Analyze patterns in insider trading activity."""
        
        if not transactions:
            return {}
        
        # Group by executive
        executives = {}
        for trans in transactions:
            exec_name = trans.executive_name
            if exec_name not in executives:
                executives[exec_name] = {
                    "transactions": [],
                    "total_shares": 0,
                    "total_value": 0,
                    "purchases": 0,
                    "sales": 0
                }
            
            executives[exec_name]["transactions"].append(trans)
            executives[exec_name]["total_shares"] += abs(trans.shares)
            executives[exec_name]["total_value"] += abs(trans.shares * trans.price)
            
            if trans.transaction_type in ["Purchase", "Award/Grant"]:
                executives[exec_name]["purchases"] += 1
            elif trans.transaction_type in ["Sale", "Disposition"]:
                executives[exec_name]["sales"] += 1
        
        # Top traders by volume
        top_traders = sorted(
            executives.items(),
            key=lambda x: x[1]["total_value"],
            reverse=True
        )[:5]
        
        # Transaction type distribution
        type_counts = {}
        for trans in transactions:
            trans_type = trans.transaction_type
            type_counts[trans_type] = type_counts.get(trans_type, 0) + 1
        
        return {
            "total_transactions": len(transactions),
            "unique_executives": len(executives),
            "top_traders": [(name, data["total_value"]) for name, data in top_traders],
            "transaction_types": type_counts,
            "total_value": sum(abs(t.shares * t.price) for t in transactions),
            "average_transaction_size": sum(abs(t.shares) for t in transactions) / len(transactions)
        }


# Helper functions
def detect_form_type(content: str) -> str:
    """Detect the type of insider trading form."""
    
    content_upper = content.upper()
    
    if "FORM 3" in content_upper:
        return "3"
    elif "FORM 4" in content_upper:
        return "4"
    elif "FORM 5" in content_upper:
        return "5"
    else:
        # Default to Form 4 (most common)
        return "4"


def validate_transaction_data(transaction: InsiderTransaction) -> bool:
    """Validate that transaction data is complete and reasonable."""
    
    # Required fields
    if not transaction.executive_name or not transaction.company_ticker:
        return False
    
    # Date format validation
    try:
        datetime.strptime(transaction.transaction_date, "%Y-%m-%d")
    except ValueError:
        return False
    
    # Numeric validations
    if transaction.shares < 0 or transaction.price < 0:
        return False
    
    # Reasonable limits
    if transaction.shares > 100000000 or transaction.price > 10000:  # Basic sanity checks
        return False
    
    return True


def summarize_insider_activity_by_company(transactions: List[InsiderTransaction]) -> Dict[str, Any]:
    """Summarize insider trading activity grouped by company."""
    
    companies = {}
    
    for trans in transactions:
        ticker = trans.company_ticker
        if ticker not in companies:
            companies[ticker] = {
                "transactions": 0,
                "executives": set(),
                "total_value": 0,
                "net_shares": 0,  # Positive for net buying, negative for net selling
                "latest_activity": None
            }
        
        companies[ticker]["transactions"] += 1
        companies[ticker]["executives"].add(trans.executive_name)
        companies[ticker]["total_value"] += abs(trans.shares * trans.price)
        
        # Calculate net shares (purchases positive, sales negative)
        if trans.transaction_type in ["Purchase", "Award/Grant"]:
            companies[ticker]["net_shares"] += trans.shares
        elif trans.transaction_type in ["Sale", "Disposition"]:
            companies[ticker]["net_shares"] -= trans.shares
        
        # Track latest activity
        if (companies[ticker]["latest_activity"] is None or 
            trans.transaction_date > companies[ticker]["latest_activity"]):
            companies[ticker]["latest_activity"] = trans.transaction_date
    
    # Convert sets to counts
    for company_data in companies.values():
        company_data["unique_executives"] = len(company_data["executives"])
        del company_data["executives"]  # Remove set object
    
    return companies
