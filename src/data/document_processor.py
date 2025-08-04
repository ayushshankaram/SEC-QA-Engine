"""
Document Processor
Handles cleaning, preprocessing, and structuring of SEC filing documents.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import html
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessedSection:
    """Represents a processed section of a SEC filing."""
    section_name: str
    content: str
    word_count: int
    section_type: str  # text, table, list, etc.
    chunk_id: str = ""  # Unique identifier for this chunk
    metadata: Dict[str, Any] = None


class DocumentProcessor:
    """Processes and cleans SEC filing documents for embedding and storage."""
    
    def __init__(self):
        """Initialize document processor."""
        
        # Section patterns for 10-K filings
        self.section_patterns = {
            "Item 1": r"Item\s+1[\.:]?\s*Business",
            "Item 1A": r"Item\s+1A[\.:]?\s*Risk\s+Factors",
            "Item 2": r"Item\s+2[\.:]?\s*Properties",
            "Item 3": r"Item\s+3[\.:]?\s*Legal\s+Proceedings",
            "Item 4": r"Item\s+4[\.:]?\s*Mine\s+Safety",
            "Item 5": r"Item\s+5[\.:]?\s*Market\s+for\s+Registrant",
            "Item 6": r"Item\s+6[\.:]?\s*Selected\s+Financial\s+Data",
            "Item 7": r"Item\s+7[\.:]?\s*Management['']?s\s+Discussion",
            "Item 7A": r"Item\s+7A[\.:]?\s*Quantitative\s+and\s+Qualitative",
            "Item 8": r"Item\s+8[\.:]?\s*Financial\s+Statements",
            "Item 9": r"Item\s+9[\.:]?\s*Changes\s+in\s+and\s+Disagreements",
            "Item 9A": r"Item\s+9A[\.:]?\s*Controls\s+and\s+Procedures",
            "Item 9B": r"Item\s+9B[\.:]?\s*Other\s+Information",
            "Item 10": r"Item\s+10[\.:]?\s*Directors[,\s]*Executive\s+Officers",
            "Item 11": r"Item\s+11[\.:]?\s*Executive\s+Compensation",
            "Item 12": r"Item\s+12[\.:]?\s*Security\s+Ownership",
            "Item 13": r"Item\s+13[\.:]?\s*Certain\s+Relationships",
            "Item 14": r"Item\s+14[\.:]?\s*Principal\s+Accountant",
            "Item 15": r"Item\s+15[\.:]?\s*Exhibits"
        }
        
        # Patterns for 10-Q sections
        self.quarterly_patterns = {
            "Part I Item 1": r"Part\s+I.*?Item\s+1[\.:]?\s*Financial\s+Statements",
            "Part I Item 2": r"Part\s+I.*?Item\s+2[\.:]?\s*Management['']?s\s+Discussion",
            "Part I Item 3": r"Part\s+I.*?Item\s+3[\.:]?\s*Quantitative\s+and\s+Qualitative",
            "Part I Item 4": r"Part\s+I.*?Item\s+4[\.:]?\s*Controls\s+and\s+Procedures",
            "Part II Item 1": r"Part\s+II.*?Item\s+1[\.:]?\s*Legal\s+Proceedings",
            "Part II Item 1A": r"Part\s+II.*?Item\s+1A[\.:]?\s*Risk\s+Factors",
            "Part II Item 2": r"Part\s+II.*?Item\s+2[\.:]?\s*Unregistered\s+Sales",
            "Part II Item 3": r"Part\s+II.*?Item\s+3[\.:]?\s*Defaults",
            "Part II Item 4": r"Part\s+II.*?Item\s+4[\.:]?\s*Mine\s+Safety",
            "Part II Item 5": r"Part\s+II.*?Item\s+5[\.:]?\s*Other\s+Information",
            "Part II Item 6": r"Part\s+II.*?Item\s+6[\.:]?\s*Exhibits"
        }
        
        # Cleaning patterns
        self.noise_patterns = [
            r"<[^>]+>",  # HTML tags
            r"&[a-zA-Z]+;",  # HTML entities
            r"\s+",  # Multiple whitespace
            r"\n\s*\n\s*\n+",  # Multiple line breaks
            r"Table\s+of\s+Contents.*?(?=Item\s+\d)",  # Table of contents
            r"Page\s+\d+\s+of\s+\d+",  # Page numbers
            r"\*{3,}",  # Multiple asterisks
            r"-{3,}",  # Multiple dashes
            r"_{3,}",  # Multiple underscores
        ]
        
        # Processing statistics
        self.processed_documents = 0
        self.extracted_sections = 0
        self.processing_errors = 0
        
        logger.info("Document processor initialized")
    
    def process_filing(self, content: str, filing_type: str) -> List[ProcessedSection]:
        """Process a complete SEC filing into structured sections."""
        
        try:
            # Clean the document
            cleaned_content = self._clean_document(content)
            
            # Extract sections based on filing type
            if filing_type == "10-K":
                sections = self._extract_10k_sections(cleaned_content)
            elif filing_type == "10-Q":
                sections = self._extract_10q_sections(cleaned_content)
            elif filing_type == "8-K":
                sections = self._extract_8k_sections(cleaned_content)
            else:
                # Generic processing for other forms
                sections = self._extract_generic_sections(cleaned_content, filing_type)
            
            self.processed_documents += 1
            self.extracted_sections += len(sections)
            
            logger.info(f"Processed {filing_type}: {len(sections)} sections extracted")
            
            return sections
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            self.processing_errors += 1
            return []
    
    def _clean_document(self, content: str) -> str:
        """Clean document content of noise and formatting issues."""
        
        cleaned = content
        
        # Decode HTML entities
        cleaned = html.unescape(cleaned)
        
        # Apply noise removal patterns
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n", "\n", cleaned)
        
        # Remove very short lines (likely artifacts)
        lines = cleaned.split("\n")
        meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return "\n".join(meaningful_lines)
    
    def _extract_10k_sections(self, content: str) -> List[ProcessedSection]:
        """Extract sections from 10-K filing."""
        
        sections = []
        
        # Find section boundaries
        section_boundaries = self._find_section_boundaries(content, self.section_patterns)
        
        # Extract content for each section
        for i, (section_name, start_pos) in enumerate(section_boundaries):
            # Determine end position
            if i + 1 < len(section_boundaries):
                end_pos = section_boundaries[i + 1][1]
            else:
                end_pos = len(content)
            
            # Extract section content
            section_content = content[start_pos:end_pos].strip()
            
            if len(section_content) > 100:  # Minimum content threshold
                processed_section = ProcessedSection(
                    section_name=section_name,
                    content=section_content,
                    word_count=len(section_content.split()),
                    section_type="text",
                    metadata={"filing_type": "10-K"}
                )
                sections.append(processed_section)
        
        return sections
    
    def _extract_10q_sections(self, content: str) -> List[ProcessedSection]:
        """Extract sections from 10-Q filing."""
        
        sections = []
        
        # Find section boundaries
        section_boundaries = self._find_section_boundaries(content, self.quarterly_patterns)
        
        # Extract content for each section
        for i, (section_name, start_pos) in enumerate(section_boundaries):
            # Determine end position
            if i + 1 < len(section_boundaries):
                end_pos = section_boundaries[i + 1][1]
            else:
                end_pos = len(content)
            
            # Extract section content
            section_content = content[start_pos:end_pos].strip()
            
            if len(section_content) > 100:  # Minimum content threshold
                processed_section = ProcessedSection(
                    section_name=section_name,
                    content=section_content,
                    word_count=len(section_content.split()),
                    section_type="text",
                    metadata={"filing_type": "10-Q"}
                )
                sections.append(processed_section)
        
        return sections
    
    def _extract_8k_sections(self, content: str) -> List[ProcessedSection]:
        """Extract sections from 8-K filing."""
        
        sections = []
        
        # 8-K forms have different item structures
        item_patterns = {
            "Item 1.01": r"Item\s+1\.01.*?Entry\s+into\s+a\s+Material",
            "Item 1.02": r"Item\s+1\.02.*?Termination\s+of\s+a\s+Material",
            "Item 1.03": r"Item\s+1\.03.*?Bankruptcy\s+or\s+Receivership",
            "Item 1.04": r"Item\s+1\.04.*?Mine\s+Safety",
            "Item 2.01": r"Item\s+2\.01.*?Completion\s+of\s+Acquisition",
            "Item 2.02": r"Item\s+2\.02.*?Results\s+of\s+Operations",
            "Item 2.03": r"Item\s+2\.03.*?Creation\s+of\s+a\s+Direct",
            "Item 2.04": r"Item\s+2\.04.*?Triggering\s+Events",
            "Item 2.05": r"Item\s+2\.05.*?Costs\s+Associated\s+with\s+Exit",
            "Item 2.06": r"Item\s+2\.06.*?Material\s+Impairments",
            "Item 3.01": r"Item\s+3\.01.*?Notice\s+of\s+Delisting",
            "Item 3.02": r"Item\s+3\.02.*?Unregistered\s+Sales",
            "Item 3.03": r"Item\s+3\.03.*?Material\s+Modification",
            "Item 4.01": r"Item\s+4\.01.*?Changes\s+in\s+Registrant",
            "Item 4.02": r"Item\s+4\.02.*?Non-Reliance\s+on\s+Previously",
            "Item 5.01": r"Item\s+5\.01.*?Changes\s+in\s+Control",
            "Item 5.02": r"Item\s+5\.02.*?Departure\s+of\s+Directors",
            "Item 5.03": r"Item\s+5\.03.*?Amendments\s+to\s+Articles",
            "Item 5.04": r"Item\s+5\.04.*?Temporary\s+Suspension",
            "Item 5.05": r"Item\s+5\.05.*?Amendment\s+to\s+Registrant",
            "Item 5.06": r"Item\s+5\.06.*?Change\s+in\s+Shell\s+Company",
            "Item 5.07": r"Item\s+5\.07.*?Submission\s+of\s+Matters",
            "Item 5.08": r"Item\s+5\.08.*?Shareholder\s+Director",
            "Item 6.01": r"Item\s+6\.01.*?ABS\s+Informational",
            "Item 6.02": r"Item\s+6\.02.*?Change\s+of\s+Servicer",
            "Item 6.03": r"Item\s+6\.03.*?Change\s+in\s+Credit\s+Enhancement",
            "Item 6.04": r"Item\s+6\.04.*?Failure\s+to\s+Make\s+a\s+Required",
            "Item 6.05": r"Item\s+6\.05.*?Securities\s+Act\s+Updating",
            "Item 7.01": r"Item\s+7\.01.*?Regulation\s+FD\s+Disclosure",
            "Item 8.01": r"Item\s+8\.01.*?Other\s+Events",
            "Item 9.01": r"Item\s+9\.01.*?Financial\s+Statements\s+and\s+Exhibits"
        }
        
        # Find section boundaries
        section_boundaries = self._find_section_boundaries(content, item_patterns)
        
        # Extract content for each section
        for i, (section_name, start_pos) in enumerate(section_boundaries):
            # Determine end position
            if i + 1 < len(section_boundaries):
                end_pos = section_boundaries[i + 1][1]
            else:
                end_pos = len(content)
            
            # Extract section content
            section_content = content[start_pos:end_pos].strip()
            
            if len(section_content) > 50:  # Lower threshold for 8-K
                processed_section = ProcessedSection(
                    section_name=section_name,
                    content=section_content,
                    word_count=len(section_content.split()),
                    section_type="text",
                    metadata={"filing_type": "8-K"}
                )
                sections.append(processed_section)
        
        return sections
    
    def _extract_generic_sections(self, content: str, filing_type: str) -> List[ProcessedSection]:
        """Extract sections from generic filing types."""
        
        sections = []
        
        # For generic documents, split by major headings or chunks
        chunks = self._split_into_chunks(content, max_chunk_size=5000)
        
        for i, chunk in enumerate(chunks):
            if len(chunk) > 100:
                processed_section = ProcessedSection(
                    section_name=f"Section {i+1}",
                    content=chunk,
                    word_count=len(chunk.split()),
                    section_type="text",
                    metadata={"filing_type": filing_type}
                )
                sections.append(processed_section)
        
        return sections
    
    def _find_section_boundaries(self, content: str, patterns: Dict[str, str]) -> List[Tuple[str, int]]:
        """Find boundaries of sections in the document."""
        
        boundaries = []
        
        for section_name, pattern in patterns.items():
            matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                boundaries.append((section_name, match.start()))
        
        # Sort by position in document
        boundaries.sort(key=lambda x: x[1])
        
        return boundaries
    
    def _split_into_chunks(self, content: str, max_chunk_size: int = 5000) -> List[str]:
        """Split content into manageable chunks."""
        
        chunks = []
        words = content.split()
        
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extract table structures from document content."""
        
        tables = []
        
        # Simple table detection based on patterns
        table_patterns = [
            r"<table[^>]*>.*?</table>",  # HTML tables
            r"\|[^\n]*\|",  # Pipe-separated tables
            r"(?m)^[\w\s]+\t[\w\s\t]+$",  # Tab-separated
        ]
        
        for i, pattern in enumerate(table_patterns):
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for j, match in enumerate(matches):
                table_content = match.group(0)
                
                table_info = {
                    "table_id": f"table_{i}_{j}",
                    "content": table_content,
                    "type": "table",
                    "word_count": len(table_content.split())
                }
                tables.append(table_info)
        
        return tables
    
    def extract_financial_data(self, content: str) -> Dict[str, Any]:
        """Extract financial data patterns from content."""
        
        financial_data = {}
        
        # Revenue patterns
        revenue_patterns = [
            r"revenue[s]?\s*[:\-]?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?",
            r"net\s+revenue[s]?\s*[:\-]?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?",
            r"total\s+revenue[s]?\s*[:\-]?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?"
        ]
        
        # Net income patterns
        income_patterns = [
            r"net\s+income\s*[:\-]?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?",
            r"net\s+earnings\s*[:\-]?\s*\$?([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?"
        ]
        
        # Extract revenue
        revenues = []
        for pattern in revenue_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                amount = match.group(1).replace(",", "")
                unit = match.group(2) or ""
                revenues.append({"amount": amount, "unit": unit})
        
        if revenues:
            financial_data["revenues"] = revenues
        
        # Extract net income
        incomes = []
        for pattern in income_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                amount = match.group(1).replace(",", "")
                unit = match.group(2) or ""
                incomes.append({"amount": amount, "unit": unit})
        
        if incomes:
            financial_data["net_incomes"] = incomes
        
        return financial_data
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get document processing statistics."""
        
        return {
            "processed_documents": self.processed_documents,
            "extracted_sections": self.extracted_sections,
            "processing_errors": self.processing_errors,
            "supported_section_patterns": len(self.section_patterns) + len(self.quarterly_patterns),
            "noise_removal_patterns": len(self.noise_patterns)
        }
    
    def validate_section_content(self, section: ProcessedSection) -> bool:
        """Validate that section content meets quality standards."""
        
        # Check minimum length
        if section.word_count < 10:
            return False
        
        # Check content quality (not just repeated characters)
        content_lower = section.content.lower()
        
        # Reject if too much repetition
        unique_words = set(content_lower.split())
        total_words = len(content_lower.split())
        
        if total_words > 0 and len(unique_words) / total_words < 0.3:
            return False
        
        # Reject if mostly numbers/symbols
        alpha_chars = sum(1 for c in section.content if c.isalpha())
        total_chars = len(section.content)
        
        if total_chars > 0 and alpha_chars / total_chars < 0.5:
            return False
        
        return True


# Helper functions
def estimate_token_count(text: str) -> int:
    """Estimate token count for text content."""
    # Rough approximation: 1 token ≈ 4 characters
    return len(text) // 4


def prioritize_sections_by_importance(sections: List[ProcessedSection]) -> List[ProcessedSection]:
    """Prioritize sections by importance for SEC analysis."""
    
    # Define importance scores for different sections
    importance_scores = {
        "Item 1": 9,    # Business
        "Item 1A": 10,  # Risk Factors
        "Item 7": 10,   # MD&A
        "Item 8": 8,    # Financial Statements
        "Item 2": 6,    # Properties
        "Item 3": 7,    # Legal Proceedings
        "Item 10": 7,   # Directors/Officers
        "Item 11": 6,   # Executive Compensation
        # 10-Q sections
        "Part I Item 2": 10,  # MD&A
        "Part I Item 1": 8,   # Financial Statements
        "Part I Item 3": 8,   # Market Risk
        # 8-K items
        "Item 2.02": 9,  # Results of Operations
        "Item 5.02": 8,  # Departure of Directors/Officers
        "Item 8.01": 7,  # Other Events
    }
    
    # Score sections
    scored_sections = []
    for section in sections:
        score = importance_scores.get(section.section_name, 5)  # Default score
        scored_sections.append((score, section))
    
    # Sort by score (descending) and return sections
    scored_sections.sort(key=lambda x: x[0], reverse=True)
    return [section for _, section in scored_sections]


def summarize_document_content(sections: List[ProcessedSection]) -> Dict[str, Any]:
    """Summarize the content of processed document sections."""
    
    total_words = sum(section.word_count for section in sections)
    section_types = {}
    
    for section in sections:
        section_type = section.section_type
        if section_type not in section_types:
            section_types[section_type] = 0
        section_types[section_type] += 1
    
    return {
        "total_sections": len(sections),
        "total_words": total_words,
        "section_types": section_types,
        "largest_section": max(sections, key=lambda x: x.word_count).section_name if sections else None,
        "estimated_tokens": estimate_token_count(" ".join(s.content for s in sections))
    }
