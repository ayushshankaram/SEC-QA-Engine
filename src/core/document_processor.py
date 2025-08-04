"""
Document Processor for SEC Filings
Handles parsing, chunking, and structure preservation for SEC documents.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a processed document."""
    content: str
    section_type: str
    chunk_id: str
    metadata: Dict
    start_position: int
    end_position: int
    embedding: Optional[List[float]] = None


class DocumentProcessor:
    """Processes SEC filing documents with structure preservation."""
    
    def __init__(self, max_chunk_size: int = 8000, chunk_overlap: int = 400):
        """Initialize document processor with chunking parameters."""
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # SEC document section patterns
        self.section_patterns = {
            "business": [
                r"item\s*1\b.*?business",
                r"business\s*(overview|description)",
                r"our\s*business"
            ],
            "risk_factors": [
                r"item\s*1a\b.*?risk\s*factors",
                r"risk\s*factors",
                r"factors\s*that\s*may\s*affect"
            ],
            "properties": [
                r"item\s*2\b.*?properties",
                r"properties"
            ],
            "legal_proceedings": [
                r"item\s*3\b.*?legal\s*proceedings",
                r"legal\s*proceedings",
                r"litigation"
            ],
            "selected_financial_data": [
                r"item\s*6\b.*?selected.*?financial.*?data",
                r"selected.*?financial.*?data"
            ],
            "md_a": [
                r"item\s*7\b.*?management.*?discussion.*?analysis",
                r"management.*?discussion.*?analysis",
                r"md&a",
                r"mda"
            ],
            "financial_statements": [
                r"item\s*8\b.*?financial\s*statements",
                r"financial\s*statements",
                r"consolidated\s*statements"
            ],
            "controls_procedures": [
                r"item\s*9a\b.*?controls.*?procedures",
                r"controls\s*and\s*procedures",
                r"internal\s*controls"
            ],
            "director_compensation": [
                r"director.*?compensation",
                r"compensation.*?discussion",
                r"executive\s*compensation"
            ],
            "audit_committee": [
                r"audit\s*committee",
                r"committee.*?report"
            ]
        }
    
    def process_document(self, content: str, filing_type: str, 
                        accession_number: str) -> List[DocumentChunk]:
        """Process a complete SEC filing document."""
        try:
            # Clean and normalize content
            cleaned_content = self._clean_content(content)
            
            # Extract sections based on filing type
            sections = self._extract_sections(cleaned_content, filing_type)
            
            # Generate chunks from sections
            chunks = []
            for section_type, section_content in sections.items():
                section_chunks = self._chunk_section(
                    section_content, 
                    section_type, 
                    accession_number
                )
                chunks.extend(section_chunks)
            
            logger.info(f"Processed {filing_type} document: {len(chunks)} chunks generated")
            return chunks
            
        except Exception as e:
            logger.error(f"Document processing failed for {accession_number}: {e}")
            return []
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize document content."""
        try:
            # Handle HTML content
            if content.strip().startswith('<'):
                soup = BeautifulSoup(content, 'html.parser')
                # Remove scripts, styles, and other non-content elements
                for element in soup(['script', 'style', 'meta', 'link']):
                    element.decompose()
                content = soup.get_text()
            
            # Normalize whitespace
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\n\s*\n', '\n\n', content)
            
            # Remove excessive punctuation
            content = re.sub(r'[.]{3,}', '...', content)
            content = re.sub(r'[-]{3,}', '---', content)
            
            # Clean up common SEC filing artifacts
            content = re.sub(r'Table of Contents', '', content, flags=re.IGNORECASE)
            content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
            
            return content.strip()
            
        except Exception as e:
            logger.warning(f"Content cleaning failed: {e}")
            return content
    
    def _extract_sections(self, content: str, filing_type: str) -> Dict[str, str]:
        """Extract specific sections from the document."""
        sections = {}
        content_lower = content.lower()
        
        # Extract sections based on patterns
        for section_name, patterns in self.section_patterns.items():
            section_content = self._extract_section_by_patterns(
                content, content_lower, patterns
            )
            if section_content:
                sections[section_name] = section_content
        
        # If no specific sections found, treat entire content as one section
        if not sections:
            sections[f"{filing_type.lower()}_content"] = content
        
        return sections
    
    def _extract_section_by_patterns(self, content: str, content_lower: str, 
                                   patterns: List[str]) -> Optional[str]:
        """Extract a section using regex patterns."""
        for pattern in patterns:
            try:
                match = re.search(pattern, content_lower)
                if match:
                    start_pos = match.start()
                    
                    # Find the end of the section (next section or document end)
                    next_section_pos = self._find_next_section_start(
                        content_lower, start_pos + len(match.group())
                    )
                    
                    if next_section_pos:
                        section_text = content[start_pos:next_section_pos]
                    else:
                        section_text = content[start_pos:]
                    
                    # Clean and validate section
                    section_text = section_text.strip()
                    if len(section_text) > 100:  # Minimum section length
                        return section_text
                        
            except re.error as e:
                logger.warning(f"Regex pattern error: {pattern} - {e}")
                continue
        
        return None
    
    def _find_next_section_start(self, content_lower: str, start_pos: int) -> Optional[int]:
        """Find the start position of the next section."""
        # Look for item patterns that indicate new sections
        item_patterns = [
            r"\n\s*item\s*\d+[a-z]?\b",
            r"\n\s*part\s*[iv]+",
            r"\n\s*signature[s]?\s*\n",
        ]
        
        min_pos = None
        for pattern in item_patterns:
            match = re.search(pattern, content_lower[start_pos:])
            if match:
                pos = start_pos + match.start()
                if min_pos is None or pos < min_pos:
                    min_pos = pos
        
        return min_pos
    
    def _chunk_section(self, section_content: str, section_type: str, 
                      accession_number: str) -> List[DocumentChunk]:
        """Split a section into appropriately sized chunks."""
        chunks = []
        
        if len(section_content) <= self.max_chunk_size:
            # Section fits in one chunk
            chunk_id = self._generate_chunk_id(accession_number, section_type, 0)
            chunk = DocumentChunk(
                content=section_content,
                section_type=section_type,
                chunk_id=chunk_id,
                metadata={
                    "accession_number": accession_number,
                    "total_chunks": 1,
                    "chunk_index": 0,
                    "character_count": len(section_content)
                },
                start_position=0,
                end_position=len(section_content)
            )
            chunks.append(chunk)
        else:
            # Split section into multiple chunks with overlap
            chunk_starts = self._calculate_chunk_positions(section_content)
            
            for i, start_pos in enumerate(chunk_starts):
                end_pos = min(start_pos + self.max_chunk_size, len(section_content))
                
                # Adjust end position to word boundary
                if end_pos < len(section_content):
                    end_pos = self._find_word_boundary(section_content, end_pos, direction='backward')
                
                chunk_content = section_content[start_pos:end_pos].strip()
                
                if chunk_content:  # Skip empty chunks
                    chunk_id = self._generate_chunk_id(accession_number, section_type, i)
                    chunk = DocumentChunk(
                        content=chunk_content,
                        section_type=section_type,
                        chunk_id=chunk_id,
                        metadata={
                            "accession_number": accession_number,
                            "total_chunks": len(chunk_starts),
                            "chunk_index": i,
                            "character_count": len(chunk_content)
                        },
                        start_position=start_pos,
                        end_position=end_pos
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _calculate_chunk_positions(self, content: str) -> List[int]:
        """Calculate optimal positions for chunk boundaries."""
        positions = [0]
        current_pos = 0
        
        while current_pos + self.max_chunk_size < len(content):
            # Calculate next chunk start with overlap
            next_pos = current_pos + self.max_chunk_size - self.chunk_overlap
            
            # Adjust to word boundary
            next_pos = self._find_word_boundary(content, next_pos, direction='forward')
            
            if next_pos <= current_pos:
                next_pos = current_pos + self.max_chunk_size // 2
            
            positions.append(next_pos)
            current_pos = next_pos
        
        return positions
    
    def _find_word_boundary(self, content: str, position: int, direction: str = 'backward') -> int:
        """Find the nearest word boundary to avoid splitting words."""
        if position <= 0:
            return 0
        if position >= len(content):
            return len(content)
        
        if direction == 'backward':
            # Look backward for whitespace or punctuation
            for i in range(position, max(0, position - 100), -1):
                if content[i] in ' \n\t.!?;':
                    return i + 1
        else:
            # Look forward for whitespace or punctuation
            for i in range(position, min(len(content), position + 100)):
                if content[i] in ' \n\t.!?;':
                    return i + 1
        
        # Fallback to original position
        return position
    
    def _generate_chunk_id(self, accession_number: str, section_type: str, 
                          chunk_index: int) -> str:
        """Generate a unique identifier for a chunk."""
        content = f"{accession_number}_{section_type}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def process_xbrl_data(self, xbrl_data: Dict) -> List[Dict]:
        """Process XBRL financial data into structured format."""
        try:
            processed_facts = []
            
            # Handle different XBRL data structures
            if "facts" in xbrl_data:
                facts = xbrl_data["facts"]
            elif "us-gaap" in xbrl_data:
                facts = xbrl_data["us-gaap"]
            else:
                facts = xbrl_data
            
            for concept, concept_data in facts.items():
                if isinstance(concept_data, dict):
                    for unit_key, unit_data in concept_data.get("units", {}).items():
                        if isinstance(unit_data, list):
                            for fact in unit_data:
                                processed_fact = {
                                    "concept": concept,
                                    "value": fact.get("val"),
                                    "unit": unit_key,
                                    "period": fact.get("period", {}).get("end"),
                                    "frame": fact.get("frame"),
                                    "form": fact.get("form"),
                                    "filed": fact.get("filed")
                                }
                                processed_facts.append(processed_fact)
            
            logger.info(f"Processed {len(processed_facts)} XBRL facts")
            return processed_facts
            
        except Exception as e:
            logger.error(f"XBRL processing failed: {e}")
            return []
    
    def extract_financial_metrics(self, content: str) -> Dict[str, List[str]]:
        """Extract financial metrics and numbers from text content."""
        metrics = {
            "revenue": [],
            "profit": [],
            "assets": [],
            "liabilities": [],
            "cash": [],
            "debt": [],
            "shares": []
        }
        
        # Patterns for different financial metrics
        patterns = {
            "revenue": [
                r"revenue[s]?\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?",
                r"net\s*sales\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?",
                r"total\s*revenue\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?"
            ],
            "profit": [
                r"net\s*income\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?",
                r"profit\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?",
                r"earnings\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?"
            ],
            "assets": [
                r"total\s*assets\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?",
                r"assets\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?"
            ],
            "cash": [
                r"cash\s*and\s*cash\s*equivalents\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?",
                r"cash\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?"
            ]
        }
        
        for metric_type, metric_patterns in patterns.items():
            for pattern in metric_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                metrics[metric_type].extend(matches)
        
        return metrics
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get document processor statistics."""
        return {
            "max_chunk_size": self.max_chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "section_patterns": len(self.section_patterns),
            "supported_sections": list(self.section_patterns.keys())
        }
