"""
XBRL Embedding Client
Handles embedding generation for XBRL financial data using structured approach.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, date
import json
import math

logger = logging.getLogger(__name__)


class XBRLClient:
    """Client for generating embeddings from XBRL financial data."""
    
    def __init__(self, embedding_dim: int = None):
        """Initialize XBRL client."""
        self.embedding_dim = embedding_dim or int(os.getenv("XBRL_EMBEDDING_DIM", 512))
        
        # Financial concept categories for structured embedding
        self.concept_categories = {
            "revenue": [
                "revenues", "revenue", "netsales", "operatingrevenues",
                "totalrevenues", "salesrevenuenet", "revenuefromcontractwithcustomerexcludingassessedtax"
            ],
            "expenses": [
                "operatingexpenses", "costofrevenue", "costofgoodssold", 
                "researchanddevelopmentexpense", "sellingandmarketingexpense",
                "generalandadministrativeexpense"
            ],
            "income": [
                "netincome", "netincomeloss", "profitloss", "incomelossfromcontinuingoperations",
                "comprehensiveincomeloss", "earningsper", "dilutedearningsper"
            ],
            "assets": [
                "assets", "totalassets", "currentassets", "noncurrentassets",
                "cashandcashequivalents", "shortterminvestments", "accountsreceivable",
                "inventory", "propertyplantandequipment", "intangibleassets", "goodwill"
            ],
            "liabilities": [
                "liabilities", "totalliabilities", "currentliabilities", "noncurrentliabilities",
                "accountspayable", "accruedliabilities", "debt", "longtermldebt", "shorttermldebt"
            ],
            "equity": [
                "stockholdersequity", "shareholdersequity", "totalequity", "retainedearnings",
                "commonstock", "preferredstock", "treasurystock", "paidincapital"
            ],
            "cashflow": [
                "cashflowfromoperatingactivities", "cashflowfrominvestingactivities",
                "cashflowfromfinancingactivities", "netcashflow", "freecashflow"
            ],
            "ratios": [
                "returnonassets", "returnonequity", "grossmargin", "operatingmargin",
                "netmargin", "debttoequity", "currentratio", "quickratio"
            ]
        }
        
        # US-GAAP and DEI taxonomy support
        self.supported_taxonomies = {
            "us-gaap": "US Generally Accepted Accounting Principles",
            "dei": "Document and Entity Information",
            "invest": "Investment Company Act",
            "currency": "Currency",
            "exch": "Exchange",
            "country": "Country"
        }
        
        # Temporal context weights
        self.temporal_weights = {
            "instant": 1.0,
            "duration": 0.8,
            "start": 0.6,
            "end": 0.9
        }
        
        # Usage tracking
        self.embed_count = 0
        self.facts_processed = 0
        
        logger.info(f"Initialized XBRL client with embedding dimension: {self.embedding_dim}")
    
    def _categorize_concept(self, concept: str) -> str:
        """Categorize a financial concept."""
        concept_lower = concept.lower()
        
        for category, keywords in self.concept_categories.items():
            for keyword in keywords:
                if keyword in concept_lower:
                    return category
        
        return "other"
    
    def _normalize_value(self, value: Union[float, int, str]) -> float:
        """Normalize financial values using logarithmic scaling."""
        try:
            if isinstance(value, str):
                # Remove common formatting
                value = value.replace(",", "").replace("$", "").replace("(", "-").replace(")", "")
                value = float(value)
            
            if value == 0:
                return 0.0
            
            # Use logarithmic scaling for large financial values
            if abs(value) >= 1:
                return math.copysign(math.log10(abs(value)) / 10.0, value)
            else:
                return value
                
        except (ValueError, TypeError):
            logger.warning(f"Could not normalize value: {value}")
            return 0.0
    
    def _encode_period(self, period: str) -> List[float]:
        """Encode temporal period information."""
        encoding = [0.0] * 12  # 12 months + year info
        
        try:
            if not period:
                return encoding
            
            # Handle different period formats
            if "T" in period:  # ISO format
                date_part = period.split("T")[0]
            else:
                date_part = period
            
            if "-" in date_part:
                parts = date_part.split("-")
                if len(parts) >= 2:
                    year = int(parts[0])
                    month = int(parts[1])
                    
                    # Encode month (one-hot style)
                    if 1 <= month <= 12:
                        encoding[month - 1] = 1.0
                    
                    # Encode year trend (normalized)
                    year_normalized = (year - 2000) / 50.0  # Normalize around 2000-2050
                    if len(encoding) > 12:
                        encoding = encoding[:12] + [year_normalized]
                    else:
                        encoding.append(year_normalized)
        
        except Exception as e:
            logger.warning(f"Could not encode period {period}: {e}")
        
        return encoding[:13]  # Ensure fixed size
    
    def _create_concept_embedding(self, concept: str, value: float, 
                                unit: str, period: str) -> List[float]:
        """Create embedding for a single XBRL concept."""
        embedding = [0.0] * self.embedding_dim
        
        # Category encoding (first part of embedding)
        category = self._categorize_concept(concept)
        category_idx = list(self.concept_categories.keys()).index(category) if category in self.concept_categories else 0
        if category_idx < len(embedding):
            embedding[category_idx] = 1.0
        
        # Value encoding (normalized, multiple positions)
        normalized_value = self._normalize_value(value)
        value_positions = [10, 11, 12, 13, 14]  # Multiple positions for value
        for i, pos in enumerate(value_positions):
            if pos < len(embedding):
                embedding[pos] = normalized_value * (0.8 ** i)  # Diminishing representation
        
        # Unit encoding
        unit_hash = hash(unit.lower() if unit else "usd") % 20
        unit_pos = 20 + (unit_hash % 10)
        if unit_pos < len(embedding):
            embedding[unit_pos] = 0.5
        
        # Period encoding
        period_encoding = self._encode_period(period)
        period_start = 35
        for i, val in enumerate(period_encoding):
            pos = period_start + i
            if pos < len(embedding):
                embedding[pos] = val
        
        # Concept name features (character-level hash features)
        concept_features = self._extract_concept_features(concept)
        feature_start = 50
        for i, feature in enumerate(concept_features[:min(50, len(embedding) - feature_start)]):
            pos = feature_start + i
            if pos < len(embedding):
                embedding[pos] = feature
        
        return embedding
    
    def _extract_concept_features(self, concept: str) -> List[float]:
        """Extract features from concept name."""
        if not concept:
            return [0.0] * 50
        
        features = []
        concept_lower = concept.lower()
        
        # Length feature
        features.append(min(len(concept_lower) / 100.0, 1.0))
        
        # Character frequency features
        char_counts = {}
        for char in concept_lower:
            if char.isalpha():
                char_counts[char] = char_counts.get(char, 0) + 1
        
        # Top character frequencies (normalized)
        total_chars = len([c for c in concept_lower if c.isalpha()])
        if total_chars > 0:
            for char in "abcdefghijklmnopqrstuvwxyz":
                freq = char_counts.get(char, 0) / total_chars
                features.append(freq)
        else:
            features.extend([0.0] * 26)
        
        # N-gram features (bigrams)
        bigrams = [concept_lower[i:i+2] for i in range(len(concept_lower)-1)]
        common_bigrams = ["in", "on", "er", "an", "re", "ed", "nd", "ou", "ea", "ti"]
        for bigram in common_bigrams:
            count = bigrams.count(bigram)
            features.append(min(count / 5.0, 1.0))
        
        # Taxonomy indicators
        taxonomy_indicators = ["gaap", "dei", "us", "ifrs", "sec"]
        for indicator in taxonomy_indicators:
            features.append(1.0 if indicator in concept_lower else 0.0)
        
        # Ensure we return exactly 50 features
        return features[:50] + [0.0] * max(0, 50 - len(features))
    
    def embed_xbrl_fact(self, fact: Dict) -> Optional[List[float]]:
        """Generate embedding for a single XBRL fact."""
        try:
            concept = fact.get("concept", "")
            value = fact.get("value", 0)
            unit = fact.get("unit", "usd")
            period = fact.get("period", "")
            
            if not concept:
                logger.warning("XBRL fact missing concept name")
                return None
            
            embedding = self._create_concept_embedding(concept, value, unit, period)
            
            # Update usage statistics
            self.embed_count += 1
            self.facts_processed += 1
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed XBRL fact: {e}")
            return None
    
    def embed_xbrl_facts(self, facts: List[Dict]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple XBRL facts."""
        if not facts:
            return []
        
        embeddings = []
        for i, fact in enumerate(facts):
            logger.debug(f"Processing XBRL fact {i+1}/{len(facts)}")
            embedding = self.embed_xbrl_fact(fact)
            embeddings.append(embedding)
        
        successful_count = len([e for e in embeddings if e is not None])
        logger.info(f"XBRL batch embedding completed: {successful_count}/{len(facts)} successful")
        
        return embeddings
    
    def aggregate_facts_embedding(self, facts: List[Dict], 
                                aggregation_method: str = "mean") -> Optional[List[float]]:
        """Create aggregated embedding from multiple related XBRL facts."""
        embeddings = self.embed_xbrl_facts(facts)
        valid_embeddings = [e for e in embeddings if e is not None]
        
        if not valid_embeddings:
            return None
        
        if aggregation_method == "mean":
            # Average all embeddings
            aggregated = np.mean(valid_embeddings, axis=0).tolist()
        elif aggregation_method == "weighted_mean":
            # Weight by value magnitude
            weights = []
            for fact in facts:
                value = abs(self._normalize_value(fact.get("value", 0)))
                weights.append(value + 0.1)  # Add small constant to avoid zero weights
            
            weighted_embeddings = []
            for emb, weight in zip(valid_embeddings, weights):
                weighted_emb = [x * weight for x in emb]
                weighted_embeddings.append(weighted_emb)
            
            total_weight = sum(weights)
            aggregated = [sum(col) / total_weight for col in zip(*weighted_embeddings)]
        
        elif aggregation_method == "max":
            # Element-wise maximum
            aggregated = np.max(valid_embeddings, axis=0).tolist()
        else:
            # Default to mean
            aggregated = np.mean(valid_embeddings, axis=0).tolist()
        
        return aggregated
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text containing XBRL concept references."""
        try:
            # Extract potential XBRL concepts from text
            concepts = self._extract_concepts_from_text(text)
            
            if not concepts:
                # Create basic text embedding using concept matching
                return self._create_text_based_embedding(text)
            
            # Create embeddings for found concepts
            fact_embeddings = []
            for concept in concepts:
                # Create a mock fact for embedding
                mock_fact = {
                    "concept": concept,
                    "value": 1.0,  # Neutral value
                    "unit": "usd",
                    "period": ""
                }
                embedding = self.embed_xbrl_fact(mock_fact)
                if embedding:
                    fact_embeddings.append(embedding)
            
            if fact_embeddings:
                # Average the concept embeddings
                return np.mean(fact_embeddings, axis=0).tolist()
            else:
                return self._create_text_based_embedding(text)
                
        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return None
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """Extract potential XBRL concepts from text."""
        concepts = []
        text_lower = text.lower()
        
        # Look for known financial terms
        all_keywords = []
        for category_keywords in self.concept_categories.values():
            all_keywords.extend(category_keywords)
        
        for keyword in all_keywords:
            if keyword in text_lower:
                concepts.append(keyword)
        
        return list(set(concepts))  # Remove duplicates
    
    def _create_text_based_embedding(self, text: str) -> List[float]:
        """Create basic embedding for text without XBRL concepts."""
        embedding = [0.0] * self.embedding_dim
        
        if not text:
            return embedding
        
        text_lower = text.lower()
        
        # Simple term frequency features
        for i, (category, keywords) in enumerate(self.concept_categories.items()):
            category_score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    category_score += 1.0
            
            if i < len(embedding):
                embedding[i] = min(category_score / len(keywords), 1.0)
        
        # Text length feature
        length_pos = len(self.concept_categories)
        if length_pos < len(embedding):
            embedding[length_pos] = min(len(text) / 1000.0, 1.0)
        
        return embedding
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for XBRL embeddings."""
        return self.embedding_dim
    
    def get_usage_stats(self) -> Dict[str, any]:
        """Get current usage statistics."""
        return {
            "embed_count": self.embed_count,
            "facts_processed": self.facts_processed,
            "embedding_dimension": self.embedding_dim,
            "supported_taxonomies": list(self.supported_taxonomies.keys()),
            "concept_categories": len(self.concept_categories)
        }
    
    def health_check(self) -> Dict[str, any]:
        """Perform health check of XBRL client."""
        try:
            # Test with sample XBRL fact
            test_fact = {
                "concept": "Revenues",
                "value": 1000000,
                "unit": "USD",
                "period": "2023-12-31"
            }
            
            test_embedding = self.embed_xbrl_fact(test_fact)
            
            if test_embedding and len(test_embedding) == self.embedding_dim:
                return {
                    "status": "healthy",
                    "embedding_dimension": len(test_embedding),
                    "usage_stats": self.get_usage_stats(),
                    "supported_categories": list(self.concept_categories.keys()),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Failed to generate test embedding or wrong dimension",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get detailed model information."""
        return {
            "model_type": "XBRL Structured Embedding",
            "embedding_dimension": self.embedding_dim,
            "supported_taxonomies": self.supported_taxonomies,
            "concept_categories": list(self.concept_categories.keys()),
            "features": [
                "Category encoding",
                "Value normalization (logarithmic)",
                "Unit encoding",
                "Temporal encoding",
                "Concept name features"
            ],
            "local_model": True
        }


# Helper functions
def parse_xbrl_period(period_str: str) -> Dict[str, any]:
    """Parse XBRL period string into structured format."""
    try:
        if "T" in period_str:
            date_part = period_str.split("T")[0]
        else:
            date_part = period_str
        
        if "/" in date_part:  # Period range
            start, end = date_part.split("/")
            return {
                "type": "duration",
                "start": start,
                "end": end
            }
        else:  # Instant
            return {
                "type": "instant",
                "date": date_part
            }
    except Exception:
        return {"type": "unknown", "raw": period_str}


def normalize_concept_name(concept: str) -> str:
    """Normalize XBRL concept name."""
    if not concept:
        return ""
    
    # Remove common prefixes
    prefixes = ["us-gaap:", "dei:", "ifrs-full:", "gaap:", "us_gaap_"]
    for prefix in prefixes:
        if concept.lower().startswith(prefix):
            concept = concept[len(prefix):]
            break
    
    return concept.strip()
