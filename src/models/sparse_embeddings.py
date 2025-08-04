"""
Sparse Embeddings Client
Handles BOW, TF-IDF, and other sparse embedding techniques for SEC documents.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Union, Set
from collections import Counter
import re
import json
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class SparseEmbeddingsClient:
    """Client for generating sparse embeddings using BOW and TF-IDF."""
    
    def __init__(self, max_features: int = 10000, min_df: int = 2, 
                 max_df: float = 0.95, ngram_range: tuple = (1, 2)):
        """Initialize sparse embeddings client."""
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        
        # Vocabulary and document frequency tracking
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.document_frequencies_ = {}
        self.total_documents = 0
        self.is_fitted = False
        
        # Financial domain-specific terms
        self.financial_terms = {
            "revenue", "income", "profit", "loss", "assets", "liabilities",
            "equity", "cash", "debt", "expenses", "costs", "earnings",
            "dividend", "stock", "shares", "securities", "investment",
            "capital", "financing", "operations", "margin", "ratio",
            "growth", "performance", "risk", "compliance", "regulatory",
            "market", "competition", "strategy", "acquisition", "merger",
            "segment", "product", "service", "customer", "technology"
        }
        
        # Stop words for financial documents
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "up", "about", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "among", "within", "without", "under", "over",
            "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "must", "shall", "this",
            "that", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "me", "him", "her", "us", "them"
        }
        
        # Usage tracking
        self.embed_count = 0
        self.fit_count = 0
        
        logger.info(f"Initialized SparseEmbeddings with max_features={max_features}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sparse embedding."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        processed_text = self._preprocess_text(text)
        return processed_text.split()
    
    def _generate_ngrams(self, tokens: List[str]) -> List[str]:
        """Generate n-grams from tokens."""
        ngrams = []
        
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngram = " ".join(tokens[i:i + n])
                ngrams.append(ngram)
        
        return ngrams
    
    def _extract_features(self, text: str) -> List[str]:
        """Extract features (tokens and n-grams) from text."""
        tokens = self._tokenize(text)
        
        # Filter out stop words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Generate n-grams
        features = self._generate_ngrams(tokens)
        
        return features
    
    def fit(self, documents: List[str]) -> 'SparseEmbeddingsClient':
        """Fit the sparse embeddings model on a collection of documents."""
        logger.info(f"Fitting sparse embeddings on {len(documents)} documents")
        
        # Reset state
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.document_frequencies_ = {}
        self.total_documents = len(documents)
        
        # Count feature occurrences across documents
        feature_doc_counts = Counter()
        all_features = set()
        
        for doc_idx, document in enumerate(documents):
            if doc_idx % 100 == 0:
                logger.debug(f"Processing document {doc_idx + 1}/{len(documents)}")
            
            features = self._extract_features(document)
            doc_features = set(features)  # Unique features in this document
            
            all_features.update(features)
            feature_doc_counts.update(doc_features)
        
        # Filter features by document frequency
        min_doc_count = max(1, int(self.min_df)) if isinstance(self.min_df, int) else max(1, int(self.min_df * len(documents)))
        max_doc_count = int(self.max_df * len(documents)) if isinstance(self.max_df, float) else self.max_df
        
        filtered_features = []
        for feature, doc_count in feature_doc_counts.items():
            if min_doc_count <= doc_count <= max_doc_count:
                filtered_features.append((feature, doc_count))
        
        # Sort by document frequency and take top features
        filtered_features.sort(key=lambda x: x[1], reverse=True)
        
        # Prioritize financial terms
        financial_features = [(f, c) for f, c in filtered_features if any(term in f for term in self.financial_terms)]
        other_features = [(f, c) for f, c in filtered_features if not any(term in f for term in self.financial_terms)]
        
        # Combine with financial terms first
        prioritized_features = financial_features + other_features
        selected_features = prioritized_features[:self.max_features]
        
        # Build vocabulary
        self.feature_names_ = [feature for feature, _ in selected_features]
        self.vocabulary_ = {feature: idx for idx, (feature, _) in enumerate(selected_features)}
        self.document_frequencies_ = {feature: count for feature, count in selected_features}
        
        self.is_fitted = True
        self.fit_count += 1
        
        logger.info(f"Fitted sparse embeddings: {len(self.vocabulary_)} features selected")
        logger.info(f"Financial terms included: {len(financial_features)}")
        
        return self
    
    def _bow_transform(self, text: str) -> List[float]:
        """Transform text to bag-of-words representation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        
        features = self._extract_features(text)
        feature_counts = Counter(features)
        
        # Create BOW vector
        bow_vector = [0.0] * len(self.vocabulary_)
        
        for feature, count in feature_counts.items():
            if feature in self.vocabulary_:
                idx = self.vocabulary_[feature]
                bow_vector[idx] = float(count)
        
        return bow_vector
    
    def _tfidf_transform(self, text: str) -> List[float]:
        """Transform text to TF-IDF representation."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        
        # Get BOW representation first
        bow_vector = self._bow_transform(text)
        
        # Calculate document length for normalization
        doc_length = sum(bow_vector)
        if doc_length == 0:
            return bow_vector
        
        # Convert to TF-IDF
        tfidf_vector = []
        for idx, tf in enumerate(bow_vector):
            if tf > 0:
                # Term frequency (normalized)
                tf_normalized = tf / doc_length
                
                # Inverse document frequency
                feature = self.feature_names_[idx]
                df = self.document_frequencies_[feature]
                idf = math.log(self.total_documents / df)
                
                # TF-IDF score
                tfidf_score = tf_normalized * idf
                tfidf_vector.append(tfidf_score)
            else:
                tfidf_vector.append(0.0)
        
        return tfidf_vector
    
    def embed_text(self, text: str, method: str = "tfidf") -> Optional[List[float]]:
        """Generate sparse embedding for a single text."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            if method == "bow":
                embedding = self._bow_transform(text)
            elif method == "tfidf":
                embedding = self._tfidf_transform(text)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            self.embed_count += 1
            return embedding
            
        except Exception as e:
            logger.error(f"Sparse embedding failed: {e}")
            return None
    
    def embed_batch(self, texts: List[str], method: str = "tfidf") -> List[Optional[List[float]]]:
        """Generate sparse embeddings for multiple texts."""
        if not texts:
            return []
        
        embeddings = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.debug(f"Processing text {i + 1}/{len(texts)}")
            
            embedding = self.embed_text(text, method)
            embeddings.append(embedding)
        
        successful_count = len([e for e in embeddings if e is not None])
        logger.info(f"Sparse batch embedding completed: {successful_count}/{len(texts)} successful")
        
        return embeddings
    
    def get_feature_importance(self, text: str, top_k: int = 10) -> List[Dict[str, any]]:
        """Get the most important features for a given text."""
        if not self.is_fitted:
            return []
        
        tfidf_vector = self._tfidf_transform(text)
        
        # Get top features by TF-IDF score
        feature_scores = []
        for idx, score in enumerate(tfidf_vector):
            if score > 0:
                feature_scores.append({
                    "feature": self.feature_names_[idx],
                    "tfidf_score": score,
                    "document_frequency": self.document_frequencies_[self.feature_names_[idx]]
                })
        
        # Sort by TF-IDF score
        feature_scores.sort(key=lambda x: x["tfidf_score"], reverse=True)
        
        return feature_scores[:top_k]
    
    def get_vocabulary_stats(self) -> Dict[str, any]:
        """Get statistics about the fitted vocabulary."""
        if not self.is_fitted:
            return {}
        
        # Analyze vocabulary composition
        unigrams = [f for f in self.feature_names_ if " " not in f]
        bigrams = [f for f in self.feature_names_ if f.count(" ") == 1]
        higher_grams = [f for f in self.feature_names_ if f.count(" ") > 1]
        
        financial_terms_count = sum(1 for f in self.feature_names_ 
                                  if any(term in f for term in self.financial_terms))
        
        return {
            "total_features": len(self.feature_names_),
            "unigrams": len(unigrams),
            "bigrams": len(bigrams),
            "higher_ngrams": len(higher_grams),
            "financial_terms": financial_terms_count,
            "total_documents_fitted": self.total_documents,
            "avg_doc_frequency": np.mean(list(self.document_frequencies_.values())),
            "max_doc_frequency": max(self.document_frequencies_.values()),
            "min_doc_frequency": min(self.document_frequencies_.values())
        }
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension (vocabulary size)."""
        return len(self.vocabulary_) if self.is_fitted else 0
    
    def get_usage_stats(self) -> Dict[str, any]:
        """Get current usage statistics."""
        return {
            "embed_count": self.embed_count,
            "fit_count": self.fit_count,
            "is_fitted": self.is_fitted,
            "vocabulary_size": len(self.vocabulary_),
            "max_features": self.max_features,
            "ngram_range": self.ngram_range
        }
    
    def health_check(self) -> Dict[str, any]:
        """Perform health check of sparse embeddings client."""
        try:
            if not self.is_fitted:
                return {
                    "status": "unfitted",
                    "message": "Model needs to be fitted on documents first",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Test with financial text
            test_text = "The company reported revenue growth and improved profit margins."
            test_embedding = self.embed_text(test_text)
            
            if test_embedding and len(test_embedding) == len(self.vocabulary_):
                return {
                    "status": "healthy",
                    "vocabulary_size": len(self.vocabulary_),
                    "embedding_dimension": len(test_embedding),
                    "vocab_stats": self.get_vocabulary_stats(),
                    "usage_stats": self.get_usage_stats(),
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
            "model_type": "Sparse Embeddings (BOW/TF-IDF)",
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "ngram_range": self.ngram_range,
            "financial_terms_count": len(self.financial_terms),
            "stop_words_count": len(self.stop_words),
            "embedding_dimension": self.get_embedding_dimension(),
            "supports_methods": ["bow", "tfidf"],
            "local_model": True
        }
    
    def save_vocabulary(self, filepath: str):
        """Save fitted vocabulary to file."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        vocab_data = {
            "vocabulary_": self.vocabulary_,
            "feature_names_": self.feature_names_,
            "document_frequencies_": self.document_frequencies_,
            "total_documents": self.total_documents,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df": self.max_df,
            "ngram_range": self.ngram_range
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str):
        """Load fitted vocabulary from file."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.vocabulary_ = vocab_data["vocabulary_"]
        self.feature_names_ = vocab_data["feature_names_"]
        self.document_frequencies_ = vocab_data["document_frequencies_"]
        self.total_documents = vocab_data["total_documents"]
        self.max_features = vocab_data["max_features"]
        self.min_df = vocab_data["min_df"]
        self.max_df = vocab_data["max_df"]
        self.ngram_range = tuple(vocab_data["ngram_range"])
        
        self.is_fitted = True
        
        logger.info(f"Vocabulary loaded from {filepath}")


# Helper functions
def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def get_financial_keywords() -> Set[str]:
    """Get comprehensive set of financial keywords."""
    return {
        # Financial statements
        "revenue", "income", "profit", "loss", "earnings", "sales",
        "assets", "liabilities", "equity", "capital", "retained",
        "cash", "debt", "payable", "receivable", "inventory",
        
        # Financial metrics
        "margin", "ratio", "return", "yield", "growth", "performance",
        "efficiency", "liquidity", "solvency", "profitability",
        
        # Business terms
        "market", "competition", "strategy", "operations", "segment",
        "product", "service", "customer", "technology", "innovation",
        
        # Regulatory
        "compliance", "regulatory", "sec", "gaap", "ifrs", "sox",
        "risk", "control", "audit", "governance", "disclosure",
        
        # Corporate actions
        "acquisition", "merger", "divestiture", "spinoff", "dividend",
        "stock", "shares", "securities", "investment", "financing"
    }
