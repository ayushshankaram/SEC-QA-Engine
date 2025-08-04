"""
Embedding Ensemble Framework
Combines multiple embedding models for comprehensive document understanding.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime
import json

# Import individual embedding clients
from src.models.voyage_client import VoyageClient
from src.models.fin_e5_client import FinE5Client
from src.models.xbrl_client import XBRLClient
from src.models.sparse_embeddings import SparseEmbeddingsClient

# Import backup/alternative clients
try:
    from src.models.financial_bert import FinancialBertClient
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingEnsemble:
    """Ensemble framework combining multiple embedding models."""
    
    def __init__(self, weights: Dict[str, float] = None, 
                 enable_models: Dict[str, bool] = None):
        """Initialize embedding ensemble with model weights."""
        
        # Load weights from environment variables or use defaults
        if weights is None:
            weights = {
                "voyage": float(os.getenv("VOYAGE_WEIGHT", 0.0)),
                "fin_e5": float(os.getenv("FINE5_WEIGHT", 0.5)),
                "xbrl": float(os.getenv("XBRL_WEIGHT", 0.3)),
                "sparse": float(os.getenv("SPARSE_WEIGHT", 0.2))
            }
        
        # Validate weights sum to 1.0
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Ensemble weights sum to {weight_sum:.3f}, normalizing to 1.0")
            weights = {k: v/weight_sum for k, v in weights.items()}
        
        self.weights = weights
        
        # Model enablement (for testing/debugging)
        self.enable_models = enable_models or {
            "voyage": False,    # Disabled due to rate limits
            "fin_e5": True,
            "xbrl": True,
            "sparse": False     # Disabled for faster testing
        }
        
        # Initialize embedding clients
        self.clients = {}
        self.model_dimensions = {}
        
        # Usage tracking
        self.embed_count = 0
        self.model_failures = {model: 0 for model in self.weights.keys()}
        
        self._initialize_clients()
        
        logger.info("Initialized EmbeddingEnsemble with 4 models")
        logger.info(f"Model weights: {self.weights}")
    
    def _initialize_clients(self):
        """Initialize all embedding clients."""
        
        # Initialize Voyage client with fallback
        if self.enable_models["voyage"]:
            try:
                self.clients["voyage"] = VoyageClient()
                self.model_dimensions["voyage"] = self.clients["voyage"].get_embedding_dimension()
                logger.info("Voyage client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Voyage client: {e}")
                self.enable_models["voyage"] = False
                
                # Try to use Financial BERT as fallback
                if FINBERT_AVAILABLE:
                    try:
                        logger.info("Attempting to use Financial BERT as Voyage fallback")
                        self.clients["voyage"] = FinancialBertClient()
                        self.model_dimensions["voyage"] = self.clients["voyage"].get_embedding_dimension()
                        logger.info("Financial BERT initialized as Voyage fallback")
                        self.enable_models["voyage"] = True
                    except Exception as fallback_e:
                        logger.error(f"Failed to initialize Financial BERT fallback: {fallback_e}")
                        self.enable_models["voyage"] = False
        
        # Initialize FinE5 client
        if self.enable_models["fin_e5"]:
            try:
                self.clients["fin_e5"] = FinE5Client()
                self.model_dimensions["fin_e5"] = self.clients["fin_e5"].get_embedding_dimension()
                logger.info("FinE5 client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize FinE5 client: {e}")
                self.enable_models["fin_e5"] = False
        
        # Initialize XBRL client
        if self.enable_models["xbrl"]:
            try:
                self.clients["xbrl"] = XBRLClient()
                self.model_dimensions["xbrl"] = self.clients["xbrl"].get_embedding_dimension()
                logger.info("XBRL client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize XBRL client: {e}")
                self.enable_models["xbrl"] = False
        
        # Initialize Sparse client (needs fitting first)
        if self.enable_models["sparse"]:
            try:
                self.clients["sparse"] = SparseEmbeddingsClient()
                logger.info("Sparse embeddings client initialized (requires fitting)")
            except Exception as e:
                logger.error(f"Failed to initialize Sparse client: {e}")
                self.enable_models["sparse"] = False
        
        # Log active models and weights
        active_models = [model for model, enabled in self.enable_models.items() if enabled]
        logger.info(f"Active embedding models: {active_models}")
        logger.info(f"Initialized EmbeddingEnsemble with {len(active_models)} models")
        logger.info(f"Model weights: {self.weights}")
    
    def fit_sparse_model(self, documents: List[str]):
        """Fit the sparse embeddings model on a collection of documents."""
        if not self.enable_models["sparse"] or "sparse" not in self.clients:
            logger.warning("Sparse model not available for fitting")
            return
        
        logger.info(f"Fitting sparse model on {len(documents)} documents")
        
        try:
            self.clients["sparse"].fit(documents)
            self.model_dimensions["sparse"] = self.clients["sparse"].get_embedding_dimension()
            logger.info(f"Sparse model fitted with {self.model_dimensions['sparse']} features")
        except Exception as e:
            logger.error(f"Failed to fit sparse model: {e}")
            self.enable_models["sparse"] = False
    
    def _normalize_embeddings(self, embeddings: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Normalize embeddings to unit vectors for ensemble combination."""
        normalized = {}
        
        for model_name, embedding in embeddings.items():
            if embedding is not None:
                # Convert to numpy for easier computation
                emb_array = np.array(embedding)
                
                # L2 normalization
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    normalized[model_name] = (emb_array / norm).tolist()
                else:
                    normalized[model_name] = embedding
            else:
                normalized[model_name] = None
        
        return normalized
    
    def _combine_embeddings(self, embeddings: Dict[str, List[float]]) -> Optional[List[float]]:
        """Combine embeddings from multiple models using weighted average."""
        
        # Filter out None embeddings
        valid_embeddings = {k: v for k, v in embeddings.items() if v is not None}
        
        if not valid_embeddings:
            logger.warning("No valid embeddings to combine")
            return None
        
        # Normalize embeddings
        normalized_embeddings = self._normalize_embeddings(valid_embeddings)
        
        # Find the maximum dimension to standardize sizes
        max_dim = max(len(emb) for emb in normalized_embeddings.values() if emb is not None)
        
        # Pad embeddings to same size
        padded_embeddings = {}
        for model_name, embedding in normalized_embeddings.items():
            if embedding is not None:
                if len(embedding) < max_dim:
                    # Pad with zeros
                    padded = embedding + [0.0] * (max_dim - len(embedding))
                    padded_embeddings[model_name] = padded
                else:
                    padded_embeddings[model_name] = embedding[:max_dim]
        
        # Weighted combination
        combined = np.zeros(max_dim)
        total_weight = 0.0
        
        for model_name, embedding in padded_embeddings.items():
            if embedding is not None:
                weight = self.weights.get(model_name, 0.0)
                combined += weight * np.array(embedding)
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined = combined / total_weight
        
        return combined.tolist()
    
    def _embed_batch(self, texts: List[str], 
                    include_xbrl_facts: List[Dict] = None) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts using all models."""
        
        if not texts:
            return []
        
        logger.debug(f"Generating ensemble embeddings for {len(texts)} texts")
        
        # Collect embeddings from all models
        all_model_embeddings = {}
        
        # Voyage embeddings
        if self.enable_models["voyage"] and "voyage" in self.clients:
            try:
                voyage_embeddings = self.clients["voyage"].embed_batch(texts)
                all_model_embeddings["voyage"] = voyage_embeddings
                logger.debug(f"Voyage embeddings: {len([e for e in voyage_embeddings if e])}/{len(texts)}")
            except Exception as e:
                logger.error(f"Voyage batch embedding failed: {e}")
                self.model_failures["voyage"] += 1
                all_model_embeddings["voyage"] = [None] * len(texts)
        
        # FinE5 embeddings
        if self.enable_models["fin_e5"] and "fin_e5" in self.clients:
            try:
                fin_e5_embeddings = self.clients["fin_e5"].embed_batch(texts)
                all_model_embeddings["fin_e5"] = fin_e5_embeddings
                logger.debug(f"FinE5 embeddings: {len([e for e in fin_e5_embeddings if e])}/{len(texts)}")
            except Exception as e:
                logger.error(f"FinE5 batch embedding failed: {e}")
                self.model_failures["fin_e5"] += 1
                all_model_embeddings["fin_e5"] = [None] * len(texts)
        
        # XBRL embeddings
        if self.enable_models["xbrl"] and "xbrl" in self.clients:
            try:
                if include_xbrl_facts:
                    # Use XBRL facts if provided
                    xbrl_embeddings = []
                    for i, text in enumerate(texts):
                        if i < len(include_xbrl_facts) and include_xbrl_facts[i]:
                            emb = self.clients["xbrl"].embed_xbrl_facts(include_xbrl_facts[i])
                        else:
                            emb = self.clients["xbrl"].embed_text(text)
                        xbrl_embeddings.append(emb)
                else:
                    # Use text-based XBRL embeddings
                    xbrl_embeddings = []
                    for text in texts:
                        emb = self.clients["xbrl"].embed_text(text)
                        xbrl_embeddings.append(emb)
                
                all_model_embeddings["xbrl"] = xbrl_embeddings
                logger.debug(f"XBRL embeddings: {len([e for e in xbrl_embeddings if e])}/{len(texts)}")
            except Exception as e:
                logger.error(f"XBRL batch embedding failed: {e}")
                self.model_failures["xbrl"] += 1
                all_model_embeddings["xbrl"] = [None] * len(texts)
        
        # Sparse embeddings
        if (self.enable_models["sparse"] and "sparse" in self.clients and 
            self.clients["sparse"].is_fitted):
            try:
                sparse_embeddings = self.clients["sparse"].embed_batch(texts)
                all_model_embeddings["sparse"] = sparse_embeddings
                logger.debug(f"Sparse embeddings: {len([e for e in sparse_embeddings if e])}/{len(texts)}")
            except Exception as e:
                logger.error(f"Sparse batch embedding failed: {e}")
                self.model_failures["sparse"] += 1
                all_model_embeddings["sparse"] = [None] * len(texts)
        
        # Combine embeddings for each text
        combined_embeddings = []
        for i in range(len(texts)):
            text_embeddings = {}
            for model_name, model_embeddings in all_model_embeddings.items():
                logger.debug(f"Processing {model_name}: type={type(model_embeddings)}, value={model_embeddings}")
                if isinstance(model_embeddings, list) and i < len(model_embeddings):
                    text_embeddings[model_name] = model_embeddings[i]
                else:
                    logger.warning(f"Invalid embeddings from {model_name}: {type(model_embeddings)} = {model_embeddings}")
                    text_embeddings[model_name] = None
            
            combined = self._combine_embeddings(text_embeddings)
            combined_embeddings.append(combined)
        
        self.embed_count += len(texts)
        
        successful_count = len([e for e in combined_embeddings if e is not None])
        logger.info(f"Ensemble batch embedding completed: {successful_count}/{len(texts)} successful")
        
        return combined_embeddings
    
    def embed_text(self, text: str, xbrl_facts: List[Dict] = None) -> Optional[List[float]]:
        """Generate ensemble embedding for a single text."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        embeddings = self._embed_batch([text], 
                                     include_xbrl_facts=[xbrl_facts] if xbrl_facts else None)
        return embeddings[0] if embeddings else None
    
    def embed_query(self, query: str) -> Optional[List[float]]:
        """Generate ensemble embedding for a query (alias for embed_text)."""
        return self.embed_text(query)
    
    def embed_batch(self, texts: List[str], 
                   batch_size: int = 10,
                   xbrl_facts_list: List[List[Dict]] = None) -> List[Optional[List[float]]]:
        """Generate ensemble embeddings for multiple texts in batches."""
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            batch_xbrl_facts = None
            if xbrl_facts_list:
                batch_xbrl_facts = xbrl_facts_list[i:i + batch_size]
            
            logger.debug(f"Processing ensemble batch {i // batch_size + 1}: {len(batch_texts)} texts")
            
            batch_embeddings = self._embed_batch(batch_texts, batch_xbrl_facts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def get_model_health(self) -> Dict[str, any]:
        """Get health status of all embedding models."""
        health_status = {}
        
        for model_name in self.weights.keys():
            if self.enable_models[model_name] and model_name in self.clients:
                try:
                    health_status[model_name] = self.clients[model_name].health_check()
                except Exception as e:
                    health_status[model_name] = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                health_status[model_name] = {
                    "status": "disabled",
                    "enabled": self.enable_models[model_name],
                    "timestamp": datetime.now().isoformat()
                }
        
        return health_status
    
    def get_usage_stats(self) -> Dict[str, any]:
        """Get comprehensive usage statistics."""
        model_stats = {}
        
        for model_name in self.weights.keys():
            if self.enable_models[model_name] and model_name in self.clients:
                try:
                    model_stats[model_name] = self.clients[model_name].get_usage_stats()
                except Exception as e:
                    model_stats[model_name] = {"error": str(e)}
            else:
                model_stats[model_name] = {"status": "disabled"}
        
        return {
            "ensemble_embed_count": self.embed_count,
            "model_failures": self.model_failures,
            "active_models": [m for m, e in self.enable_models.items() if e],
            "model_weights": self.weights,
            "model_dimensions": self.model_dimensions,
            "individual_models": model_stats
        }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get detailed information about all models in the ensemble."""
        model_info = {}
        
        for model_name in self.weights.keys():
            if self.enable_models[model_name] and model_name in self.clients:
                try:
                    model_info[model_name] = self.clients[model_name].get_model_info()
                except Exception as e:
                    model_info[model_name] = {"error": str(e)}
            else:
                model_info[model_name] = {"status": "disabled"}
        
        return {
            "ensemble_type": "Weighted Average Combination",
            "total_models": len(self.weights),
            "active_models": len([m for m, e in self.enable_models.items() if e]),
            "model_weights": self.weights,
            "normalization": "L2 normalization",
            "combination_method": "weighted_average",
            "models": model_info
        }
    
    def health_check(self) -> Dict[str, any]:
        """Perform comprehensive health check of the ensemble."""
        try:
            model_health = self.get_model_health()
            healthy_models = [name for name, status in model_health.items() 
                            if status.get("status") == "healthy"]
            
            # Test with sample text
            test_text = "The company reported strong revenue growth of 15% year-over-year."
            test_embedding = self.embed_text(test_text)
            
            return {
                "status": "healthy" if test_embedding else "degraded",
                "active_models": len(healthy_models),
                "total_models": len(self.weights),
                "healthy_models": healthy_models,
                "test_embedding_generated": test_embedding is not None,
                "embedding_dimension": len(test_embedding) if test_embedding else 0,
                "model_health": model_health,
                "usage_stats": self.get_usage_stats(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def close(self):
        """Close all embedding clients and cleanup resources."""
        for model_name, client in self.clients.items():
            try:
                if hasattr(client, 'close'):
                    client.close()
                logger.debug(f"Closed {model_name} client")
            except Exception as e:
                logger.warning(f"Error closing {model_name} client: {e}")
        
        logger.info("EmbeddingEnsemble closed")


# Context manager support
class EmbeddingEnsemble(EmbeddingEnsemble):
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Helper functions
def create_default_ensemble() -> EmbeddingEnsemble:
    """Create ensemble with default configuration."""
    return EmbeddingEnsemble()


def create_testing_ensemble(enable_only: List[str]) -> EmbeddingEnsemble:
    """Create ensemble with only specific models enabled (for testing)."""
    enable_models = {model: model in enable_only for model in ["voyage", "fin_e5", "xbrl", "sparse"]}
    return EmbeddingEnsemble(enable_models=enable_models)
