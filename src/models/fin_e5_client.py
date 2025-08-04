"""
Financial E5 Client for Local Document Embeddings
Handles local embedding generation using fine-tuned E5 models for financial documents.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from datetime import datetime

logger = logging.getLogger(__name__)


class FinE5Client:
    """Client for local Financial E5 embeddings."""
    
    def __init__(self, model_path: str = None, device: str = None, 
                 max_length: int = 512):
        """Initialize Financial E5 client with local model."""
        self.model_path = model_path or os.getenv("FIN_E5_MODEL_PATH", "intfloat/e5-large-v2")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        # Model and tokenizer
        self.tokenizer = None
        self.model = None
        self.embedding_dim = 1024  # E5-large dimension
        
        # Usage tracking
        self.embed_count = 0
        self.total_tokens_processed = 0
        
        self._load_model()
        
        logger.info(f"Initialized FinE5 client with model: {self.model_path}")
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load the tokenizer and model."""
        try:
            logger.info(f"Loading model: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Get actual embedding dimension
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt", 
                                          truncation=True, max_length=self.max_length)
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
                test_output = self.model(**test_input)
                self.embedding_dim = test_output.last_hidden_state.shape[-1]
            
            logger.info(f"Model loaded successfully, embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on model output."""
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for financial document embedding."""
        if not text:
            return ""
        
        # Add instruction prefix for E5 models (improves performance)
        # For financial documents, we use a specific instruction
        instruction = "passage: "
        
        # Clean and prepare text
        text = text.strip()
        
        # Truncate if too long for tokenizer
        if len(text) > self.max_length * 4:  # Rough character to token ratio
            text = text[:self.max_length * 4]
        
        return instruction + text
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Tokenize
            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Convert to list
                embedding = embeddings.cpu().numpy()[0].tolist()
            
            # Update usage statistics
            self.embed_count += 1
            self.total_tokens_processed += inputs['input_ids'].shape[1]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts in batches."""
        if not texts:
            return []
        
        # Filter out empty texts and keep track of original positions
        text_mapping = []
        valid_texts = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                text_mapping.append((i, len(valid_texts)))
                valid_texts.append(text.strip())
            else:
                text_mapping.append((i, None))
        
        if not valid_texts:
            logger.warning("No valid texts provided for batch embedding")
            return [None] * len(texts)
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            
            logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch_texts)} texts")
            
            try:
                # Preprocess batch
                processed_texts = [self._preprocess_text(text) for text in batch_texts]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    processed_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=self.max_length
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                    
                    # Normalize embeddings
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    
                    # Convert to list
                    batch_embeddings = embeddings.cpu().numpy().tolist()
                
                all_embeddings.extend(batch_embeddings)
                
                # Update usage statistics
                self.embed_count += len(batch_texts)
                self.total_tokens_processed += inputs['input_ids'].shape[0] * inputs['input_ids'].shape[1]
                
            except Exception as e:
                logger.error(f"Batch embedding failed for batch starting at index {i}: {e}")
                # Add None for each text in failed batch
                all_embeddings.extend([None] * len(batch_texts))
        
        # Map embeddings back to original positions
        result = [None] * len(texts)
        valid_embedding_idx = 0
        
        for original_idx, valid_idx in text_mapping:
            if valid_idx is not None and valid_embedding_idx < len(all_embeddings):
                result[original_idx] = all_embeddings[valid_embedding_idx]
                valid_embedding_idx += 1
        
        successful_count = len([e for e in result if e is not None])
        logger.info(f"Batch embedding completed: {successful_count}/{len(texts)} successful")
        
        return result
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for this model."""
        return self.embedding_dim
    
    def get_usage_stats(self) -> Dict[str, any]:
        """Get current usage statistics."""
        return {
            "embed_count": self.embed_count,
            "total_tokens_processed": self.total_tokens_processed,
            "model_path": self.model_path,
            "device": self.device,
            "embedding_dimension": self.embedding_dim,
            "max_length": self.max_length
        }
    
    def health_check(self) -> Dict[str, any]:
        """Perform health check of the model."""
        try:
            # Test with a financial text sample
            test_text = "The company reported revenue of $1.2 billion for Q3 2023."
            test_embedding = self.embed_text(test_text)
            
            if test_embedding and len(test_embedding) == self.embedding_dim:
                return {
                    "status": "healthy",
                    "model_loaded": True,
                    "model_path": self.model_path,
                    "device": self.device,
                    "embedding_dimension": len(test_embedding),
                    "usage_stats": self.get_usage_stats(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Failed to generate test embedding or wrong dimension",
                    "model_path": self.model_path,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_path": self.model_path,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get detailed model information."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "embedding_dimension": self.embedding_dim,
            "max_length": self.max_length,
            "model_type": "E5-large-v2",
            "supports_financial_text": True,
            "local_model": True
        }
    
    def clear_cache(self):
        """Clear GPU memory cache if using CUDA."""
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    
    def close(self):
        """Clean up resources."""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        self.clear_cache()
        logger.info("FinE5 client closed")


# Context manager support
class FinE5Client(FinE5Client):
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
