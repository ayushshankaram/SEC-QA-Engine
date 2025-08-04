"""
Financial BERT Embeddings Client
Free alternative using Hugging Face transformers for financial document embeddings.
"""

import os
import logging
from typing import List, Dict, Optional, Union
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FinancialBertClient:
    """Client for financial BERT embeddings as a free alternative to Voyage."""
    
    def __init__(self, model_name: str = None):
        """Initialize Financial BERT embeddings client."""
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required but not available")
        
        # Use a financial domain BERT model
        self.model_name = model_name or "nlpaueb/sec-bert-base"  # SEC-BERT trained on SEC filings
        
        try:
            logger.info(f"Loading financial BERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Check if CUDA is available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            
            logger.info(f"Financial BERT model loaded successfully on {self.device}")
            logger.info(f"Model dimension: {self.get_embedding_dimension()}")
            
        except Exception as e:
            logger.error(f"Failed to load financial BERT model: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        return self.model.config.hidden_size
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_single(self, text: str) -> Optional[List[float]]:
        """Embed a single text."""
        if not text or not text.strip():
            return None
        
        try:
            # Tokenize
            encoded_input = self.tokenizer(
                text.strip(), 
                padding=True, 
                truncation=True, 
                max_length=512,  # BERT's max length
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
            # Convert to list and return
            embedding = sentence_embeddings.cpu().numpy().flatten().tolist()
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for text: {str(e)}")
            return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> Dict[str, Union[List[List[float]], int, str]]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of batches to process (smaller for memory efficiency)
            
        Returns:
            Dict containing embeddings, successful count, and status
        """
        logger.info(f"Starting Financial BERT batch embedding: {len(texts)} texts")
        
        if not texts:
            return {
                "embeddings": [],
                "successful_count": 0,
                "total_count": 0,
                "status": "no_texts"
            }
        
        all_embeddings = []
        successful_count = 0
        
        # Process in smaller batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Filter empty texts in batch
                valid_texts = [text.strip() for text in batch if text and text.strip()]
                
                if not valid_texts:
                    # Add None for all texts in this batch
                    all_embeddings.extend([None] * len(batch))
                    continue
                
                # Tokenize batch
                encoded_input = self.tokenizer(
                    valid_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Convert to lists
                batch_embeddings = sentence_embeddings.cpu().numpy().tolist()
                
                # Map back to original batch order (accounting for filtered texts)
                valid_idx = 0
                for text in batch:
                    if text and text.strip():
                        if valid_idx < len(batch_embeddings):
                            all_embeddings.append(batch_embeddings[valid_idx])
                            successful_count += 1
                            valid_idx += 1
                        else:
                            all_embeddings.append(None)
                    else:
                        all_embeddings.append(None)
                
                logger.debug(f"Batch {i//batch_size + 1}: {len(valid_texts)} embeddings successful")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # Add None for all texts in failed batch
                all_embeddings.extend([None] * len(batch))
        
        logger.info(f"Financial BERT batch embedding completed: {successful_count}/{len(texts)} successful")
        
        return {
            "embeddings": all_embeddings,
            "successful_count": successful_count,
            "total_count": len(texts),
            "model": self.model_name,
            "status": "completed" if successful_count > 0 else "failed"
        }
    
    def close(self):
        """Close the client and free resources."""
        if hasattr(self, 'model'):
            # Move model back to CPU and clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                self.model.cpu()
                torch.cuda.empty_cache()
        logger.info("Financial BERT client closed")


# For backward compatibility
FinBertClient = FinancialBertClient
