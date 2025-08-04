"""
Voyage AI Client for Financial Document Embeddings
Handles embedding generation using Voyage Finance-2 model with rate limiting and token management.
"""

import os
import logging
import time
import requests
from typing import List, Dict, Optional, Union
import queue
import threading
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class VoyageClient:
    """Client for Voyage AI Finance-2 embedding model with rate limiting."""
    
    def __init__(self, api_key: str = None, model: str = None, 
                 token_limit: int = None):
        """Initialize Voyage client with token management."""
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.model = model or os.getenv("VOYAGE_MODEL", "voyage-finance-2")
        self.token_limit = token_limit or int(os.getenv("VOYAGE_TOKEN_LIMIT", 1000000))
        
        if not self.api_key:
            raise ValueError("Voyage API key is required")
        
        # Token usage tracking
        self.tokens_used = 0
        self.request_count = 0
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        # Rate limiting setup
        self.request_queue = queue.Queue()
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 3.0  # 3 seconds between requests (very conservative)
        
        # Request retry configuration
        self.max_retries = 2  # Reduced from 3
        self.retry_delay = 5.0  # Longer retry delay
        
        # Failure tracking for auto-disable
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.disabled = False
        
        logger.info(f"Initialized Voyage client with model: {self.model}")
        logger.info(f"Token limit: {self.token_limit:,} tokens")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Voyage Finance-2 uses similar tokenization to other models
        # Rough estimate: 1 token ≈ 4 characters for English text
        return max(1, len(text) // 4)
    
    def _check_token_limit(self, estimated_tokens: int) -> bool:
        """Check if request would exceed token limit."""
        if self.tokens_used + estimated_tokens > self.token_limit:
            logger.warning(f"Token limit would be exceeded: {self.tokens_used + estimated_tokens} > {self.token_limit}")
            return False
        return True
    
    def _rate_limit(self):
        """Implement rate limiting for API requests."""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def _make_embedding_request(self, texts: List[str], retry_count: int = 0) -> Optional[List[List[float]]]:
        """Make embedding request with proper error handling and retries."""
        self._rate_limit()
        
        # Estimate tokens for this request
        total_estimated_tokens = sum(self._estimate_tokens(text) for text in texts)
        
        if not self._check_token_limit(total_estimated_tokens):
            logger.error("Request would exceed token limit - aborting")
            return None
        
        request_data = {
            "input": texts,
            "model": self.model
        }
        
        try:
            response = self.session.post(
                "https://api.voyageai.com/v1/embeddings",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                
                # Update token usage
                actual_tokens = data.get("usage", {}).get("total_tokens", total_estimated_tokens)
                self.tokens_used += actual_tokens
                self.request_count += 1
                
                logger.debug(f"Voyage API success: {len(embeddings)} embeddings, {actual_tokens} tokens")
                return embeddings
                
            elif response.status_code == 429:  # Rate limit
                if retry_count < self.max_retries:
                    retry_delay = self.retry_delay * (2 ** retry_count)
                    logger.warning(f"Rate limited, retrying in {retry_delay} seconds (attempt {retry_count + 1})")
                    time.sleep(retry_delay)
                    return self._make_embedding_request(texts, retry_count + 1)
                else:
                    logger.error("Rate limit exceeded, max retries reached")
                    return None
                    
            else:
                logger.error(f"Voyage API error: {response.status_code} - {response.text}")
                if retry_count < self.max_retries:
                    time.sleep(self.retry_delay)
                    return self._make_embedding_request(texts, retry_count + 1)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if retry_count < self.max_retries:
                time.sleep(self.retry_delay)
                return self._make_embedding_request(texts, retry_count + 1)
            return None
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        embeddings = self._make_embedding_request([text])
        return embeddings[0] if embeddings else None
    
    def embed_batch(self, texts: List[str], batch_size: int = 5) -> Dict[str, Union[List[List[float]], int, str]]:
        """
        Generate embeddings for multiple texts in batches.
        Returns a structured result with success metrics.
        """
        if self.disabled:
            logger.warning("Voyage client is disabled due to consecutive failures")
            return {
                "embeddings": [None] * len(texts),
                "successful_count": 0,
                "total_count": len(texts),
                "status": "disabled"
            }
        
        if not texts:
            return {
                "embeddings": [],
                "successful_count": 0,
                "total_count": 0,
                "status": "no_texts"
            }
        
        logger.info(f"Starting Voyage batch embedding: {len(texts)} texts")
        
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
            return {
                "embeddings": [None] * len(texts),
                "successful_count": 0,
                "total_count": len(texts),
                "status": "no_valid_texts"
            }
        
        # Process in smaller batches to avoid rate limits
        all_embeddings = []
        successful_requests = 0
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            
            logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch)} texts")
            batch_embeddings = self._make_embedding_request(batch)
            
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
                successful_requests += 1
                self.consecutive_failures = 0  # Reset failure counter on success
            else:
                # If batch fails, add None for each text in batch
                logger.warning(f"Batch embedding failed for batch starting at index {i}")
                all_embeddings.extend([None] * len(batch))
                self.consecutive_failures += 1
                
                # Disable client if too many consecutive failures
                if self.consecutive_failures >= self.max_consecutive_failures:
                    logger.error(f"Disabling Voyage client after {self.consecutive_failures} consecutive failures")
                    self.disabled = True
                    break
        
        # Map embeddings back to original positions
        result = [None] * len(texts)
        valid_embedding_idx = 0
        
        for original_idx, valid_idx in text_mapping:
            if valid_idx is not None and valid_embedding_idx < len(all_embeddings):
                result[original_idx] = all_embeddings[valid_embedding_idx]
                valid_embedding_idx += 1
        
        successful_count = sum(1 for emb in result if emb is not None)
        
        logger.info(f"Voyage batch embedding completed: {successful_count}/{len(texts)} successful")
        
        return {
            "embeddings": result,
            "successful_count": successful_count,
            "total_count": len(texts),
            "status": "completed" if successful_count > 0 else "failed"
        }
        
        logger.info(f"Batch embedding completed: {len([e for e in result if e is not None])}/{len(texts)} successful")
        return result
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for this model."""
        # Voyage Finance-2 returns 1024-dimensional embeddings
        return 1024
    
    def get_usage_stats(self) -> Dict[str, any]:
        """Get current usage statistics."""
        return {
            "tokens_used": self.tokens_used,
            "tokens_remaining": self.token_limit - self.tokens_used,
            "token_limit": self.token_limit,
            "usage_percentage": (self.tokens_used / self.token_limit) * 100,
            "request_count": self.request_count,
            "model": self.model
        }
    
    def health_check(self) -> Dict[str, any]:
        """Perform health check of Voyage API connection."""
        try:
            # Test with a small text
            test_embedding = self.embed_text("test")
            
            if test_embedding:
                return {
                    "status": "healthy",
                    "api_key_valid": True,
                    "model": self.model,
                    "embedding_dimension": len(test_embedding),
                    "usage_stats": self.get_usage_stats(),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "Failed to generate test embedding",
                    "model": self.model,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model,
                "timestamp": datetime.now().isoformat()
            }
    
    def reset_usage_stats(self):
        """Reset usage statistics (for testing purposes)."""
        self.tokens_used = 0
        self.request_count = 0
        logger.info("Usage statistics reset")
    
    def close(self):
        """Close the session and cleanup resources."""
        if self.session:
            self.session.close()
        logger.info("Voyage client closed")


# Context manager support
class VoyageClient(VoyageClient):
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
