"""
OpenAI Embeddings Client
Free alternative/backup to Voyage AI for document embeddings.
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


class OpenAIEmbeddingsClient:
    """Client for OpenAI embeddings as a free alternative to Voyage."""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize OpenAI embeddings client."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or "text-embedding-3-small"  # Cheaper, smaller model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Rate limiting setup (OpenAI has generous limits)
        self.request_queue = queue.Queue()
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second (conservative)
        
        # Request retry configuration
        self.max_retries = 3
        self.retry_delay = 2.0
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        logger.info(f"Initialized OpenAI embeddings client with model: {self.model}")
        
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model."""
        # text-embedding-3-small has 1536 dimensions
        # text-embedding-3-large has 3072 dimensions
        if "large" in self.model:
            return 3072
        return 1536
    
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
        
        request_data = {
            "input": texts,
            "model": self.model
        }
        
        try:
            response = self.session.post(
                "https://api.openai.com/v1/embeddings",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                return embeddings
                
            elif response.status_code == 429:  # Rate limited
                if retry_count < self.max_retries:
                    wait_time = self.retry_delay * (2 ** retry_count)
                    logger.warning(f"Rate limited, retrying in {wait_time} seconds (attempt {retry_count + 1})")
                    time.sleep(wait_time)
                    return self._make_embedding_request(texts, retry_count + 1)
                else:
                    logger.error("Rate limit exceeded, max retries reached")
                    return None
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Exception during API request: {str(e)}")
            if retry_count < self.max_retries:
                time.sleep(self.retry_delay)
                return self._make_embedding_request(texts, retry_count + 1)
            return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> Dict[str, Union[List[List[float]], int, str]]:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of batches to process
            
        Returns:
            Dict containing embeddings, successful count, and status
        """
        logger.info(f"Starting OpenAI embedding batch: {len(texts)} texts")
        
        all_embeddings = []
        successful_count = 0
        
        # Process in smaller batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                batch_embeddings = self._make_embedding_request(batch)
                
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
                    successful_count += len(batch)
                    logger.debug(f"Batch {i//batch_size + 1}: {len(batch)} embeddings successful")
                else:
                    logger.warning(f"Batch embedding failed for batch starting at index {i}")
                    # Add None placeholders for failed embeddings
                    all_embeddings.extend([None] * len(batch))
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                all_embeddings.extend([None] * len(batch))
        
        logger.info(f"OpenAI batch embedding completed: {successful_count}/{len(texts)} successful")
        
        return {
            "embeddings": all_embeddings,
            "successful_count": successful_count,
            "total_count": len(texts),
            "model": self.model
        }
    
    def embed_single(self, text: str) -> Optional[List[float]]:
        """Embed a single text."""
        result = self._make_embedding_request([text])
        if result and len(result) > 0:
            return result[0]
        return None
    
    def close(self):
        """Close the client session."""
        if hasattr(self, 'session'):
            self.session.close()
        logger.info("OpenAI embeddings client closed")


# For backward compatibility if needed
OpenAIClient = OpenAIEmbeddingsClient
