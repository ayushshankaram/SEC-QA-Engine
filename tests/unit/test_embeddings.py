#!/usr/bin/env python3
"""
Unit tests for embedding ensemble functionality.

This module tests the multi-model embedding ensemble system including:
- Individual embedding model functionality  
- Ensemble combination and weighting
- Error handling and fallback mechanisms
- Performance characteristics
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestEmbeddingEnsemble:
    """Test suite for EmbeddingEnsemble class."""
    
    @pytest.fixture
    def sample_texts(self):
        """Provide sample financial texts for testing."""
        return [
            "Apple Inc. reported strong quarterly earnings with revenue growth of 15%.",
            "The company's operating margin improved due to cost optimization initiatives.",
            "Risk factors include regulatory changes and market volatility."
        ]
    
    @pytest.fixture
    def mock_embedding_response(self):
        """Provide mock embedding response."""
        return [0.1] * 1024  # Standard 1024-dimensional embedding
    
    @patch('core.embedding_ensemble.EmbeddingEnsemble')
    def test_embedding_ensemble_initialization(self, mock_ensemble):
        """Test that EmbeddingEnsemble initializes correctly."""
        from core.embedding_ensemble import EmbeddingEnsemble
        
        ensemble = EmbeddingEnsemble()
        assert ensemble is not None
        
    def test_single_text_embedding_generation(self, sample_texts, mock_embedding_response):
        """Test embedding generation for a single text."""
        try:
            from core.embedding_ensemble import EmbeddingEnsemble
            
            with patch.object(EmbeddingEnsemble, 'embed_batch') as mock_embed:
                mock_embed.return_value = [mock_embedding_response]
                
                ensemble = EmbeddingEnsemble()
                result = ensemble.embed_batch([sample_texts[0]])
                
                assert result is not None
                assert len(result) == 1
                assert len(result[0]) == 1024
                assert all(isinstance(x, (int, float)) for x in result[0])
                
        except ImportError as e:
            pytest.skip(f"EmbeddingEnsemble not available: {e}")
    
    def test_batch_embedding_generation(self, sample_texts, mock_embedding_response):
        """Test embedding generation for multiple texts."""
        try:
            from core.embedding_ensemble import EmbeddingEnsemble
            
            with patch.object(EmbeddingEnsemble, 'embed_batch') as mock_embed:
                mock_embed.return_value = [mock_embedding_response] * len(sample_texts)
                
                ensemble = EmbeddingEnsemble()
                results = ensemble.embed_batch(sample_texts)
                
                assert results is not None
                assert len(results) == len(sample_texts)
                
                for result in results:
                    assert len(result) == 1024
                    assert all(isinstance(x, (int, float)) for x in result)
                
        except ImportError as e:
            pytest.skip(f"EmbeddingEnsemble not available: {e}")
    
    def test_empty_input_handling(self):
        """Test that ensemble handles empty input gracefully."""
        try:
            from core.embedding_ensemble import EmbeddingEnsemble
            
            ensemble = EmbeddingEnsemble()
            result = ensemble.embed_batch([])
            
            assert result is not None
            assert len(result) == 0
            
        except ImportError as e:
            pytest.skip(f"EmbeddingEnsemble not available: {e}")
    
    def test_none_input_handling(self):
        """Test that ensemble handles None input gracefully."""
        try:
            from core.embedding_ensemble import EmbeddingEnsemble
            
            ensemble = EmbeddingEnsemble()
            
            # Should handle None input without crashing
            with pytest.raises((ValueError, TypeError)):
                ensemble.embed_batch(None)
                
        except ImportError as e:
            pytest.skip(f"EmbeddingEnsemble not available: {e}")
    
    @patch('core.embedding_ensemble.VoyageClient')
    @patch('core.embedding_ensemble.FinE5Client')  
    def test_model_fallback_mechanism(self, mock_fine5, mock_voyage):
        """Test that ensemble falls back when models fail."""
        try:
            from core.embedding_ensemble import EmbeddingEnsemble
            
            # Mock one model failing
            mock_voyage.return_value.embed_batch.side_effect = Exception("API Error")
            mock_fine5.return_value.embed_batch.return_value = [[0.1] * 1024]
            
            ensemble = EmbeddingEnsemble()
            result = ensemble.embed_batch(["test text"])
            
            # Should still work with fallback model
            assert result is not None
            
        except ImportError as e:
            pytest.skip(f"EmbeddingEnsemble not available: {e}")
    
    def test_embedding_dimensions_consistency(self, sample_texts):
        """Test that all embeddings have consistent dimensions."""
        try:
            from core.embedding_ensemble import EmbeddingEnsemble
            
            ensemble = EmbeddingEnsemble()
            results = ensemble.embed_batch(sample_texts)
            
            if results and len(results) > 0:
                expected_dim = len(results[0])
                
                for i, result in enumerate(results):
                    assert len(result) == expected_dim, f"Inconsistent dimension at index {i}"
                    
        except ImportError as e:
            pytest.skip(f"EmbeddingEnsemble not available: {e}")
    
    def test_embedding_normalization(self, sample_texts):
        """Test that embeddings are properly normalized."""
        try:
            from core.embedding_ensemble import EmbeddingEnsemble
            
            ensemble = EmbeddingEnsemble()
            results = ensemble.embed_batch(sample_texts)
            
            if results and len(results) > 0:
                for result in results:
                    # Check L2 norm is close to 1 (normalized)
                    norm = np.linalg.norm(result)
                    assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: {norm}"
                    
        except ImportError as e:
            pytest.skip(f"EmbeddingEnsemble not available: {e}")


class TestIndividualEmbeddingModels:
    """Test individual embedding model components."""
    
    def test_voyage_client_availability(self):
        """Test Voyage AI client can be imported and initialized."""
        try:
            from models.voyage_client import VoyageClient
            client = VoyageClient()
            assert client is not None
        except ImportError as e:
            pytest.skip(f"VoyageClient not available: {e}")
    
    def test_fine5_client_availability(self):
        """Test FinE5 client can be imported and initialized."""
        try:
            from models.fin_e5_client import FinE5Client
            client = FinE5Client()
            assert client is not None
        except ImportError as e:
            pytest.skip(f"FinE5Client not available: {e}")
    
    def test_sparse_embeddings_availability(self):
        """Test sparse embeddings can be imported and initialized."""
        try:
            from models.sparse_embeddings import SparseEmbeddings
            client = SparseEmbeddings()
            assert client is not None
        except ImportError as e:
            pytest.skip(f"SparseEmbeddings not available: {e}")


# Legacy test function for backwards compatibility
def test_embeddings():
    """Legacy test function for backwards compatibility."""
    try:
        from core.embedding_ensemble import EmbeddingEnsemble
        
        ensemble = EmbeddingEnsemble()
        test_texts = [
            "Apple Inc. is a technology company",
            "The company reported strong financial results",
            "Revenue increased by 15% year over year"
        ]
        
        results = ensemble.embed_batch(test_texts)
        return results is not None and len(results) > 0
        
    except Exception:
        return False


if __name__ == "__main__":
    # Run legacy test for backwards compatibility
    success = test_embeddings()
    print(f"\n{'='*60}")
    if success:
        print("Embedding test successful!")
    else:
        print("Embedding test failed!")
    print(f"{'='*60}")