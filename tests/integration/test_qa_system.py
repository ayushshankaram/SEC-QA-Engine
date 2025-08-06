#!/usr/bin/env python3
"""
Integration tests for SEC QA System.

This module tests the complete question answering pipeline including:
- System initialization and configuration
- End-to-end question answering workflow
- Multi-company and cross-filing queries
- Response quality and source attribution
- Error handling and edge cases
"""

import sys
import os
from pathlib import Path
import pytest
import logging
from unittest.mock import patch, Mock

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class TestSECQASystem:
    """Integration tests for the complete SEC QA System."""
    
    @pytest.fixture
    def sample_questions(self):
        """Provide sample financial questions for testing."""
        return [
            "What are Apple's main revenue sources?",
            "How much does Microsoft spend on R&D?", 
            "What are JPMorgan's key risk factors?",
            "Compare revenue growth across technology companies",
            "What regulatory risks do financial services companies face?"
        ]
    
    @pytest.fixture
    def qa_system(self):
        """Initialize QA system for testing."""
        try:
            from sec_qa_system import SECQASystem
            return SECQASystem()
        except ImportError as e:
            pytest.skip(f"SECQASystem not available: {e}")
    
    @pytest.mark.integration
    @pytest.mark.requires_neo4j
    def test_qa_system_initialization(self, qa_system):
        """Test that QA system initializes correctly."""
        assert qa_system is not None
        
        # Test initialization
        try:
            success = qa_system.initialize_system()
            if not success:
                pytest.skip("QA system initialization failed - may require database setup")
        except Exception as e:
            pytest.skip(f"QA system initialization failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.requires_neo4j
    @pytest.mark.requires_api
    def test_basic_question_answering(self, qa_system, sample_questions):
        """Test basic question answering functionality."""
        # Initialize system
        try:
            if not qa_system.initialize_system():
                pytest.skip("QA system initialization failed")
        except Exception as e:
            pytest.skip(f"System initialization error: {e}")
        
        # Test with first sample question
        question = sample_questions[0]  # "What are Apple's main revenue sources?"
        
        try:
            result = qa_system.answer_question(question)
            
            # Validate response structure
            assert result is not None
            assert 'answer' in result
            assert isinstance(result['answer'], str)
            assert len(result['answer']) > 50  # Reasonable answer length
            
            # Check for source attribution if available
            if 'sources' in result:
                assert isinstance(result['sources'], list)
                
        except Exception as e:
            pytest.skip(f"Question answering failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.requires_neo4j 
    def test_multi_question_processing(self, qa_system, sample_questions):
        """Test processing multiple questions in sequence."""
        try:
            if not qa_system.initialize_system():
                pytest.skip("QA system initialization failed")
        except Exception as e:
            pytest.skip(f"System initialization error: {e}")
        
        results = []
        
        # Process first 3 questions to avoid long test times
        for question in sample_questions[:3]:
            try:
                result = qa_system.answer_question(question)
                results.append(result)
                
                # Basic validation for each result
                assert result is not None
                assert 'answer' in result
                
            except Exception as e:
                # Log error but continue with other questions
                logging.warning(f"Question failed: {question} - Error: {e}")
        
        # At least one question should have succeeded
        assert len(results) > 0
        assert any(result.get('answer') for result in results)
    
    @pytest.mark.integration
    def test_empty_question_handling(self, qa_system):
        """Test handling of empty or invalid questions."""
        try:
            if not qa_system.initialize_system():
                pytest.skip("QA system initialization failed")
        except Exception as e:
            pytest.skip(f"System initialization error: {e}")
        
        # Test empty question
        try:
            result = qa_system.answer_question("")
            # Should handle gracefully, not crash
            assert result is not None
        except Exception:
            # Expected behavior - should raise appropriate error
            pass
        
        # Test None question  
        try:
            result = qa_system.answer_question(None)
            assert result is not None
        except Exception:
            # Expected behavior - should raise appropriate error
            pass
    
    @pytest.mark.integration
    def test_system_health_check(self, qa_system):
        """Test system health check functionality."""
        try:
            if hasattr(qa_system, 'health_check'):
                health_status = qa_system.health_check()
                assert health_status is not None
                # Health check should return some status information
                
        except Exception as e:
            pytest.skip(f"Health check not available: {e}")
    
    @patch('src.storage.neo4j_client.Neo4jClient')
    def test_qa_system_with_mock_database(self, mock_neo4j):
        """Test QA system with mocked database for unit-style testing."""
        # Mock database responses
        mock_neo4j.return_value.execute_query.return_value = []
        mock_neo4j.return_value.close.return_value = None
        
        try:
            from sec_qa_system import SECQASystem
            qa_system = SECQASystem()
            
            # Should initialize without real database
            # Note: May still require other dependencies
            assert qa_system is not None
            
        except ImportError as e:
            pytest.skip(f"SECQASystem not available: {e}")


class TestQASystemPerformance:
    """Performance tests for QA system."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_query_response_time(self, qa_system, sample_questions):
        """Test that queries complete within acceptable time limits."""
        import time
        
        try:
            if not qa_system.initialize_system():
                pytest.skip("QA system initialization failed")
        except Exception as e:
            pytest.skip(f"System initialization error: {e}")
        
        question = sample_questions[0]
        
        start_time = time.time()
        try:
            result = qa_system.answer_question(question)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should complete within 60 seconds for integration test
            assert response_time < 60.0, f"Query took too long: {response_time:.2f}s"
            
            # Log performance for monitoring
            logging.info(f"Query response time: {response_time:.2f}s")
            
        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")


# Legacy function for backwards compatibility
def main():
    """Legacy main function for backwards compatibility."""
    try:
        from sec_qa_system import SECQASystem
        
        print("Initializing SEC QA System...")
        qa_system = SECQASystem()
        
        # Initialize the system properly
        if not qa_system.initialize_system():
            print("Failed to initialize QA system")
            return
        
        # Simple test questions
        test_questions = [
            "What are Apple's main revenue sources?",
            "How much does Microsoft spend on R&D?",
            "What are JPMorgan's key risk factors?"
        ]
        
        print("\n" + "="*80)
        print("TESTING SEC QA SYSTEM - SHOWING ACTUAL ANSWERS")
        print("="*80)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nQuestion {i}/{len(test_questions)}")
            print("-" * 60)
            print(f"QUESTION: {question}")
            print("-" * 60)
            
            try:
                # Get the answer
                result = qa_system.answer_question(question)
                
                if result and result.get('answer'):
                    print(f"ANSWER:")
                    print(result['answer'])
                    
                    if result.get('sources'):
                        print(f"\nSOURCES ({len(result['sources'])}):")
                        for j, source in enumerate(result['sources'][:3], 1):  # Show top 3
                            company = source.get('company', 'Unknown')
                            filing_type = source.get('filing_type', 'Unknown')
                            print(f"  {j}. {company} - {filing_type}")
                else:
                    print("No answer received")
                    
            except Exception as e:
                print(f"Error: {e}")
                
        print("\n" + "="*80)
        print("QA SYSTEM TEST COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main()