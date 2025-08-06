# Testing Framework - SEC Filings QA Engine

## Overview

This directory contains the comprehensive testing suite for the SEC Filings QA Engine. The tests are organized into multiple categories to ensure thorough validation of system functionality, performance, and reliability.

## Test Structure

```
tests/
├── unit/           # Individual component tests
├── integration/    # System integration tests  
├── performance/    # Performance and load tests
├── fixtures/       # Test data and mocks
└── README.md       # This documentation
```

## Test Categories

### Unit Tests (`/unit/`)
Tests individual components in isolation with mocked dependencies.

- **test_embeddings.py**: Embedding generation and ensemble functionality
- **test_components.py**: Core system components
- **test_pipeline.py**: Data processing pipeline components
- **test_content_retrieval.py**: Content retrieval and ranking algorithms

**Running Unit Tests:**
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_embeddings.py -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html
```

### Integration Tests (`/integration/`)
Tests system components working together with real dependencies.

- **test_system_integration.py**: End-to-end system functionality
- **test_company_ingestion.py**: Complete company data ingestion process
- **test_qa_system.py**: Question answering system integration

**Running Integration Tests:**
```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with real Neo4j instance (requires setup)
pytest tests/integration/ --neo4j-uri=bolt://localhost:7687 -v
```

### Performance Tests (`/performance/`)
Tests system performance, scalability, and resource usage.

- **test_query_performance.py**: Query response time benchmarks
- **test_embedding_performance.py**: Embedding generation speed tests
- **test_memory_usage.py**: Memory consumption analysis
- **test_concurrent_load.py**: Concurrent user simulation

**Running Performance Tests:**
```bash
# Run performance tests
pytest tests/performance/ -v --benchmark-only

# Generate performance report
pytest tests/performance/ --benchmark-json=benchmark_results.json
```

### Test Fixtures (`/fixtures/`)
Shared test data, mocks, and utilities used across test categories.

- **sample_sec_filings.json**: Sample SEC filing data for testing
- **mock_embeddings.pkl**: Pre-generated test embeddings
- **test_companies.yaml**: Test company configurations
- **conftest.py**: Shared pytest fixtures and configuration

## Test Configuration

### Environment Setup

Create a test environment configuration file:

```bash
# tests/.env.test
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test_password
NEO4J_DATABASE=test_database

# Use mock services for CI/CD
OPENAI_API_KEY=mock_key_for_testing
VOYAGE_API_KEY=mock_key_for_testing
ENABLE_MOCK_SERVICES=true
```

### Pytest Configuration

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --strict-config
    --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    slow: Slow running tests
    requires_neo4j: Tests requiring Neo4j database
    requires_api: Tests requiring external API access
```

## Test Data Management

### Sample Data Generation

Generate test data for development and testing:

```python
# tests/fixtures/generate_test_data.py
from tests.fixtures.data_generator import TestDataGenerator

generator = TestDataGenerator()
generator.create_sample_filings(companies=["AAPL", "MSFT"], count=5)
generator.create_mock_embeddings(dimension=1024)
generator.export_test_fixtures()
```

### Database Test Setup

```python
# tests/conftest.py
import pytest
from neo4j import GraphDatabase

@pytest.fixture(scope="session")
def test_neo4j():
    """Set up test Neo4j database"""
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "test_password")
    )
    
    # Create test database
    with driver.session(database="system") as session:
        session.run("CREATE DATABASE test_database IF NOT EXISTS")
    
    yield driver
    
    # Cleanup test database
    with driver.session(database="system") as session:
        session.run("DROP DATABASE test_database IF EXISTS")
    
    driver.close()
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      neo4j:
        image: neo4j:5.12
        env:
          NEO4J_AUTH: neo4j/test_password
        options: >-
          --health-cmd "cypher-shell -u neo4j -p test_password 'RETURN 1'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: pytest tests/unit/ --cov=src --cov-report=xml
    
    - name: Run integration tests
      run: pytest tests/integration/ -m "not slow"
      env:
        NEO4J_URI: bolt://localhost:7687
        ENABLE_MOCK_SERVICES: true
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Test Development Guidelines

### Writing Effective Tests

1. **Follow AAA Pattern** (Arrange, Act, Assert):
```python
def test_embedding_generation():
    # Arrange
    text = "Apple Inc. reported strong quarterly results"
    ensemble = EmbeddingEnsemble()
    
    # Act
    embedding = ensemble.embed_single(text)
    
    # Assert
    assert len(embedding) == 1024
    assert isinstance(embedding, list)
    assert all(isinstance(x, float) for x in embedding)
```

2. **Use Descriptive Test Names**:
```python
def test_query_returns_relevant_results_for_revenue_questions():
    """Test that revenue-related queries return relevant SEC filing sections"""
    pass

def test_embedding_ensemble_handles_empty_input_gracefully():
    """Test that embedding ensemble returns appropriate response for empty input"""
    pass
```

3. **Mock External Dependencies**:
```python
@pytest.fixture
def mock_openai_client():
    with patch('openai.ChatCompletion.create') as mock:
        mock.return_value = {
            'choices': [{'message': {'content': 'Test response'}}]
        }
        yield mock
```

### Test Coverage Goals

- **Unit Tests**: 90%+ code coverage
- **Integration Tests**: All major user workflows covered
- **Performance Tests**: Key performance metrics benchmarked
- **Error Handling**: All error paths tested

## Running Tests Locally

### Full Test Suite
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m performance
```

### Test Development Workflow
1. Write failing test first (TDD approach)
2. Implement minimal code to pass test
3. Refactor while maintaining passing tests
4. Ensure test coverage remains above threshold
5. Run full test suite before committing

## Troubleshooting Common Test Issues

### Neo4j Connection Issues
```bash
# Check Neo4j is running
docker ps | grep neo4j

# Verify test database access
cypher-shell -u neo4j -p test_password -d test_database "RETURN 1"
```

### Performance Test Failures
- Ensure adequate system resources (memory, CPU)
- Check for interference from other processes
- Review baseline performance metrics
- Consider environment-specific performance variations

### Mock Service Issues
- Verify mock fixtures are properly configured
- Check test isolation (tests affecting each other)
- Ensure cleanup in teardown methods
- Review fixture scoping (session vs function)

## Contributing to Tests

When contributing new features or fixes:

1. **Add corresponding tests** for new functionality
2. **Update existing tests** when modifying behavior
3. **Ensure all tests pass** before submitting PR
4. **Maintain test coverage** above established thresholds
5. **Document complex test scenarios** in comments

## Support

For questions about testing:
- Review existing test examples in each category
- Check the main project documentation
- Open issues for test-specific problems
- Contact the development team for complex scenarios

---

*Comprehensive testing ensures the reliability and maintainability of the SEC Filings QA Engine in production environments.*