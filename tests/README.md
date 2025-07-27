# Tests Directory

This directory contains all tests for the AI Blog Generation Team project.

## Test Categories

### Unit Tests
- `test_data_models.py` - Tests for data models and validation
- `test_dependencies.py` - Tests for shared dependencies
- `test_research_agent.py` - Unit tests for Research Agent
- `test_writing_agent.py` - Unit tests for Writing Agent
- `test_critique_agent.py` - Unit tests for Critique Agent
- `test_orchestrator_agent.py` - Unit tests for Orchestrator Agent
- `test_progress_tracking.py` - Tests for progress tracking functionality
- `test_error_handling.py` - Tests for error handling mechanisms

### Integration Tests
- `test_research_integration.py` - Integration tests for Research Agent with real APIs
- `test_writing_integration.py` - Integration tests for Writing Agent with real models
- `test_critique_integration.py` - Integration tests for Critique Agent with real models
- `test_revision_workflow_integration.py` - Tests for the complete revision workflow
- `test_orchestrator_integration.py` - Full workflow integration tests
- `test_orchestrator_progress_integration.py` - Progress tracking integration tests
- `test_agent_error_integration.py` - Error handling integration tests

## Running Tests

### Prerequisites
Make sure you have the required environment variables set:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

### Run All Integration Tests
From the project root:
```bash
python run_tests.py
```

### Run Individual Tests
```bash
# Run a specific integration test
python tests/test_orchestrator_integration.py

# Run unit tests with pytest
pytest tests/test_data_models.py -v

# Run all unit tests
pytest tests/ -k "not integration" -v

# Run all tests
pytest tests/ -v
```

### Test Requirements
- **Unit Tests**: Only require the project dependencies
- **Integration Tests**: Require valid API keys for OpenAI and Tavily
- **Network**: Integration tests require internet connectivity

## Test Structure

Each integration test follows this pattern:
1. Check for required API keys
2. Initialize models and dependencies
3. Create agents
4. Run test scenarios
5. Verify results
6. Display comprehensive output

Unit tests use mocking to avoid external dependencies and focus on testing individual components in isolation.