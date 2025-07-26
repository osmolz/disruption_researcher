# AI Disruption Agent Testing Suite

## Overview
Comprehensive pytest suite (`test_agent_new.py`) for testing GPT Researcher agent focused on AI disruption analysis with Y/X investment framework. Tests cover robustness, accuracy, and edge cases for company assessment scenarios like Procore (Low Y/Low X "AI Enhances SaaS").

## Test Coverage Areas

### 1. Agent Initialization (TestAgentInitialization)
- **Environment Variables**: Tests loading of YX_* configs vs defaults
- **Config Validation**: Ensures minimum 4 sources, trusted domains validation  
- **Trusted Domains**: Tests domain format validation and sufficiency checks
- **Framework Setup**: Validates sub-queries and quality gates configuration

### 2. ReAct Loop Orchestration (TestReActLoop)
- **Basic Flow**: Tests reason/action/observe/evaluate cycle completion
- **Retry Logic**: Tests sub-query retries on insufficient quality (median < 7)
- **Parallel Execution**: Tests concurrent sub-query processing without interference
- **Turn Tracking**: Validates state management across research turns

### 3. Sub-Query Handling (TestSubQueryHandling)
- **Specialization**: Tests each sub-query focuses on correct area:
  - `industry_landscape_analysis` → market context
  - `disruption_framework_assessment` → Y/X scores  
  - `competitive_positioning_analysis` → competitive moat
- **Retry Limits**: Tests max retry enforcement per sub-query
- **Quality Thresholds**: Tests retry triggering on quality < 7

### 4. Evaluation Rubric (TestEvaluationRubric)
- **Self-Consistency**: Tests 3-variant evaluation with median calculation
- **Quality Thresholds**: Tests median >= 7 requirement for continuation
- **Source Requirements**: Tests minimum 4 sources or escape condition
- **Discard Logic**: Tests "Underspecification..." logging and context discarding

### 5. Quality Guards (TestQualityGuards)
- **Token Limits**: Tests early termination when tokens > 2000
- **Source Validation**: Tests insufficient sources logging and discard
- **Multi-Criteria Gates**: Tests combined source/token/quality enforcement
- **Guard Enforcement**: Tests quality gate combinations pass/fail correctly

### 6. JSON Output Structure (TestJSONOutputs)
- **Structure Compliance**: Tests required fields (y_scores_evidence, x_scores_evidence, framework_bucket)
- **Y/X Framework Bucketing**: Tests accurate categorization:
  - Procore: Low Y/Low X → "AI Enhances SaaS"
  - Zendesk: Medium Y/Low-Medium X → "AI-First Product Evolution"
- **Rationale Alignment**: Tests rationale matches Y/X framework dimensions

### 7. Edge Cases (TestEdgeCases)
- **Weak Data Retry**: Tests progressive quality improvement through retries
- **Private Company Adjustments**: Tests score adjustments:
  - Early stage private: -2 points
  - Mature private: -1 point  
  - Public: No adjustment
- **Malformed Responses**: Tests fallback mechanisms for invalid JSON/data
- **Over-Scoring Caps**: Tests score capping when citations < 3 for scores >= 8

### 8. Integration Scenarios (TestIntegrationScenarios)
- **End-to-End Flow**: Tests complete Procore analysis pipeline
- **Quality Assurance**: Tests multi-checkpoint QA pipeline validation
- **Performance**: Tests concurrent analysis safety and memory limits

## Running the Tests

### Prerequisites
```bash
pip install pytest pytest-mock pytest-asyncio pytest-cov
```

### ✅ Working Test Suite (Recommended)
Due to complex langchain import dependencies, use the simplified test suite:

```bash
# Run simplified agent tests (RECOMMENDED)
python -m pytest tests/test_agent_simple.py -v

# Run with coverage reporting
python -m pytest tests/test_agent_simple.py -v --cov-report=term-missing

# Run specific test class
python -m pytest tests/test_agent_simple.py::TestAgentInitialization -v
```

### 🚧 Full Integration Tests (Import Issues)
The comprehensive test suite may have langchain import issues:

```bash
# Run all agent tests (may fail due to imports)
pytest tests/test_agent_new.py -v

# Run specific test class
pytest tests/test_agent_new.py::TestAgentInitialization -v
```

### Advanced Test Options
```bash
# Run only async tests
pytest tests/test_agent_new.py -k "asyncio" -v

# Run parametrized tests
pytest tests/test_agent_new.py -k "parametrize" -v

# Run edge case tests only
pytest tests/test_agent_new.py::TestEdgeCases -v

# Run with detailed logging
pytest tests/test_agent_new.py -v -s --log-cli-level=INFO
```

## Mock Strategy

### LLM Response Mocking
- **No Real API Calls**: All LLM interactions are mocked
- **Varied Scenarios**: Weak/strong context with scores 1-10, citations 0-6
- **Cost = 0**: No actual LLM costs incurred during testing
- **Consistent Results**: Deterministic mock responses for reliable testing

### Key Mock Fixtures
- `mock_llm_responses`: Provides weak/strong/over-score scenarios
- `mock_conductor_responses`: Simulates research conductor with source counts
- `mock_research_components`: Patches all external dependencies
- `sample_queries`: Provides realistic disruption analysis queries

## Expected Test Outcomes

### Successful Test Run
- **Coverage**: Should achieve >90% code coverage
- **All Tests Pass**: 18+ test methods across 8 test classes
- **Quality Validation**: Confirms rubric compliance (median>=7, min 4 sources)
- **Framework Accuracy**: Validates Y/X bucketing for test companies

### Key Assertions Verified
- Environment config loading vs defaults
- ReAct loop state management and turn tracking
- Sub-query retry logic and parallel execution  
- 3-variant self-consistency evaluation
- Quality guard enforcement (tokens, sources, median)
- JSON structure compliance with Y/X framework
- Edge case handling (private adjustments, malformed responses)
- Over-scoring caps based on citation sufficiency

## Hard Case Testing Examples

### Procore Scenario
```python
# Tests Low Y (~3.2) / Low X (~3.5) → "AI Enhances SaaS"
# Validates: Trusted sources from 2023-2025, rubric-gated evidence
# Guards: No over-estimation, proper citation requirements
```

### Edge Case Coverage
```python
# Weak data → retry → strong data progression
# Private company inference adjustments (-1/-2)
# Malformed response fallback to conservative scores
# Token limit exceeded → early termination with logging
```

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure `gpt_researcher` package is in Python path
2. **Async Test Failures**: Check `pytest-asyncio` is installed and `@pytest.mark.asyncio` decorators are present
3. **Mock Failures**: Verify mock patches target correct module paths
4. **Coverage Issues**: Ensure agent.py methods are actually called during tests

### Debug Tips
```bash
# Run with detailed output
pytest tests/test_agent_new.py -v -s --tb=long

# Run single test for debugging  
pytest tests/test_agent_new.py::TestAgentInitialization::test_init_with_env_variables -v -s

# Check mock call history
pytest tests/test_agent_new.py -v --pdb  # Enter debugger on failure
```

## Performance Characteristics
- **Execution Time**: ~30-60 seconds for full suite
- **Memory Usage**: <200MB during testing (monitored)
- **Concurrency Safe**: Tests verify thread-safe concurrent analysis
- **Deterministic**: All tests should pass consistently with mocked dependencies

This test suite ensures the AI disruption agent maintains high accuracy, robustness, and compliance with the Y/X investment framework across diverse company scenarios and edge cases. :-) 