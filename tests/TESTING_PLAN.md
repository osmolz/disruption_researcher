# AI Disruption Agent Testing Plan & Methodology

## High-Level Plan

This comprehensive testing strategy follows a **step-by-step approach** to validate the AI disruption agent's robustness and accuracy for company assessment using the Y/X investment framework.

### Core Testing Philosophy
- **Mock-First**: All LLM calls mocked to ensure deterministic, cost-free testing
- **Hard Cases Focus**: Emphasis on edge cases like Procore (Low Y/Low X) with rigorous validation
- **Rubric Compliance**: Validates 3-variant self-consistency, median>=7, min 4 sources
- **Framework Accuracy**: Tests Y/X bucketing with trusted 2023-2025 sources

---

## Step-by-Step Implementation Plan

### Step 1: Add Fixtures (Mock Infrastructure)
**Purpose**: Create comprehensive mock ecosystem for varied testing scenarios

<details>
<summary><strong>Implementation Details</strong></summary>

```python
@pytest.fixture
def mock_llm_responses():
    """CoT: Mock LLM returns JSON scores/variants for different scenarios"""
    return {
        "weak_context": {
            "variants": [scores 4-6, citations 1-2],
            "median": 5,  # Below threshold
            "should_discard": True
        },
        "strong_context": {
            "variants": [scores 7-8, citations 3-4], 
            "median": 7,  # Meets threshold
            "should_discard": False
        }
    }
```

**Key Assertions**: Mock returns varied scores 1-10, citations 0-6, deterministic results
</details>

### Step 2: Test Initialization (Configuration & Setup)
**Purpose**: Validate agent initialization with environment vs default configurations

<details>
<summary><strong>Test Implementation</strong></summary>

```python
def test_init_with_env_variables(mock_research_components):
    """Assert trusted_domains loaded from environment"""
    env_vars = {"TRUSTED_DOMAINS": "techcrunch.com,forbes.com,bloomberg.com"}
    
    with patch.dict(os.environ, env_vars):
        agent = GPTResearcher("Test query")
        assert "techcrunch.com" in agent.framework_configs["trusted_domains"]
        assert agent.framework_configs["quality_gates"]["min_sources"] == 4
```

**Key Assertions**: Environment loading, config validation, trusted domains sufficiency
</details>

### Step 3: Test ReAct Loop (Reason/Action/Observe/Eval)
**Purpose**: Validate ReAct orchestration with turn tracking and parallel sub-queries

<details>
<summary><strong>Test Implementation</strong></summary>

```python
@pytest.mark.asyncio
async def test_conduct_research_basic_flow(mock_research_components):
    """Mock LLM: Parse XML action/sub_query for incomplete state"""
    async def mock_conduct_research():
        agent.react_state["turn_count"] = 3
        agent.react_state["completed_sub_queries"] = ["industry_landscape", "framework_assessment", "competitive_positioning"]
        return {"status": "completed", "turns": 3}
    
    result = await agent.conduct_research()
    assert result["status"] == "completed"
    assert agent.react_state["validation_flags"]["sufficient_sources"]
```

**Key Assertions**: Turn tracking, sub-query completion, state management, parallel execution
</details>

### Step 4: Test Sub-Query Handling (Industry/Framework/Competition)
**Purpose**: Validate specialized sub-query execution with retry logic

<details>
<summary><strong>Test Implementation</strong></summary>

```python
@pytest.mark.parametrize("sub_query,expected_focus", [
    ("industry_landscape_analysis", "market_context"),
    ("disruption_framework_assessment", "y_x_scores"),
    ("competitive_positioning_analysis", "competitive_moat")
])
async def test_sub_query_specialization(sub_query, expected_focus):
    """Mock loop: 3 turns, sub-queries parallel gather, eval pass/fail/retry"""
    result = await mock_execute_specialized_query(sub_query)
    assert result["focus"] == expected_focus
    assert result["sources"] >= 3
```

**Key Assertions**: Specialization focus areas, retry limits, parallel execution
</details>

### Step 5: Test Evaluation Rubric (3 Variants Self-Consistency)
**Purpose**: Validate 3-variant evaluation with median>=7 threshold and source requirements

<details>
<summary><strong>Test Implementation</strong></summary>

```python
async def test_three_variant_self_consistency(mock_llm_responses):
    """3 variants, median<7 discard/log, clean_text called"""
    variants = mock_llm_responses["strong_context"]["variants"]
    scores = [v["score"] for v in variants]
    median_score = median(scores)
    
    decision = await agent._eval_context(context)
    assert len(decision["variants"]) == 3
    assert decision["median"] >= 7
    assert decision["continue"] is True
```

**Key Assertions**: 3-variant consistency, median calculation, threshold enforcement, logging
</details>

### Step 6: Test Quality Guards (Token/Source Limits)
**Purpose**: Validate quality gates with token>2000 and sources<4 enforcement

<details>
<summary><strong>Test Implementation</strong></summary>

```python
async def test_token_limit_guard(caplog):
    """Assert logs 'Underspecification...' on weak data"""
    agent.react_state["token_count"] = 2500  # Exceeds limit
    
    result = await mock_check_token_limit()
    assert result["stop_early"] is True
    assert "Token limit exceeded" in caplog.text
```

**Key Assertions**: Token limit enforcement, insufficient sources logging, early termination
</details>

### Step 7: Test JSON Output Structure (Y/X Framework Compliance)
**Purpose**: Validate JSON structure with y_scores/rationale matching framework

<details>
<summary><strong>Test Implementation</strong></summary>

```python
async def test_json_structure_compliance(mock_llm_responses):
    """Mock context: assert JSON keys/y_scores/rationale, fallback on error"""
    report_json = await agent.write_report()
    report_data = json.loads(report_json)
    
    assert "y_scores_evidence" in report_data
    assert "framework_bucket" in report_data
    assert report_data["framework_bucket"] == "AI Enhances SaaS"  # Procore case
```

**Key Assertions**: JSON structure, Y/X scores evidence, framework bucketing accuracy
</details>

### Step 8: Test Edge Cases (Private Companies, Malformed Responses)
**Purpose**: Validate edge case handling with private inference and fallbacks

<details>
<summary><strong>Test Implementation</strong></summary>

```python
@pytest.mark.parametrize("company_type,adjustment", [
    ("private_early", -2), ("private_mature", -1), ("public", 0)
])
async def test_private_company_inference_adjustments(company_type, adjustment):
    """No configs → error JSON, weak → 'Underspecification...' log/empty"""
    result = await mock_private_company_analysis()
    
    if company_type == "private_early":
        assert result["adjusted_scores"]["y_score"] == 4.0  # 6.0 - 2.0
    assert result["adjustment"] == adjustment
```

**Key Assertions**: Private company adjustments (-1/-2), malformed response fallbacks, over-scoring caps
</details>

---

## Testing Methodology & Verification

### Mock Strategy (No Real LLM Calls)
```python
# Always mock LLM to return fake JSON with random scores 1-10/cites 0-4
@pytest.fixture 
def mock_llm_responses():
    return {
        "weak_context": {"median": 5, "should_discard": True},
        "strong_context": {"median": 8, "should_discard": False}  
    }
```

### Logging Verification (caplog)
```python
def test_underspecification_logging(caplog):
    # Use caplog to assert logs (e.g., "Underspecification...")
    assert "Underspecification detected" in caplog.text
```

### Async Testing (pytest.mark.asyncio)
```python
@pytest.mark.asyncio
async def test_async_method():
    # pytest.mark.asyncio for async methods
    result = await agent.conduct_research()
```

### Parametrized Hard Cases
```python
@pytest.mark.parametrize("company,expected_bucket", [
    ("Procore", "AI Enhances SaaS"),    # Low Y/Low X
    ("Zendesk", "AI-First Product Evolution")  # Medium Y/Low-Medium X
])
```

---

## Coverage Requirements & Validation

### Target Coverage: >90%
- **Initialization**: Config loading, validation, environment handling
- **ReAct Loop**: Turn management, sub-query orchestration, state tracking  
- **Evaluation**: 3-variant consistency, median thresholds, source validation
- **Quality Guards**: Token limits, source counts, early termination
- **JSON Output**: Structure compliance, framework bucketing, rationale alignment
- **Edge Cases**: Private adjustments, malformed responses, retry mechanisms

### Rubric Compliance Testing
- **Self-Consistency**: 3 variants with median>=7 only if recent/trusted/relevant
- **Citations**: min>=2 cites per claim, cap scores<5 without sufficient evidence
- **Sources**: min 4 sources or escape with "Underspecification..." logging
- **Framework**: Accurate Y/X bucketing (Procore → "AI Enhances SaaS")

### Hard Case Scenarios
1. **Procore Analysis**: Low Y~3.2/Low X~3.5 with trusted 2023-2025 sources
2. **Zendesk Analysis**: Medium Y~6.2/Low-Medium X~4.1 transformation assessment
3. **Private Companies**: Early stage (-2 adjustment) vs mature (-1 adjustment)
4. **Weak Data**: Retry mechanism with progressive quality improvement
5. **Over-Scoring**: Cap high scores without sufficient citations
6. **Token Limits**: Early termination when context exceeds 2000 tokens

---

## Execution Strategy

### Test Execution Commands
```bash
# Full comprehensive test suite
python tests/run_agent_tests.py --coverage

# Edge cases only  
python tests/run_agent_tests.py --edge-cases

# Quick validation
python tests/run_agent_tests.py --quick

# Debug mode
python tests/run_agent_tests.py --debug
```

### Expected Outcomes
- **18+ Test Methods**: Across 8 comprehensive test classes
- **Zero LLM Costs**: All API calls mocked for cost-free testing
- **Deterministic Results**: Consistent pass/fail behavior across runs
- **Performance Monitoring**: Memory <200MB, execution <60 seconds
- **Framework Validation**: Accurate Y/X bucketing for all test companies

This comprehensive testing approach ensures the AI disruption agent maintains **high accuracy, robustness, and compliance** with the Y/X investment framework across diverse company scenarios and challenging edge cases. :-) 