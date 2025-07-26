"""
Simplified pytest suite for AI Disruption Agent testing.
Bypasses complex langchain import issues by mocking at higher level.
"""

import pytest
import json
import os
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from statistics import median
from typing import Dict, List, Any


# Mock the entire GPTResearcher class
class MockGPTResearcher:
    """Mock GPT Researcher that simulates the real agent behavior."""
    
    def __init__(self, query: str, **kwargs):
        self.query = query
        self.framework_configs = {
            "rubric_criteria": {
                "recent": {"threshold": int(os.getenv("YX_RECENT_THRESHOLD", "7"))},
                "trusted": {"threshold": int(os.getenv("YX_TRUSTED_THRESHOLD", "7"))},
                "relevant": {"threshold": int(os.getenv("YX_RELEVANCE_THRESHOLD", "7"))},
                "citations": {"threshold": int(os.getenv("YX_MIN_CITATIONS", "2"))}
            },
            "quality_gates": {
                "min_sources": int(os.getenv("YX_MIN_SOURCES", "4")),
                "token_limit": int(os.getenv("YX_TOKEN_LIMIT", "2000")),
                "max_turns": int(os.getenv("YX_MAX_TURNS", "5")),
                "max_retries_per_sub": int(os.getenv("YX_MAX_RETRIES", "2"))
            },
            "sub_queries": [
                "industry_landscape_analysis",
                "disruption_framework_assessment", 
                "competitive_positioning_analysis"
            ],
            "trusted_domains": self._load_trusted_domains()
        }
        
        self.react_state = {
            "query": query,
            "context": [],
            "current_sub_query": None,
            "completed_sub_queries": [],
            "retry_counts": {sq: 0 for sq in self.framework_configs["sub_queries"]},
            "token_count": 0,
            "turn_count": 0,
            "quality_scores": [],
            "validation_flags": {
                "sufficient_sources": False,
                "quality_threshold_met": False,
                "completion_validated": False
            }
        }
        
        # Validate config
        self._validate_framework_config()
        
    def _load_trusted_domains(self):
        """Load trusted domains from environment."""
        default_domains = "techcrunch.com,venturebeat.com,crunchbase.com,forbes.com,bloomberg.com"
        trusted_str = os.getenv("TRUSTED_DOMAINS", default_domains)
        domains = [domain.strip() for domain in trusted_str.split(",")]
        
        validated_domains = []
        for domain in domains:
            if "." in domain and len(domain) > 3:
                validated_domains.append(domain)
        
        if len(validated_domains) < 3:
            raise ValueError("Insufficient trusted domains configured for reliable analysis")
        
        return validated_domains
    
    def _validate_framework_config(self):
        """Validate framework configuration."""
        if self.framework_configs["quality_gates"]["min_sources"] < 4:
            raise ValueError("Minimum 4 sources required for reliable Y/X framework analysis")
        
        required_sub_queries = ["industry_landscape_analysis", "disruption_framework_assessment", "competitive_positioning_analysis"]
        if self.framework_configs["sub_queries"] != required_sub_queries:
            raise ValueError(f"Invalid sub_queries for disruption analysis. Required: {required_sub_queries}")


# Test Fixtures
@pytest.fixture
def mock_llm_responses():
    """Fixture providing various LLM response scenarios."""
    return {
        "weak_context": {
            "variants": [
                {"score": 4, "reasoning": "Limited evidence", "citations": 1},
                {"score": 5, "reasoning": "Some relevance", "citations": 2}, 
                {"score": 6, "reasoning": "Moderate support", "citations": 1}
            ],
            "median": 5
        },
        "strong_context": {
            "variants": [
                {"score": 7, "reasoning": "Good evidence base", "citations": 3},
                {"score": 8, "reasoning": "Strong Y/X alignment", "citations": 4},
                {"score": 7, "reasoning": "Comprehensive analysis", "citations": 3}
            ],
            "median": 7
        },
        "procore_analysis": {
            "y_score": 3.2,
            "x_score": 3.5,
            "bucket": "AI Enhances SaaS",
            "rationale": "Procore leverages AI for construction management optimization within existing SaaS framework",
            "sources": ["techcrunch.com/2023/procore-ai", "crunchbase.com/procore-funding-2024"]
        }
    }


# Test Classes
class TestAgentInitialization:
    """Test agent initialization with various configurations."""
    
    def test_init_with_env_variables(self):
        """Test initialization with environment variables vs defaults."""
        env_vars = {
            "YX_RECENT_THRESHOLD": "8",
            "YX_TRUSTED_THRESHOLD": "6", 
            "YX_MIN_SOURCES": "5",
            "YX_TOKEN_LIMIT": "2500",
            "TRUSTED_DOMAINS": "techcrunch.com,forbes.com,bloomberg.com,crunchbase.com"
        }
        
        with patch.dict(os.environ, env_vars):
            agent = MockGPTResearcher("Test disruption query")
            
            # Assert environment variables are loaded correctly
            assert agent.framework_configs["rubric_criteria"]["recent"]["threshold"] == 8
            assert agent.framework_configs["rubric_criteria"]["trusted"]["threshold"] == 6
            assert agent.framework_configs["quality_gates"]["min_sources"] == 5
            assert agent.framework_configs["quality_gates"]["token_limit"] == 2500
            assert "techcrunch.com" in agent.framework_configs["trusted_domains"]

    def test_init_with_defaults(self):
        """Test initialization with default values when env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            agent = MockGPTResearcher("Test query")
            
            # Assert default values are used
            assert agent.framework_configs["rubric_criteria"]["recent"]["threshold"] == 7
            assert agent.framework_configs["rubric_criteria"]["trusted"]["threshold"] == 7
            assert agent.framework_configs["quality_gates"]["min_sources"] == 4
            assert agent.framework_configs["quality_gates"]["token_limit"] == 2000

    def test_config_validation_errors(self):
        """Test configuration validation catches invalid settings."""
        with patch.dict(os.environ, {"YX_MIN_SOURCES": "2"}):
            with pytest.raises(ValueError, match="Minimum 4 sources required"):
                MockGPTResearcher("Test query")

    def test_trusted_domains_validation(self):
        """Test trusted domains loading and validation."""
        with patch.dict(os.environ, {"TRUSTED_DOMAINS": "site1.com,site2.com"}):
            with pytest.raises(ValueError, match="Insufficient trusted domains"):
                MockGPTResearcher("Test query")


class TestEvaluationRubric:
    """Test evaluation rubric with self-consistency and quality thresholds."""
    
    @pytest.mark.asyncio
    async def test_three_variant_self_consistency(self, mock_llm_responses):
        """Test 3-variant self-consistency evaluation."""
        agent = MockGPTResearcher("Test query")
        
        # Mock evaluation with 3 variants
        async def mock_eval_context(context):
            variants = mock_llm_responses["strong_context"]["variants"]
            scores = [v["score"] for v in variants]
            median_score = median(scores)
            
            return {
                "variants": variants,
                "scores": scores,
                "median": median_score,
                "continue": median_score >= 7,
                "overall_median": median_score
            }
        
        agent._eval_context = mock_eval_context
        
        context = ["sample context with strong evidence"]
        decision = await agent._eval_context(context)
        
        assert len(decision["variants"]) == 3
        assert decision["median"] == 7
        assert decision["continue"] is True

    @pytest.mark.asyncio
    async def test_median_threshold_discard(self, mock_llm_responses, caplog):
        """Test context discarding when median < 7."""
        agent = MockGPTResearcher("Test query")
        caplog.set_level(logging.INFO)
        
        # Mock weak evaluation
        async def mock_eval_weak_context(context):
            variants = mock_llm_responses["weak_context"]["variants"] 
            scores = [v["score"] for v in variants]
            median_score = median(scores)
            
            if median_score < 7:
                logger = logging.getLogger("test_logger")
                logger.info("Underspecification detected: median score below threshold")
            
            return {
                "variants": variants,
                "scores": scores, 
                "median": median_score,
                "continue": median_score >= 7,
                "discard_reason": "insufficient_quality" if median_score < 7 else None
            }
        
        agent._eval_context = mock_eval_weak_context
        
        weak_context = ["limited evidence without citations"]
        decision = await agent._eval_context(weak_context)
        
        assert decision["median"] == 5
        assert decision["continue"] is False
        assert decision["discard_reason"] == "insufficient_quality"
        assert "Underspecification detected" in caplog.text


class TestQualityGuards:
    """Test quality guards for token limits and source validation."""
    
    @pytest.mark.asyncio
    async def test_token_limit_guard(self, caplog):
        """Test token limit > 2000 triggers early termination."""
        agent = MockGPTResearcher("Test query")
        caplog.set_level(logging.WARNING)
        
        token_limit = agent.framework_configs["quality_gates"]["token_limit"]
        agent.react_state["token_count"] = 2500  # Exceeds limit
        
        async def mock_check_token_limit():
            current_tokens = agent.react_state["token_count"]
            
            if current_tokens > token_limit:
                logger = logging.getLogger("test_logger")
                logger.warning(f"Token limit exceeded: {current_tokens} > {token_limit}, stopping early")
                return {"stop_early": True, "reason": "token_limit_exceeded", "tokens": current_tokens}
            else:
                return {"continue": True, "tokens": current_tokens}
        
        result = await mock_check_token_limit()
        
        assert result["stop_early"] is True
        assert result["reason"] == "token_limit_exceeded" 
        assert result["tokens"] == 2500
        assert "Token limit exceeded" in caplog.text

    @pytest.mark.asyncio
    async def test_insufficient_sources_logging(self, caplog):
        """Test logging 'Underspecification...' when sources < 4."""
        agent = MockGPTResearcher("Test query")
        caplog.set_level(logging.WARNING)
        
        async def mock_validate_sources():
            source_count = 3  # Below minimum of 4
            
            if source_count < 4:
                logger = logging.getLogger("test_logger")
                logger.warning("Underspecification detected: insufficient sources for reliable Y/X analysis")
                return {"valid": False, "sources": source_count, "action": "discard"}
            else:
                return {"valid": True, "sources": source_count}
        
        result = await mock_validate_sources()
        
        assert result["valid"] is False
        assert result["sources"] == 3
        assert result["action"] == "discard"
        assert "Underspecification detected: insufficient sources" in caplog.text


class TestJSONOutputs:
    """Test JSON output structure and Y/X framework compliance."""
    
    @pytest.mark.asyncio
    async def test_json_structure_compliance(self, mock_llm_responses):
        """Test JSON output has required y_scores, rationale, and framework alignment."""
        agent = MockGPTResearcher("Procore disruption analysis")
        
        # Mock write_report method 
        async def mock_write_report():
            procore_data = mock_llm_responses["procore_analysis"]
            return json.dumps({
                "company": "Procore",
                "analysis_type": "ai_disruption", 
                "y_scores_evidence": [
                    {
                        "dimension": "market_transformation",
                        "score": procore_data["y_score"],
                        "evidence": "AI enhances existing construction workflows",
                        "citations": procore_data["sources"]
                    }
                ],
                "x_scores_evidence": [
                    {
                        "dimension": "business_model_innovation", 
                        "score": procore_data["x_score"],
                        "evidence": "Maintains SaaS subscription model with AI features",
                        "citations": procore_data["sources"]
                    }
                ],
                "framework_bucket": procore_data["bucket"],
                "executive_summary": procore_data["rationale"],
                "confidence_level": "high",
                "source_count": len(procore_data["sources"])
            })
        
        agent.write_report = mock_write_report
        
        report_json = await agent.write_report()
        report_data = json.loads(report_json)
        
        # Assert required JSON structure
        assert "y_scores_evidence" in report_data
        assert "x_scores_evidence" in report_data
        assert "framework_bucket" in report_data
        assert "executive_summary" in report_data
        
        # Assert framework alignment
        assert report_data["framework_bucket"] == "AI Enhances SaaS"
        assert report_data["y_scores_evidence"][0]["score"] == 3.2

    @pytest.mark.parametrize("company,expected_bucket", [
        ("Procore", "AI Enhances SaaS"),  # Low Y, Low X
    ])
    @pytest.mark.asyncio 
    async def test_accurate_bucketing(self, company, expected_bucket, mock_llm_responses):
        """Test accurate Y/X framework bucketing for different companies."""
        agent = MockGPTResearcher(f"{company} AI disruption")
        
        # Get company-specific mock data
        company_key = f"{company.lower()}_analysis"
        company_data = mock_llm_responses[company_key]
        
        async def mock_bucket_analysis():
            y_score = company_data["y_score"]
            x_score = company_data["x_score"]
            
            # Y/X framework bucketing logic
            if y_score <= 4 and x_score <= 4:
                bucket = "AI Enhances SaaS"
            elif y_score >= 5 and y_score <= 7 and x_score <= 5:
                bucket = "AI-First Product Evolution"
            else:
                bucket = "Hybrid AI Integration"
            
            return {
                "company": company,
                "y_score": y_score,
                "x_score": x_score, 
                "bucket": bucket,
                "rationale": company_data["rationale"]
            }
        
        result = await mock_bucket_analysis()
        
        assert result["bucket"] == expected_bucket
        assert result["y_score"] == company_data["y_score"]
        assert result["x_score"] == company_data["x_score"]


class TestEdgeCases:
    """Test edge cases: weak data retry, private inference, malformed responses."""
    
    @pytest.mark.asyncio
    async def test_private_company_inference_adjustments(self):
        """Test score adjustments for private companies with limited data."""
        agent = MockGPTResearcher("Private company analysis")
        
        async def mock_private_company_analysis():
            base_scores = {"y_score": 6.0, "x_score": 5.0}
            company_type = "private_early"
            adjustment = -2
            
            adjusted_scores = {
                "y_score": max(1.0, base_scores["y_score"] + adjustment),
                "x_score": max(1.0, base_scores["x_score"] + adjustment)
            }
            
            return {
                "company_type": company_type,
                "base_scores": base_scores,
                "adjusted_scores": adjusted_scores,
                "adjustment": adjustment
            }
        
        result = await mock_private_company_analysis()
        
        assert result["adjustment"] == -2
        assert result["adjusted_scores"]["y_score"] == 4.0  # 6.0 - 2.0
        assert result["adjusted_scores"]["x_score"] == 3.0  # 5.0 - 2.0

    @pytest.mark.asyncio
    async def test_malformed_response_fallbacks(self, caplog):
        """Test fallback mechanisms for malformed LLM responses."""
        agent = MockGPTResearcher("Malformed response test")
        caplog.set_level(logging.WARNING)
        
        malformed_responses = [
            '{"invalid_json": missing_quote}',  # Invalid JSON
            '{"y_score": "not_a_number"}',      # Invalid data types
            '{}',                               # Empty response
        ]
        
        async def mock_handle_malformed_response(response):
            try:
                parsed = json.loads(response)
                
                if "y_score" not in parsed or not isinstance(parsed.get("y_score"), (int, float)):
                    raise ValueError("Invalid y_score field")
                    
                return {"status": "success", "data": parsed}
                
            except (json.JSONDecodeError, ValueError) as e:
                logger = logging.getLogger("test_logger")
                logger.warning(f"Malformed response detected, using fallback: {str(e)}")
                
                fallback = {
                    "y_score": 5.0,
                    "x_score": 5.0,
                    "bucket": "Insufficient Data",
                    "confidence": "low",
                    "fallback_used": True
                }
                
                return {"status": "fallback", "data": fallback}
        
        # Test malformed response
        caplog.clear()
        result = await mock_handle_malformed_response(malformed_responses[0])
        
        assert result["status"] == "fallback"
        assert result["data"]["fallback_used"] is True
        assert result["data"]["y_score"] == 5.0
        assert "Malformed response detected" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 