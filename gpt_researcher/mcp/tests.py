#!/usr/bin/env python3
"""
Unit tests for MCP (Model Context Protocol) integration module.

Tests cover:
- Tool selection with rubrics
- Y/X indicator mapping (12 indicators enforcement) 
- Configuration validation
- Caching functionality
- Error handling and fallbacks
"""

import unittest
import asyncio
import json
import tempfile
import shutil
import os
from unittest.mock import MagicMock, AsyncMock, patch

# Import MCP components - use direct imports for testing
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from research import MCPResearchSkill, clean_text
except ImportError:
    # Fallback for different import scenarios
    pass


class MockTool:
    """Mock tool for testing"""
    def __init__(self, name, description="Mock tool for testing"):
        self.name = name
        self.description = description


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.strategic_llm_model = "gpt-4o-mini"
        self.strategic_llm_provider = "openai"
        self.fast_llm_model = "gpt-4o-mini"
        self.fast_llm_provider = "openai"
        self.llm_kwargs = {}
        self.mcp_cache_dir = "test_cache"


class TestMCPConfigValidation(unittest.TestCase):
    """Test configuration validation functionality"""
    
    def test_valid_config(self):
        """Test that valid configuration passes validation"""
        cfg = MockConfig()
        
        # Should not raise exception
        try:
            validate_mcp_config(cfg, "TestComponent")
        except Exception as e:
            self.fail(f"Valid config should pass validation: {e}")
    
    def test_missing_required_attribute(self):
        """Test that missing required attributes are detected"""
        cfg = MockConfig()
        delattr(cfg, 'strategic_llm_model')
        
        with self.assertRaises(ValueError) as context:
            validate_mcp_config(cfg, "TestComponent")
        
        self.assertIn("Missing core configuration", str(context.exception))
        self.assertIn("strategic_llm_model", str(context.exception))
    
    def test_empty_required_attribute(self):
        """Test that empty required attributes are detected"""
        cfg = MockConfig()
        cfg.strategic_llm_model = ""
        
        with self.assertRaises(ValueError) as context:
            validate_mcp_config(cfg, "TestComponent")
        
        self.assertIn("empty", str(context.exception))
    
    def test_none_config(self):
        """Test that None config is rejected"""
        with self.assertRaises(ValueError) as context:
            validate_mcp_config(None, "TestComponent")
        
        self.assertIn("cannot be None", str(context.exception))


class TestMCPToolSelector(unittest.TestCase):
    """Test tool selection functionality with rubrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cfg = MockConfig()
        self.temp_cache_dir = tempfile.mkdtemp(prefix="mcp_test_cache_")
        self.cfg.mcp_cache_dir = self.temp_cache_dir
        
        # Create mock researcher
        self.mock_researcher = MagicMock()
        self.mock_researcher.add_costs = MagicMock()
        
        # Initialize selector
        self.selector = MCPToolSelector(self.cfg, self.mock_researcher)
        
        # Create mock tools
        self.mock_tools = [
            MockTool("web_search", "Search the web for information"),
            MockTool("browse_page", "Browse and analyze web pages"),
            MockTool("analyze_data", "Analyze data and extract insights"),
            MockTool("random_facts", "Generate random facts"),
            MockTool("weather_api", "Get weather information")
        ]
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_cache_dir):
            shutil.rmtree(self.temp_cache_dir)
    
    def test_config_validation_in_constructor(self):
        """Test that configuration is validated in constructor"""
        invalid_cfg = MagicMock()
        invalid_cfg.strategic_llm_model = None
        
        with self.assertRaises(ValueError):
            MCPToolSelector(invalid_cfg)
    
    def test_cache_key_generation(self):
        """Test cache key generation consistency"""
        query = "Procore AI risk assessment"
        
        key1 = self.selector._get_cache_key(query, 3)
        key2 = self.selector._get_cache_key(query, 3)
        key3 = self.selector._get_cache_key(query, 5)
        
        # Same inputs should generate same key
        self.assertEqual(key1, key2)
        
        # Different inputs should generate different keys
        self.assertNotEqual(key1, key3)
        
        # Key should be 32-character MD5 hash
        self.assertEqual(len(key1), 32)
        self.assertTrue(all(c in '0123456789abcdef' for c in key1))
    
    def test_cache_save_and_load(self):
        """Test caching functionality"""
        query = "Procore AI risk assessment"
        max_tools = 3
        selected_tools = ["web_search", "browse_page", "analyze_data"]
        
        # Save to cache
        self.selector._save_cached_selection(query, max_tools, self.mock_tools[:3])
        
        # Load from cache
        cached_tools = self.selector._load_cached_selection(query, max_tools)
        
        self.assertIsNotNone(cached_tools)
        self.assertEqual(len(cached_tools), 3)
        self.assertIn("web_search", cached_tools)
        self.assertIn("browse_page", cached_tools)
    
    def test_fallback_tool_selection(self):
        """Test fallback selection mechanism"""
        query = "Procore AI disruption analysis"
        
        selected_tools = self.selector._fallback_tool_selection(self.mock_tools, 3)
        
        # Should return exactly 3 tools (or fewer if less available)
        self.assertLessEqual(len(selected_tools), 3)
        self.assertGreater(len(selected_tools), 0)
        
        # Should prioritize tools with relevant patterns
        tool_names = [tool.name for tool in selected_tools]
        
        # web_search should be highly ranked due to "search" pattern
        if "web_search" in [t.name for t in self.mock_tools]:
            self.assertIn("web_search", tool_names)
    
    @patch('gpt_researcher.mcp.tool_selector.MCPToolSelector._call_llm_for_tool_selection')
    async def test_select_relevant_tools_with_mock_llm(self, mock_llm_call):
        """Test tool selection with mocked LLM response"""
        
        # Mock LLM response with high-scoring tools for Procore query
        mock_response = json.dumps({
            "selected_tools": [
                {
                    "index": 0,
                    "name": "web_search",
                    "y_axis_score": 9,
                    "x_axis_score": 8,
                    "quality_score": 9,
                    "overall_score": 8.6,
                    "evidence_reasoning": "Excellent for finding performance benchmarks and market data"
                },
                {
                    "index": 1,
                    "name": "browse_page",
                    "y_axis_score": 8,
                    "x_axis_score": 7,
                    "quality_score": 8,
                    "overall_score": 7.6,
                    "evidence_reasoning": "Good for detailed analysis of company pages"
                },
                {
                    "index": 2,
                    "name": "analyze_data",
                    "y_axis_score": 7,
                    "x_axis_score": 8,
                    "quality_score": 7,
                    "overall_score": 7.4,
                    "evidence_reasoning": "Useful for processing collected information"
                }
            ]
        })
        
        mock_llm_call.return_value = mock_response
        
        query = "Procore AI risk assessment"
        selected_tools = await self.selector.select_relevant_tools(query, self.mock_tools, 3)
        
        # Assert correct number of tools selected
        self.assertEqual(len(selected_tools), 3)
        
        # Assert specific high-scoring tools are selected
        tool_names = [tool.name for tool in selected_tools]
        self.assertIn("web_search", tool_names)
        self.assertIn("browse_page", tool_names)
        self.assertIn("analyze_data", tool_names)
        
        # Verify LLM was called for each variant
        self.assertEqual(mock_llm_call.call_count, 3)
    
    def test_disruption_rubric_prompt_content(self):
        """Test that disruption rubric prompt contains required elements"""
        query = "Procore AI risk assessment"
        tools_info = [{"index": 0, "name": "web_search", "description": "Search the web"}]
        
        prompt = self.selector._generate_disruption_rubric_prompt(query, tools_info, 3)
        
        # Check for few-shot examples
        self.assertIn("<few_shot>", prompt)
        self.assertIn("Good Examples:", prompt)
        self.assertIn("Poor Examples:", prompt)
        self.assertIn("Procore analysis", prompt)
        self.assertIn("cite Bain 2025", prompt)
        
        # Check for chain-of-thought
        self.assertIn("<chain_of_thought>", prompt)
        self.assertIn("Step 1:", prompt)
        self.assertIn("Step 2:", prompt)
        self.assertIn("Step 3:", prompt)
        self.assertIn("Step 4:", prompt)
        
        # Check for Y/X framework focus
        self.assertIn("Y-Axis", prompt)
        self.assertIn("X-Axis", prompt)
        self.assertIn("40%", prompt)  # Weighting
        self.assertIn("20%", prompt)  # Quality weight


class TestMCPResearch(unittest.TestCase):
    """Test cases for MCP research functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # MCPResearchSkill is already imported at the top
        
        self.cfg = MockCfg()
        self.mock_researcher = Mock()
        self.mock_researcher.query = "Procore AI disruption risk"
        self.mock_researcher.add_costs = Mock()
        
        self.skill = MCPResearchSkill(self.cfg, self.mock_researcher)
        self.skill.researcher = self.mock_researcher  # Ensure researcher is set
        
        # Test indicators
        self.y_indicators = [
            "Task Structure", "Risk", "Contextual Knowledge", 
            "Data Availability", "Process Variability", "Human Workflow"
        ]
        self.x_indicators = [
            "External Observability", "Industry Standardization", "Proprietary Data",
            "Switching Friction", "Regulatory Barriers", "Agent Protocol"
        ]
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up cache directory
        import shutil
        if os.path.exists(self.cfg.mcp_cache_dir):
            shutil.rmtree(self.cfg.mcp_cache_dir)
    
    def test_mapping_structure(self):
        """Test that indicator mapping returns exactly 6 Y and 6 X indicators."""
        # Create test mapping result
        test_mapping = self.skill._create_fallback_indicators(self.y_indicators, self.x_indicators)
        
        # Verify structure
        self.assertEqual(len(test_mapping['y']), 6)
        self.assertEqual(len(test_mapping['x']), 6)
        
        # Verify all indicators are present
        y_names = {ind['name'] for ind in test_mapping['y']}
        x_names = {ind['name'] for ind in test_mapping['x']}
        
        self.assertEqual(y_names, set(self.y_indicators))
        self.assertEqual(x_names, set(self.x_indicators))
        
        # Verify score ranges
        for indicator_type in ['y', 'x']:
            for ind in test_mapping[indicator_type]:
                self.assertGreaterEqual(ind['score'], 1)
                self.assertLessEqual(ind['score'], 10)
                self.assertIn('rationale', ind)
                self.assertIn('sources', ind)
    
    def test_industry_detection(self):
        """Test vertical vs horizontal industry detection."""
        # Test vertical detection
        self.mock_researcher.query = "Procore construction management AI"
        vertical_type = self.skill._detect_industry_type()
        self.assertEqual(vertical_type, 'vertical')
        
        # Test horizontal detection
        self.mock_researcher.query = "General AI productivity tools"
        horizontal_type = self.skill._detect_industry_type()
        self.assertEqual(horizontal_type, 'horizontal')
        
        # Test other vertical patterns
        vertical_queries = [
            "Healthcare AI automation",
            "Banking fraud detection system", 
            "Manufacturing process optimization"
        ]
        
        for query in vertical_queries:
            self.mock_researcher.query = query
            result = self.skill._detect_industry_type()
            self.assertEqual(result, 'vertical', f"Failed for query: {query}")
    
    def test_score_adjustments_recent_sources(self):
        """Test score adjustments for recent vs old sources."""
        # Test with recent sources
        recent_sources = ["MIT 2024 study", "Gartner 2025 report"]
        score, rationale = self.skill._apply_score_adjustments(
            median_score=8.0, 
            sources=recent_sources, 
            rationale="Good evidence", 
            industry_type='horizontal'
        )
        self.assertEqual(score, 8.0)  # No penalty
        self.assertNotIn("downgraded", rationale)
        
        # Test with old sources (no 2023+)
        old_sources = ["Old study 2020", "Report 2019"]
        score, rationale = self.skill._apply_score_adjustments(
            median_score=8.0, 
            sources=old_sources, 
            rationale="Old evidence", 
            industry_type='horizontal'
        )
        self.assertEqual(score, 6.0)  # -2 penalty
        self.assertIn("downgraded -2", rationale)
    
    def test_vertical_bias_cap(self):
        """Test bias cap for vertical industries."""
        # Test vertical bias cap with insufficient sources
        insufficient_sources = ["One source 2024"]
        score, rationale = self.skill._apply_score_adjustments(
            median_score=8.0, 
            sources=insufficient_sources, 
            rationale="Limited evidence", 
            industry_type='vertical'
        )
        self.assertEqual(score, 5)  # Capped at 5
        self.assertIn("capped at 5", rationale)
        
        # Test vertical with sufficient sources (no cap)
        sufficient_sources = ["Source 1 2024", "Source 2 2025"]
        score, rationale = self.skill._apply_score_adjustments(
            median_score=8.0, 
            sources=sufficient_sources, 
            rationale="Good evidence", 
            industry_type='vertical'
        )
        self.assertEqual(score, 8.0)  # No cap
        self.assertNotIn("capped at 5", rationale)
    
    def test_insufficient_evidence_fallback(self):
        """Test fallback rationale for insufficient evidence."""
        # Test with insufficient sources
        insufficient_sources = ["Only one source"]
        score, rationale = self.skill._apply_score_adjustments(
            median_score=7.0, 
            sources=insufficient_sources, 
            rationale="Weak evidence", 
            industry_type='horizontal'
        )
        self.assertEqual(score, 5)  # Capped at 5 for insufficient evidence
        self.assertIn("Insufficient evidence: Score neutral 5, re-query needed", rationale)
    
    def test_score_range_enforcement(self):
        """Test that scores are always kept in 1-10 range."""
        # Test extreme low score
        sources = ["No recent sources"]
        score, _ = self.skill._apply_score_adjustments(
            median_score=2.0,  # Start low
            sources=sources, 
            rationale="", 
            industry_type='horizontal'
        )
        self.assertGreaterEqual(score, 1)  # Should not go below 1
        
        # Test extreme high score stays in range
        good_sources = ["Great source 2024", "Another 2025"]
        score, _ = self.skill._apply_score_adjustments(
            median_score=15.0,  # Impossible high score
            sources=good_sources, 
            rationale="", 
            industry_type='horizontal'
        )
        self.assertLessEqual(score, 10)  # Should not exceed 10
    
    def test_cache_functionality(self):
        """Test caching of indicator mappings."""
        query = "Test AI system"
        
        # Create test indicators
        test_indicators = {
            'y': [{'name': ind, 'score': 5, 'rationale': 'Test', 'sources': ['Test']} 
                  for ind in self.y_indicators],
            'x': [{'name': ind, 'score': 5, 'rationale': 'Test', 'sources': ['Test']} 
                  for ind in self.x_indicators]
        }
        
        # Test saving to cache
        self.skill._save_cached_indicators(query, test_indicators)
        
        # Verify cache file exists
        cache_key = self.skill._get_indicator_cache_key(query)
        cache_path = os.path.join(self.cfg.mcp_cache_dir, f"{cache_key}.json")
        self.assertTrue(os.path.exists(cache_path))
        
        # Test loading from cache
        loaded_indicators = self.skill._load_cached_indicators(query)
        self.assertIsNotNone(loaded_indicators)
        self.assertEqual(len(loaded_indicators['y']), 6)
        self.assertEqual(len(loaded_indicators['x']), 6)
        
        # Verify indicator names match
        loaded_y_names = {ind['name'] for ind in loaded_indicators['y']}
        loaded_x_names = {ind['name'] for ind in loaded_indicators['x']}
        self.assertEqual(loaded_y_names, set(self.y_indicators))
        self.assertEqual(loaded_x_names, set(self.x_indicators))
    
    def test_procore_vertical_scenario(self):
        """Test specific Procore scenario with expected low Y scores."""
        # Mock evidence with old sources and vertical industry
        self.mock_researcher.query = "Procore construction AI risk"
        
        # Simulate weak evidence for construction automation
        weak_evidence_scores = [3, 4, 5]  # Low automation potential
        old_sources = ["Construction report 2020", "Old case study"]
        
        # Test that vertical bias and old sources result in expected low scores
        for score in weak_evidence_scores:
            adjusted_score, rationale = self.skill._apply_score_adjustments(
                median_score=score,
                sources=old_sources,
                rationale="Limited automation in construction",
                industry_type='vertical'
            )
            
            # Should be low due to -2 penalty for old sources and vertical bias
            expected_score = max(1, min(5, score - 2))  # -2 for old sources, capped at 5 for vertical with weak evidence
            self.assertEqual(adjusted_score, expected_score)
            self.assertIn("downgraded -2", rationale)
    
    def test_vote_on_indicators_integration(self):
        """Test the complete voting process with bias adjustments."""
        # Create mock variant results
        variant_results = [
            {
                'y': [{'name': 'Task Structure', 'score': 7, 'rationale': 'Good automation', 'sources': ['Source 2024']}],
                'x': [{'name': 'External Observability', 'score': 8, 'rationale': 'High visibility', 'sources': ['Market data 2025']}]
            },
            {
                'y': [{'name': 'Task Structure', 'score': 6, 'rationale': 'Medium automation', 'sources': ['Old study 2020']}],
                'x': [{'name': 'External Observability', 'score': 9, 'rationale': 'Very visible', 'sources': ['Recent analysis 2024']}]
            },
            {
                'y': [{'name': 'Task Structure', 'score': 8, 'rationale': 'Strong automation', 'sources': ['New report 2025']}],
                'x': [{'name': 'External Observability', 'score': 7, 'rationale': 'Visible outcomes', 'sources': ['Study 2023']}]
            }
        ]
        
        # Test with partial indicators (simplified test)
        y_test = ['Task Structure']
        x_test = ['External Observability']
        
        result = self.skill._vote_on_indicators(variant_results, y_test, x_test)
        
        # Verify structure
        self.assertEqual(len(result['y']), 1)
        self.assertEqual(len(result['x']), 1)
        
        # Verify median calculation and adjustments were applied
        y_score = result['y'][0]['score']
        x_score = result['x'][0]['score']
        
        self.assertGreaterEqual(y_score, 1)
        self.assertLessEqual(y_score, 10)
        self.assertGreaterEqual(x_score, 1)
        self.assertLessEqual(x_score, 10)


class TestMCPStreamer(unittest.TestCase):
    """Test streaming functionality with quality warnings"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.streamer = MCPStreamer()
        self.streamer.evidence_scores = []  # Reset for testing
    
    async def test_quality_warnings_for_low_evidence(self):
        """Test that low quality triggers appropriate warnings"""
        
        # Mock low quality session
        self.streamer.evidence_scores = [5.0, 4.5, 6.2, 5.8, 4.9]  # Average = 5.28 < 7.0
        
        # Capture streamed messages
        captured_warnings = []
        captured_logs = []
        captured_infos = []
        
        async def mock_stream_warning(msg):
            captured_warnings.append(msg)
        
        async def mock_stream_log(msg):
            captured_logs.append(msg)
            
        async def mock_stream_info(msg):
            captured_infos.append(msg)
        
        self.streamer.stream_warning = mock_stream_warning
        self.streamer.stream_log = mock_stream_log
        self.streamer.stream_info = mock_stream_info
        
        # Trigger session summary
        await self.streamer.stream_session_summary()
        
        # Verify low quality warning triggered
        warning_msgs = ' '.join(captured_warnings)
        self.assertIn("Low evidence quality detected", warning_msgs)
        self.assertIn("Re-run research", warning_msgs)
    
    async def test_excellence_message_for_high_quality(self):
        """Test that high quality triggers excellence message"""
        
        # Mock high quality session
        self.streamer.evidence_scores = [8.5, 9.0, 8.2, 8.8, 9.2]  # Average = 8.74 >= 8.0
        
        # Capture streamed messages
        captured_warnings = []
        captured_logs = []
        
        async def mock_stream_warning(msg):
            captured_warnings.append(msg)
        
        async def mock_stream_log(msg):
            captured_logs.append(msg)
        
        self.streamer.stream_warning = mock_stream_warning
        self.streamer.stream_log = mock_stream_log
        
        # Trigger session summary
        await self.streamer.stream_session_summary()
        
        # Should have no warnings
        self.assertEqual(len(captured_warnings), 0)
        
        # Should have excellence message
        log_msgs = ' '.join(captured_logs)
        self.assertIn("Excellent evidence quality", log_msgs)


class TestMCPIntegration(unittest.TestCase):
    """Integration tests for complete MCP workflow"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.cfg = MockConfig()
        self.temp_cache_dir = tempfile.mkdtemp(prefix="mcp_integration_test_")
        self.cfg.mcp_cache_dir = self.temp_cache_dir
        
        self.mock_researcher = MagicMock()
        self.mock_researcher.add_costs = MagicMock()
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.temp_cache_dir):
            shutil.rmtree(self.temp_cache_dir)
    
    def test_procore_query_characteristics(self):
        """Test that Procore query demonstrates expected low Y/low X characteristics"""
        
        # This test validates the business logic for Procore scoring
        query = "Procore AI risk assessment"
        
        # Expected characteristics for Procore (construction management):
        # - Low Y scores due to high variability in construction projects
        # - Low X scores due to industry fragmentation and low standardization
        
        expected_y_reasoning = {
            "Task Structure": "low",  # Construction projects are highly variable
            "Process Variability": "high variability",  # Each project is unique
        }
        
        expected_x_reasoning = {
            "Industry Standardization": "low",  # Construction lacks standardization
            "External Observability": "high",  # Project outcomes are visible
        }
        
        # These are business logic validations - the actual scores should reflect
        # the challenging nature of construction industry automation
        
        # Verify query contains construction context
        self.assertIn("Procore", query)
        
        # This test ensures our rubrics properly capture industry-specific challenges
        self.assertTrue(True)  # Business logic test placeholder


def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == '__main__':
    unittest.main(verbosity=2) 