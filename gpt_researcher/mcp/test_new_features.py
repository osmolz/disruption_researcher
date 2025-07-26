"""
Test file for new MCP research functionality focused on Y/X indicators.
"""
import unittest
import json
import os
import tempfile
import sys
from unittest.mock import Mock


# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from research import MCPResearchSkill
except ImportError:
    print("Warning: Could not import MCPResearchSkill - some tests may be skipped")
    MCPResearchSkill = None


class MockCfg:
    """Mock configuration for testing."""
    def __init__(self):
        self.strategic_llm_model = "gpt-4"
        self.strategic_llm_provider = "openai"
        self.fast_llm_model = "gpt-3.5-turbo"
        self.fast_llm_provider = "openai"
        self.llm_kwargs = {}
        self.mcp_cache_dir = tempfile.mkdtemp()


class TestNewMCPFeatures(unittest.TestCase):
    """Test new MCP features: fallbacks, bias caps, caching, improved prompts."""
    
    def setUp(self):
        """Set up test fixtures."""
        if MCPResearchSkill is None:
            self.skipTest("MCPResearchSkill not available")
            
        self.cfg = MockCfg()
        self.mock_researcher = Mock()
        self.mock_researcher.query = "Procore AI disruption risk"
        self.mock_researcher.add_costs = Mock()
        
        self.skill = MCPResearchSkill(self.cfg, self.mock_researcher)
        self.skill.researcher = self.mock_researcher
        
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
        if hasattr(self.cfg, 'mcp_cache_dir') and os.path.exists(self.cfg.mcp_cache_dir):
            shutil.rmtree(self.cfg.mcp_cache_dir)
    
    def test_indicator_structure_enforcement(self):
        """Test that indicator mapping returns exactly 6 Y and 6 X indicators."""
        test_mapping = self.skill._create_fallback_indicators(self.y_indicators, self.x_indicators)
        
        # Verify structure
        self.assertEqual(len(test_mapping['y']), 6, "Must have exactly 6 Y indicators")
        self.assertEqual(len(test_mapping['x']), 6, "Must have exactly 6 X indicators")
        
        # Verify all indicators are present
        y_names = {ind['name'] for ind in test_mapping['y']}
        x_names = {ind['name'] for ind in test_mapping['x']}
        
        self.assertEqual(y_names, set(self.y_indicators))
        self.assertEqual(x_names, set(self.x_indicators))
        
        # Verify score ranges (1-10)
        for indicator_type in ['y', 'x']:
            for ind in test_mapping[indicator_type]:
                self.assertGreaterEqual(ind['score'], 1, f"Score too low: {ind['score']}")
                self.assertLessEqual(ind['score'], 10, f"Score too high: {ind['score']}")
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
    
    def test_recent_source_penalty(self):
        """Test score downgrade for lack of recent sources."""
        # Test with recent sources (no penalty)
        recent_sources = ["MIT 2024 study", "Gartner 2025 report"]
        score, rationale = self.skill._apply_score_adjustments(
            median_score=8.0, 
            sources=recent_sources, 
            rationale="Good evidence", 
            industry_type='horizontal'
        )
        self.assertEqual(score, 8.0, "Recent sources should not be penalized")
        self.assertNotIn("downgraded", rationale)
        
        # Test with old sources (-2 penalty)
        old_sources = ["Old study 2020", "Report 2019"]
        score, rationale = self.skill._apply_score_adjustments(
            median_score=8.0, 
            sources=old_sources, 
            rationale="Old evidence", 
            industry_type='horizontal'
        )
        self.assertEqual(score, 6.0, "Old sources should get -2 penalty")
        self.assertIn("downgraded -2", rationale)
    
    def test_vertical_bias_cap(self):
        """Test bias cap for vertical industries with insufficient evidence."""
        # Test vertical bias cap
        insufficient_sources = ["One source 2024"]
        score, rationale = self.skill._apply_score_adjustments(
            median_score=8.0, 
            sources=insufficient_sources, 
            rationale="Limited evidence", 
            industry_type='vertical'
        )
        self.assertEqual(score, 5, "Vertical industries should be capped at 5 with insufficient evidence")
        self.assertIn("capped at 5", rationale)
        
        # Test vertical with sufficient sources (no cap)
        sufficient_sources = ["Source 1 2024", "Source 2 2025"]
        score, rationale = self.skill._apply_score_adjustments(
            median_score=8.0, 
            sources=sufficient_sources, 
            rationale="Good evidence", 
            industry_type='vertical'
        )
        self.assertEqual(score, 8.0, "Sufficient evidence should not be capped")
        self.assertNotIn("capped at 5", rationale)
    
    def test_insufficient_evidence_fallback(self):
        """Test fallback rationale for insufficient evidence."""
        insufficient_sources = ["Only one source"]
        score, rationale = self.skill._apply_score_adjustments(
            median_score=7.0, 
            sources=insufficient_sources, 
            rationale="Weak evidence", 
            industry_type='horizontal'
        )
        self.assertEqual(score, 5, "Insufficient evidence should default to neutral score 5")
        self.assertIn("Insufficient evidence: Score neutral 5, re-query needed", rationale)
    
    def test_score_range_clamping(self):
        """Test that scores are always kept in valid 1-10 range."""
        # Test extreme low score
        sources = ["No recent sources"]
        score, _ = self.skill._apply_score_adjustments(
            median_score=2.0,
            sources=sources, 
            rationale="", 
            industry_type='horizontal'
        )
        self.assertGreaterEqual(score, 1, "Score should not go below 1")
        
        # Test extreme high score
        good_sources = ["Great source 2024", "Another 2025"]
        score, _ = self.skill._apply_score_adjustments(
            median_score=15.0,  # Impossible high score
            sources=good_sources, 
            rationale="", 
            industry_type='horizontal'
        )
        self.assertLessEqual(score, 10, "Score should not exceed 10")
    
    def test_caching_functionality(self):
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
        self.assertTrue(os.path.exists(cache_path), "Cache file should be created")
        
        # Test loading from cache
        loaded_indicators = self.skill._load_cached_indicators(query)
        self.assertIsNotNone(loaded_indicators, "Should load cached indicators")
        self.assertEqual(len(loaded_indicators['y']), 6, "Cached Y indicators should be 6")
        self.assertEqual(len(loaded_indicators['x']), 6, "Cached X indicators should be 6")
        
        # Verify indicator names match
        loaded_y_names = {ind['name'] for ind in loaded_indicators['y']}
        loaded_x_names = {ind['name'] for ind in loaded_indicators['x']}
        self.assertEqual(loaded_y_names, set(self.y_indicators))
        self.assertEqual(loaded_x_names, set(self.x_indicators))
    
    def test_procore_scenario_simulation(self):
        """Test specific Procore scenario with expected characteristics."""
        # Simulate Procore (vertical) with weak evidence
        self.mock_researcher.query = "Procore construction AI risk"
        
        weak_evidence_scores = [3, 4, 5]  # Low automation potential
        old_sources = ["Construction report 2020", "Old case study"]
        
        for score in weak_evidence_scores:
            adjusted_score, rationale = self.skill._apply_score_adjustments(
                median_score=score,
                sources=old_sources,
                rationale="Limited automation in construction",
                industry_type='vertical'
            )
            
            # Should be low due to penalties and vertical bias
            expected_score = max(1, min(5, score - 2))  # -2 for old sources, cap at 5 for vertical
            self.assertEqual(adjusted_score, expected_score, 
                           f"Procore scenario: score {score} should become {expected_score}")
            self.assertIn("downgraded -2", rationale, "Should mention source penalty")


if __name__ == '__main__':
    print("Running tests for new MCP features...")
    unittest.main(verbosity=2) 