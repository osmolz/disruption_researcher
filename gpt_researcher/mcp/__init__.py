"""
MCP (Model Context Protocol) Integration Module for GPT-Researcher

This module provides comprehensive MCP integration with the following optimizations:
- YC-style rubric evaluation for Y/X disruption indicators 
- Self-consistency tool selection with 3-variant voting
- Parallel tool execution with asyncio.gather
- Evidence quality scoring and cost tracking
- Fallback mechanisms for environments without MCP adapters
- Enhanced streaming with detailed progress and metrics
- Tool selection caching for improved performance
- Configuration validation for robust integration

Components:
- MCPClientManager: Handles MCP client lifecycle with explicit cleanup
- MCPToolSelector: Intelligent tool selection with rubric-based evaluation and caching
- MCPResearchSkill: Parallel research execution with evidence scoring
- MCPStreamer: Enhanced streaming with cost and quality metrics
- clean_text: Text cleaning utility for Y/X framework analysis
"""

import logging

logger = logging.getLogger(__name__)

def validate_mcp_config(cfg, component_name: str = "MCP"):
    """
    Validate MCP configuration for any component.
    
    Args:
        cfg: Configuration object to validate
        component_name: Name of the component for error messages
        
    Raises:
        ValueError: If required configuration is missing or invalid
    """
    if not cfg:
        raise ValueError(f"{component_name}: Configuration object cannot be None")
    
    # Core required attributes for MCP functionality
    core_required = ['strategic_llm_model', 'strategic_llm_provider']
    
    # Check for core requirements
    missing_core = []
    for attr in core_required:
        if not hasattr(cfg, attr):
            missing_core.append(attr)
        elif not getattr(cfg, attr):  # Check for empty values
            missing_core.append(f"{attr} (empty)")
    
    if missing_core:
        raise ValueError(f"{component_name}: Missing core configuration: {missing_core}")
    
    # Check for llm_kwargs (can be empty dict but should exist)
    if not hasattr(cfg, 'llm_kwargs'):
        logger.warning(f"{component_name}: llm_kwargs not found, using empty dict")
        cfg.llm_kwargs = {}
    
    # Optional attributes with defaults
    if not hasattr(cfg, 'fast_llm_model'):
        cfg.fast_llm_model = cfg.strategic_llm_model
        logger.debug(f"{component_name}: Using strategic_llm_model for fast_llm_model")
        
    if not hasattr(cfg, 'fast_llm_provider'):
        cfg.fast_llm_provider = cfg.strategic_llm_provider
        logger.debug(f"{component_name}: Using strategic_llm_provider for fast_llm_provider")
    
    # Cache directory for tool selection
    if not hasattr(cfg, 'mcp_cache_dir'):
        cfg.mcp_cache_dir = 'mcp_cache'
        logger.debug(f"{component_name}: Using default cache directory: mcp_cache")
    
    logger.debug(f"{component_name}: Configuration validation passed")

# Check for MCP adapters availability
HAS_MCP_ADAPTERS = False
try:
    import langchain_mcp_adapters
    HAS_MCP_ADAPTERS = True
except ImportError:
    pass

# Expose main components and utilities
if HAS_MCP_ADAPTERS:
    from .client import MCPClientManager, call_llm_fallback
    from .tool_selector import MCPToolSelector  
    from .research import MCPResearchSkill, clean_text
    from .streaming import MCPStreamer
    
    __all__ = [
        "MCPClientManager", 
        "MCPToolSelector",
        "MCPResearchSkill", 
        "MCPStreamer",
        "clean_text",
        "call_llm_fallback",
        "validate_mcp_config",
        "HAS_MCP_ADAPTERS"
    ]
else:
    # Provide fallback implementations when MCP adapters not available
    from .client import call_llm_fallback
    from .research import clean_text
    
    __all__ = [
        "clean_text",
        "call_llm_fallback", 
        "validate_mcp_config",
        "HAS_MCP_ADAPTERS"
    ]

# Module metadata for the AI Disruption Analyzer
__version__ = "1.1.0"  # Updated for caching and validation improvements
__author__ = "MCP Integration Optimizer"
__description__ = "Enhanced MCP integration with Y/X rubrics, self-consistency, parallel processing, caching, and validation"

# Configuration constants for disruption analysis
DISRUPTION_RUBRIC_WEIGHTS = {
    "y_axis": 0.4,  # Performance/capability improvements
    "x_axis": 0.4,  # Market adoption/ecosystem changes  
    "quality": 0.2  # Source credibility and recency
}

EVIDENCE_QUALITY_THRESHOLDS = {
    "minimum_selection": 7.0,   # Minimum score for tool selection
    "high_quality": 8.0,        # High quality evidence threshold
    "acceptable": 5.0           # Acceptable quality threshold
}

SELF_CONSISTENCY_CONFIG = {
    "variant_count": 3,         # Number of variants for voting
    "consensus_threshold": 0.5   # Minimum consensus for selection
}

# Caching configuration
CACHE_CONFIG = {
    "default_dir": "mcp_cache",    # Default cache directory
    "max_age_hours": 24,           # Cache expiry (24 hours)
    "max_cache_files": 100         # Maximum cached selections
} 