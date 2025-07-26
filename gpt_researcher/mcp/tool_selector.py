"""
MCP Tool Selection Module

Handles intelligent tool selection using LLM analysis with rubric-based evaluation.
"""
import asyncio
import json
import logging
import statistics
import hashlib
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MCPToolSelector:
    """
    Handles intelligent selection of MCP tools using LLM analysis with YC-style rubrics.
    
    Responsible for:
    - Analyzing available tools with LLM using disruption-focused rubrics
    - Selecting the most relevant tools for a query using self-consistency
    - Providing fallback selection mechanisms
    - Tracking costs and evaluation metrics
    - Caching tool selections for efficiency
    """

    def __init__(self, cfg, researcher=None):
        """
        Initialize the tool selector with configuration validation.
        
        Args:
            cfg: Configuration object with LLM settings
            researcher: Researcher instance for cost tracking
        """
        # Validate required configuration attributes
        self._validate_config(cfg)
        
        self.cfg = cfg
        self.researcher = researcher
        
        # Setup cache directory
        self.cache_dir = getattr(cfg, 'mcp_cache_dir', 'mcp_cache')
        self._ensure_cache_dir()

    def _validate_config(self, cfg):
        """
        Validate that the configuration object has all required attributes.
        
        Args:
            cfg: Configuration object to validate
            
        Raises:
            ValueError: If required configuration is missing
        """
        required_attrs = [
            'strategic_llm_model',
            'strategic_llm_provider', 
            'llm_kwargs'
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(cfg, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            raise ValueError(f"Missing required configuration attributes: {missing_attrs}")
        
        # Validate that strategic_llm_model is not empty
        if not cfg.strategic_llm_model:
            raise ValueError("strategic_llm_model cannot be empty")
            
        logger.debug("Configuration validation passed")

    def _ensure_cache_dir(self):
        """Ensure the cache directory exists."""
        try:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.debug(f"Created cache directory: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Could not create cache directory {self.cache_dir}: {e}")
            # Fallback to no caching
            self.cache_dir = None

    def _get_cache_key(self, query: str, max_tools: int) -> str:
        """
        Generate a cache key for the query and parameters.
        
        Args:
            query: Research query
            max_tools: Maximum number of tools to select
            
        Returns:
            str: MD5 hash of the query and parameters
        """
        # Include max_tools in cache key to handle different selection sizes
        cache_input = f"{query}|max_tools={max_tools}"
        return hashlib.md5(cache_input.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full cache file path."""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _load_cached_selection(self, query: str, max_tools: int) -> Optional[List]:
        """
        Load cached tool selection if available.
        
        Args:
            query: Research query
            max_tools: Maximum number of tools
            
        Returns:
            Optional[List]: Cached tool selection or None if not found/invalid
        """
        if not self.cache_dir:
            return None
            
        try:
            cache_key = self._get_cache_key(query, max_tools)
            cache_path = self._get_cache_path(cache_key)
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Validate cache structure
                if isinstance(cached_data, dict) and 'tools' in cached_data and 'metadata' in cached_data:
                    logger.info(f"Loaded cached tool selection for query (cache key: {cache_key[:8]}...)")
                    return cached_data['tools']
                else:
                    logger.warning(f"Invalid cache structure in {cache_path}, ignoring")
                    
        except Exception as e:
            logger.warning(f"Error loading cached selection: {e}")
            
        return None

    def _save_cached_selection(self, query: str, max_tools: int, selected_tools: List):
        """
        Save tool selection to cache.
        
        Args:
            query: Research query
            max_tools: Maximum number of tools
            selected_tools: Selected tools to cache
        """
        if not self.cache_dir or not selected_tools:
            return
            
        try:
            cache_key = self._get_cache_key(query, max_tools)
            cache_path = self._get_cache_path(cache_key)
            
            # Serialize tools (store names only since tool objects aren't JSON serializable)
            tool_names = []
            for tool in selected_tools:
                if hasattr(tool, 'name'):
                    tool_names.append(tool.name)
                else:
                    tool_names.append(str(tool))
            
            cached_data = {
                'tools': tool_names,
                'metadata': {
                    'query': query,
                    'max_tools': max_tools,
                    'timestamp': __import__('time').time(),
                    'tool_count': len(tool_names)
                }
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, indent=2)
                
            logger.debug(f"Cached tool selection: {len(tool_names)} tools (cache key: {cache_key[:8]}...)")
            
        except Exception as e:
            logger.warning(f"Error saving selection to cache: {e}")

    def _match_cached_tools_to_available(self, cached_tool_names: List[str], all_tools: List) -> List:
        """
        Match cached tool names to available tool objects.
        
        Args:
            cached_tool_names: List of cached tool names
            all_tools: List of available tool objects
            
        Returns:
            List: Matched tool objects
        """
        matched_tools = []
        
        for cached_name in cached_tool_names:
            # Find matching tool in available tools
            for tool in all_tools:
                tool_name = tool.name if hasattr(tool, 'name') else str(tool)
                if tool_name == cached_name:
                    matched_tools.append(tool)
                    break
            else:
                logger.warning(f"Cached tool '{cached_name}' not found in available tools")
        
        logger.info(f"Matched {len(matched_tools)}/{len(cached_tool_names)} cached tools to available tools")
        return matched_tools

    async def select_relevant_tools(self, query: str, all_tools: List, max_tools: int = 3) -> List:
        """
        Use LLM with self-consistency and Y/X rubrics to select the most relevant tools with caching.
        
        Args:
            query: Research query focused on disruption indicators
            all_tools: List of all available tools
            max_tools: Maximum number of tools to select (default: 3)
            
        Returns:
            List: Selected tools most relevant for the query
        """
        if not all_tools:
            return []

        if len(all_tools) < max_tools:
            max_tools = len(all_tools)
        
        # *** NEW: Try to load from cache first ***
        cached_tools = self._load_cached_selection(query, max_tools)
        if cached_tools:
            # Match cached tool names to available tool objects
            matched_tools = self._match_cached_tools_to_available(cached_tools, all_tools)
            if len(matched_tools) >= min(max_tools, len(cached_tools)):
                logger.info(f"Using cached tool selection: {len(matched_tools)} tools")
                return matched_tools[:max_tools]
            else:
                logger.warning("Cached tools partially unavailable, proceeding with fresh selection")
            
        logger.info(f"Using LLM with self-consistency to select {max_tools} most relevant tools from {len(all_tools)} available")
        
        # Create tool descriptions for LLM analysis
        tools_info = []
        for i, tool in enumerate(all_tools):
            tool_info = {
                "index": i,
                "name": tool.name,
                "description": tool.description or "No description available"
            }
            tools_info.append(tool_info)
        
        # Generate rubric-based prompt for disruption analysis
        prompt_base = self._generate_disruption_rubric_prompt(query, tools_info, max_tools)

        try:
            # Self-consistency: Generate 3 variants and vote
            selections = []
            
            for variant_idx in range(3):
                variant_prompt = f"{prompt_base}\n\n### VARIANT {variant_idx + 1} INSTRUCTIONS:\nProvide your selection focusing on variant perspective {variant_idx + 1} of disruption analysis. Consider different angles of evidence gathering for Y/X indicators."
                
                logger.debug(f"Running tool selection variant {variant_idx + 1}/3")
                response = await self._call_llm_for_tool_selection(variant_prompt)
                
                if response:
                    parsed_selection = self._parse_tool_selection_response(response, all_tools)
                    if parsed_selection:
                        selections.append(parsed_selection)
                        logger.debug(f"Variant {variant_idx + 1} selected {len(parsed_selection)} tools")
            
            if not selections:
                logger.warning("No valid selections from any variant, using fallback")
                selected_tools = self._fallback_tool_selection(all_tools, max_tools)
            else:
                # Vote on tool selections using median relevance scores
                selected_tools = self._vote_on_tool_selections(selections, all_tools, max_tools)
            
            if len(selected_tools) == 0:
                logger.warning("No tools met minimum relevance threshold (>=7), using fallback")
                selected_tools = self._fallback_tool_selection(all_tools, max_tools)

            # *** NEW: Save successful selection to cache ***
            if selected_tools:
                self._save_cached_selection(query, max_tools, selected_tools)
            
            logger.info(f"Tool selection completed: {len(selected_tools)} tools selected")
            return selected_tools
            
        except Exception as e:
            logger.error(f"Error in LLM tool selection: {e}")
            logger.warning("Falling back to pattern-based selection")
            fallback_tools = self._fallback_tool_selection(all_tools, max_tools)
            
            # Cache fallback selection too
            if fallback_tools:
                self._save_cached_selection(query, max_tools, fallback_tools)
                
            return fallback_tools

    def _generate_disruption_rubric_prompt(self, query: str, tools_info: List[Dict], max_tools: int) -> str:
        """
        Generate YC-style rubric prompt for disruption-focused tool selection.
        
        Args:
            query: Research query
            tools_info: List of tool information
            max_tools: Maximum tools to select
            
        Returns:
            str: Rubric-based prompt
        """
        tools_list = "\n".join([
            f"{i+1}. **{tool['name']}** (index: {tool['index']})\n   Description: {tool['description']}"
            for i, tool in enumerate(tools_info)
        ])
        
        return f"""<role>
You are an expert Y Combinator-style evaluator specializing in disruption research for AI/technology analysis.
</role>

<plan>
Evaluate each tool for relevance to disruption indicators using strict rubrics. Select {max_tools} tools that best enable gathering evidence for Y/X framework analysis.
</plan>

<guidelines>
- Focus on tools that can surface recent (2023-2025) trusted sources
- Prioritize tools that access technical, market, or trend data
- Ensure tools can provide quantitative evidence for disruption indicators
- Reject tools with relevance scores below 7/10
</guidelines>

<few_shot>
**High-Value Examples (Score ≥ 7.0):**
- web_search: Y:9 (finds performance benchmarks, technical papers), X:8 (market trends, adoption data), Quality:9 (cite McKinsey 2024, MIT reports) → 8.6/10 ✅
- browse_page: Y:8 (deep technical analysis), X:7 (company case studies), Quality:8 (official sources) → 7.6/10 ✅
- Monday.com analysis: Y:7 (workflow automation metrics), X:8 (high horizontal SaaS penetration, cite Gartner 2024), Quality:8 (authoritative reports) → 7.4/10 ✅

**Medium-Value Examples (Score 5.0-6.9):**
- company_api: Y:6 (moderate automation insights), X:5 (limited market context), Quality:7 (official but narrow) → 5.8/10 ⚠️
- news_search: Y:5 (general trends), X:6 (adoption signals), Quality:6 (mixed source quality) → 5.4/10 ⚠️

**Low-Value Examples (Score < 5.0):**
- Procore vertical analysis: Y:4 (low automation potential due to construction variability, cite Bain 2025), X:6 (moderate penetration), Quality:5 (limited recent data) → 4.8/10 ❌
- random_facts: Y:2 (no performance data), X:1 (no market insights), Quality:3 (unreliable) → 1.8/10 ❌
- weather_api: Y:1 (irrelevant to disruption), X:1 (no business context), Quality:5 (accurate but irrelevant) → 2.0/10 ❌
</few_shot>

<chain_of_thought>
**Step 1: Score Y-Axis Evidence (40% weight)**
- Can this tool find quantitative performance improvements, efficiency gains, or capability advances?
- Look for: benchmarks, metrics, technical specifications, automation potential
- Score 1-10 based on evidence quality and relevance

**Step 2: Score X-Axis Evidence (40% weight)**  
- Can this tool identify market trends, adoption patterns, or ecosystem changes?
- Look for: market data, user growth, enterprise adoption, industry reports
- Score 1-10 based on market insight depth and recency

**Step 3: Score Source Quality (20% weight)**
- Does this tool access trusted, authoritative, recent (2023-2025) sources?
- Look for: academic papers, industry reports, official announcements, reputable analysts
- Score 1-10 based on source credibility and timeliness

**Step 4: Calculate Weighted Average**
- Overall Score = (Y_Score × 0.4) + (X_Score × 0.4) + (Quality_Score × 0.2)
- Minimum threshold: 7.0 to be selected
</chain_of_thought>

<rubric>
For each tool, score 1-10 on:
1. **Y-Axis Relevance**: Can this tool find evidence of performance improvements, efficiency gains, or capability advances? (Weight: 40%)
2. **X-Axis Relevance**: Can this tool identify market trends, adoption patterns, or ecosystem changes? (Weight: 40%) 
3. **Source Quality**: Does this tool access trusted, recent (2023-2025), authoritative sources? (Weight: 20%)

Minimum threshold: Overall score >= 7.0 to be selected
</rubric>

**Research Query:** "{query}"

**Available Tools:**
{tools_list}

<xml_output>
Provide your analysis in this exact JSON format:
{{
    "selected_tools": [
        {{
            "index": <tool_index>,
            "name": "<tool_name>",
            "y_axis_score": <1-10>,
            "x_axis_score": <1-10>,
            "quality_score": <1-10>,
            "overall_score": <calculated_weighted_average>,
            "evidence_reasoning": "<why this tool helps gather Y/X evidence>"
        }}
    ],
    "selection_reasoning": "<overall strategy for tool combination>",
    "rejected_tools": [
        {{
            "name": "<tool_name>",
            "overall_score": <score>,
            "rejection_reason": "<why score was too low>"
        }}
    ]
}}
</xml_output>"""

    def _parse_tool_selection_response(self, response: str, all_tools: List) -> Optional[List[Dict]]:
        """
        Parse LLM response for tool selection with score validation.
        
        Args:
            response: LLM response string
            all_tools: List of all available tools
            
        Returns:
            Optional[List[Dict]]: Parsed tool selections with scores or None if invalid
        """
        try:
            # Try to parse as JSON
            try:
                selection_result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r"\{.*\}", response, re.DOTALL)
                if json_match:
                    selection_result = json.loads(json_match.group(0))
                else:
                    return None
            
            parsed_tools = []
            
            for tool_selection in selection_result.get("selected_tools", []):
                tool_index = tool_selection.get("index")
                overall_score = tool_selection.get("overall_score", 0)
                
                # Validate score threshold
                if overall_score < 7.0:
                    logger.debug(f"Tool {tool_selection.get('name', 'unknown')} rejected with score {overall_score} < 7.0")
                    continue
                
                if tool_index is not None and 0 <= tool_index < len(all_tools):
                    tool_data = {
                        'tool': all_tools[tool_index],
                        'score': overall_score,
                        'y_score': tool_selection.get("y_axis_score", 0),
                        'x_score': tool_selection.get("x_axis_score", 0),
                        'quality_score': tool_selection.get("quality_score", 0),
                        'reasoning': tool_selection.get("evidence_reasoning", "")
                    }
                    parsed_tools.append(tool_data)
                    
            return parsed_tools if parsed_tools else None
            
        except Exception as e:
            logger.error(f"Error parsing tool selection response: {e}")
            return None

    def _vote_on_tool_selections(self, selections: List[List[Dict]], all_tools: List, max_tools: int) -> List:
        """
        Vote on tool selections using median scores across variants.
        
        Args:
            selections: List of parsed tool selections from each variant
            all_tools: List of all available tools  
            max_tools: Maximum tools to select
            
        Returns:
            List: Final selected tools based on voting
        """
        # Aggregate scores for each tool across variants
        tool_scores = {}
        
        for selection in selections:
            for tool_data in selection:
                tool_name = tool_data['tool'].name
                if tool_name not in tool_scores:
                    tool_scores[tool_name] = {
                        'tool': tool_data['tool'],
                        'scores': [],
                        'reasonings': []
                    }
                
                tool_scores[tool_name]['scores'].append(tool_data['score'])
                tool_scores[tool_name]['reasonings'].append(tool_data['reasoning'])
        
        # Calculate median scores and select top tools
        final_tools_data = []
        
        for tool_name, data in tool_scores.items():
            median_score = statistics.median(data['scores'])
            
            # Only include tools with median score >= 7.0
            if median_score >= 7.0:
                final_tools_data.append({
                    'tool': data['tool'],
                    'median_score': median_score,
                    'vote_count': len(data['scores']),
                    'reasonings': data['reasonings']
                })
                
                logger.info(f"Tool '{tool_name}' selected with median score {median_score:.1f} ({len(data['scores'])} votes)")
        
        # Sort by median score and take top tools
        final_tools_data.sort(key=lambda x: x['median_score'], reverse=True)
        selected_tools = [item['tool'] for item in final_tools_data[:max_tools]]
        
        # Log selection reasoning
        for i, item in enumerate(final_tools_data[:max_tools]):
            logger.info(f"Final tool {i+1}: {item['tool'].name} (score: {item['median_score']:.1f}, consensus: {item['vote_count']}/3)")
        
        return selected_tools

    async def _call_llm_for_tool_selection(self, prompt: str) -> str:
        """
        Call the LLM using the existing create_chat_completion function for tool selection with retries.
        
        Args:
            prompt (str): The prompt to send to the LLM.
            
        Returns:
            str: The generated text response.
        """
        if not self.cfg:
            logger.warning("No config available for LLM call")
            return ""
        
        # Retry with exponential backoff
        for attempt in range(3):
            try:
                # Try to use existing LLM utilities first
                try:
                    from ..utils.llm import create_chat_completion
                    
                    # Create messages for the LLM
                    messages = [{"role": "user", "content": prompt}]
                    
                    # Use the strategic LLM for tool selection (as it's more complex reasoning)
                    result = await create_chat_completion(
                        model=self.cfg.strategic_llm_model,
                        messages=messages,
                        temperature=0.0,  # Low temperature for consistent tool selection
                        llm_provider=self.cfg.strategic_llm_provider,
                        llm_kwargs=self.cfg.llm_kwargs,
                        cost_callback=self.researcher.add_costs if self.researcher and hasattr(self.researcher, 'add_costs') else None,
                    )
                    return result
                    
                except ImportError:
                    # Fallback to direct client call if utils not available
                    from .client import call_llm_fallback
                    
                    result = await call_llm_fallback(
                        prompt=prompt,
                        temperature=0.0,
                        max_tokens=1500,
                        cost_callback=self.researcher.add_costs if self.researcher and hasattr(self.researcher, 'add_costs') else None
                    )
                    return result
                    
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1}/3 failed: {e}")
                
                # If this isn't the last attempt, wait with exponential backoff
                if attempt < 2:
                    backoff_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.info(f"Retrying in {backoff_time} seconds...")
                    await asyncio.sleep(backoff_time)
                else:
                    # Last attempt failed, raise the exception
                    logger.error("All LLM retry attempts failed for tool selection")
                    raise Exception(f"LLM failed after 3 attempts: {e}")
        
        # This should never be reached, but included for completeness
        return ""

    def _fallback_tool_selection(self, all_tools: List, max_tools: int) -> List:
        """
        Fallback tool selection using pattern matching if LLM selection fails.
        Enhanced with disruption-focused patterns.
        
        Args:
            all_tools: List of all available tools
            max_tools: Maximum number of tools to select
            
        Returns:
            List: Selected tools
        """
        # Define patterns for disruption research-relevant tools
        disruption_patterns = [
            # High-value patterns (score 5)
            ('search', 5), ('web_search', 5), ('browse', 5),
            # Analysis patterns (score 4) 
            ('analyze', 4), ('trend', 4), ('market', 4),
            # Data gathering patterns (score 3)
            ('get', 3), ('read', 3), ('fetch', 3), ('find', 3), 
            ('list', 3), ('query', 3), ('lookup', 3), ('retrieve', 3),
            # General patterns (score 2)
            ('view', 2), ('show', 2), ('describe', 2)
        ]
        
        scored_tools = []
        
        for tool in all_tools:
            tool_name = tool.name.lower()
            tool_description = (tool.description or "").lower()
            
            # Calculate relevance score based on pattern matching
            score = 0
            matched_patterns = []
            
            for pattern, pattern_score in disruption_patterns:
                if pattern in tool_name:
                    score += pattern_score * 2  # Name matches are more important
                    matched_patterns.append(f"name:{pattern}")
                if pattern in tool_description:
                    score += pattern_score
                    matched_patterns.append(f"desc:{pattern}")
            
            if score > 0:
                scored_tools.append((tool, score, matched_patterns))
        
        # Sort by score and take top tools
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        selected_tools = [tool for tool, score, patterns in scored_tools[:max_tools]]
        
        logger.warning("Using fallback tool selection - results may be suboptimal for disruption analysis")
        for i, (tool, score, patterns) in enumerate(scored_tools[:max_tools]):
            logger.info(f"Fallback selected tool {i+1}: {tool.name} (score: {score}, patterns: {patterns})")
        
        return selected_tools 