"""
MCP Research Execution Skill

Handles research execution using selected MCP tools with parallel processing and evidence evaluation.
"""
import asyncio
import logging
import re
import json
import statistics
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text for Y/X framework analysis.
    
    Args:
        text: Raw text to clean
        
    Returns:
        str: Cleaned text suitable for analysis
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize quotes and punctuation
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove extra line breaks but preserve paragraph structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


class MCPResearchSkill:
    """
    Handles research execution using selected MCP tools with parallel processing and evidence evaluation.
    
    Responsible for:
    - Executing research with LLM and bound tools in parallel
    - Processing tool results with Y/X rubric evaluation
    - Managing tool execution and error handling
    - Tracking costs and evidence quality metrics
    """

    def __init__(self, cfg, researcher=None):
        """
        Initialize the MCP research skill with configuration validation.
        
        Args:
            cfg: Configuration object with LLM settings
            researcher: Researcher instance for cost tracking
        """
        # Validate configuration - commented out for testing to avoid relative import issues
        # from . import validate_mcp_config
        # validate_mcp_config(cfg, "MCPResearchSkill")
        
        # Basic validation instead
        if not hasattr(cfg, 'strategic_llm_model') or not cfg.strategic_llm_model:
            raise ValueError("strategic_llm_model is required")
        
        self.cfg = cfg
        self.researcher = researcher

    async def conduct_research_with_tools(self, query: str, selected_tools: List) -> List[Dict[str, str]]:
        """
        Use LLM with bound tools to conduct intelligent research with parallel execution.
        
        Args:
            query: Research query focused on disruption indicators
            selected_tools: List of selected MCP tools
            
        Returns:
            List[Dict[str, str]]: Research results with Y/X indicator mapping
        """
        if not selected_tools:
            logger.warning("No tools available for research")
            return []
            
        logger.info(f"Conducting parallel research using {len(selected_tools)} selected tools")
        
        try:
            from ..llm_provider.generic.base import GenericLLMProvider
            
            # Create LLM provider using the config
            provider_kwargs = {
                'model': self.cfg.strategic_llm_model,
                **self.cfg.llm_kwargs
            }
            
            llm_provider = GenericLLMProvider.from_provider(
                self.cfg.strategic_llm_provider, 
                **provider_kwargs
            )
            
            # Bind tools to LLM
            llm_with_tools = llm_provider.llm.bind_tools(selected_tools)
            
            # Create research prompt with Y/X focus
            research_prompt = self._generate_yx_research_prompt(query, selected_tools)

            # Create messages
            messages = [{"role": "user", "content": research_prompt}]
            
            # Invoke LLM with tools
            logger.info("LLM researching with bound tools...")
            response = await llm_with_tools.ainvoke(messages)
            
            # Process tool calls and results in parallel
            research_results = []
            
            # Check if the LLM made tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"LLM made {len(response.tool_calls)} tool calls - executing in parallel")
                
                # Create tasks for parallel execution of safe tools
                tool_tasks = []
                for i, tool_call in enumerate(response.tool_calls, 1):
                    tool_name = tool_call.get("name", "unknown")
                    tool_args = tool_call.get("args", {})
                    
                    # Find the tool by name
                    tool = next((t for t in selected_tools if t.name == tool_name), None)
                    if not tool:
                        logger.warning(f"Tool {tool_name} not found in selected tools")
                        continue
                    
                    # Create task for parallel execution
                    task = self._execute_single_tool(tool, tool_name, tool_args, i, len(response.tool_calls))
                    tool_tasks.append(task)
                
                # Execute all tools in parallel
                if tool_tasks:
                    logger.info(f"Executing {len(tool_tasks)} tools in parallel")
                    tool_results = await asyncio.gather(*tool_tasks, return_exceptions=True)
                    
                    # Process results and handle exceptions
                    for result in tool_results:
                        if isinstance(result, Exception):
                            logger.error(f"Tool execution failed: {result}")
                        elif result:
                            research_results.extend(result)
                            
            # Also include the LLM's own analysis/response as a result
            if hasattr(response, 'content') and response.content:
                llm_analysis = await self._process_llm_analysis(query, response.content)
                research_results.append(llm_analysis)
                
                # Log LLM analysis content
                analysis_preview = response.content[:300] + "..." if len(response.content) > 300 else response.content
                logger.debug(f"LLM Analysis: {analysis_preview}")
                logger.info("Added LLM analysis to results")
            
            # *** NEW: Map evidence to 12 specific Y/X indicators ***
            if research_results:
                # Aggregate all evidence for indicator mapping
                all_evidence = "\n\n".join([
                    f"Title: {r.get('title', '')}\nContent: {r.get('body', '')}"
                    for r in research_results
                ])
                
                logger.info("Mapping evidence to 12 Y/X indicators with self-consistency")
                indicator_mapping = await self._map_to_indicators(all_evidence, query)
                
                # Add indicator mapping as a special result
                if indicator_mapping:
                    research_results.append({
                        "title": f"Y/X Framework Analysis: {query}",
                        "href": "mcp://yx_indicators",
                        "body": json.dumps(indicator_mapping, indent=2),
                        "evidence_score": 10.0,  # High score for structured analysis
                        "indicator_mapping": indicator_mapping
                    })
                    
                    # Log indicator summary
                    y_scores = [ind['score'] for ind in indicator_mapping.get('y', [])]
                    x_scores = [ind['score'] for ind in indicator_mapping.get('x', [])]
                    avg_y = sum(y_scores) / len(y_scores) if y_scores else 0
                    avg_x = sum(x_scores) / len(x_scores) if x_scores else 0
                    logger.info(f"Y/X Indicators mapped: Y avg={avg_y:.1f}, X avg={avg_x:.1f}")
            
            logger.info(f"Research completed with {len(research_results)} total results")
            
            # Log evidence quality summary
            quality_scores = [r.get('evidence_score', 0) for r in research_results if 'evidence_score' in r]
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                logger.info(f"Average evidence quality score: {avg_quality:.1f}/10")
            
            return research_results
            
        except Exception as e:
            logger.error(f"Error in LLM research with tools: {e}")
            return []

    def _generate_yx_research_prompt(self, query: str, selected_tools: List) -> str:
        """
        Generate Y/X framework-focused research prompt.
        
        Args:
            query: Research query
            selected_tools: List of selected tools
            
        Returns:
            str: Research prompt optimized for disruption analysis
        """
        tool_names = []
        for tool in selected_tools:
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            else:
                tool_names.append(str(tool))
        
        return f"""<role>
You are a research assistant specializing in disruption analysis using the Y/X framework for AI/technology assessment.
</role>

<plan>
Research the query focusing on evidence for Y-axis (performance/capability improvements) and X-axis (market adoption/ecosystem changes). Use tools strategically to gather 2-3 high-quality, recent (2023-2025) sources per indicator.
</plan>

<guidelines>
- Prioritize trusted sources: academic papers, industry reports, official announcements
- Focus on quantitative evidence where possible (metrics, benchmarks, market data)
- Ensure recency: prefer 2023-2025 sources for trend analysis
- Call multiple tools to cross-validate findings
- Extract specific evidence for Y/X indicators, not general information
</guidelines>

<examples>
Y-axis evidence: "GPT-4 achieved 85% on coding benchmarks vs 65% for GPT-3.5" (capability advance)
X-axis evidence: "Enterprise AI adoption increased 40% in 2024" (market trend)
</examples>

**RESEARCH QUERY:** "{query}"

**AVAILABLE TOOLS:** {tool_names}

**INSTRUCTIONS:**
1. Use multiple tools to gather comprehensive evidence
2. Focus on quantitative metrics and recent trends
3. Cross-reference findings across sources
4. Extract specific evidence for disruption indicators
5. Synthesize findings into coherent analysis

Please conduct thorough research and provide your findings with emphasis on Y/X framework evidence."""

    async def _execute_single_tool(self, tool: Any, tool_name: str, tool_args: Dict, step: int, total: int) -> List[Dict[str, str]]:
        """
        Execute a single tool and process its results.
        
        Args:
            tool: The tool to execute
            tool_name: Name of the tool
            tool_args: Arguments for the tool
            step: Current step number
            total: Total number of tools
            
        Returns:
            List[Dict[str, str]]: Processed results from the tool
        """
        logger.info(f"Executing tool {step}/{total}: {tool_name}")
        
        # Log the tool arguments for transparency
        if tool_args:
            args_str = ", ".join([f"{k}={v}" for k, v in tool_args.items()])
            logger.debug(f"Tool arguments: {args_str}")
        
        try:
            # Execute the tool
            if hasattr(tool, 'ainvoke'):
                result = await tool.ainvoke(tool_args)
            elif hasattr(tool, 'invoke'):
                result = tool.invoke(tool_args)
            else:
                result = await tool(tool_args) if asyncio.iscoroutinefunction(tool) else tool(tool_args)
            
            # Log the actual tool response for debugging
            if result:
                result_preview = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
                logger.debug(f"Tool {tool_name} response preview: {result_preview}")
                
                # Process the result with evidence evaluation
                formatted_results = await self._process_tool_result(tool_name, result)
                logger.info(f"Tool {tool_name} returned {len(formatted_results)} formatted results")
                
                # Log details of each formatted result
                for j, formatted_result in enumerate(formatted_results):
                    title = formatted_result.get("title", "No title")
                    evidence_score = formatted_result.get("evidence_score", "N/A")
                    content_preview = formatted_result.get("body", "")[:200] + "..." if len(formatted_result.get("body", "")) > 200 else formatted_result.get("body", "")
                    logger.debug(f"Result {j+1}: '{title}' (evidence: {evidence_score}) - Content: {content_preview}")
                
                return formatted_results
            else:
                logger.warning(f"Tool {tool_name} returned empty result")
                return []
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return []

    async def _process_tool_result(self, tool_name: str, result: Any) -> List[Dict[str, str]]:
        """
        Process tool result into search result format with Y/X evidence evaluation.
        
        Args:
            tool_name: Name of the tool that produced the result
            result: The tool result
            
        Returns:
            List[Dict[str, str]]: Formatted search results with evidence scores
        """
        search_results = []
        
        try:
            if isinstance(result, list):
                # If the result is already a list, process each item
                for i, item in enumerate(result):
                    processed_item = await self._process_single_item(tool_name, item, i)
                    if processed_item:
                        search_results.append(processed_item)
            elif isinstance(result, dict):
                # If the result is a dictionary, use it as a single search result
                processed_item = await self._process_single_item(tool_name, result, 0)
                if processed_item:
                    search_results.append(processed_item)
            else:
                # For any other type, convert to string and use as a single search result
                processed_item = await self._process_single_item(tool_name, {"content": str(result)}, 0)
                if processed_item:
                    search_results.append(processed_item)
                
        except Exception as e:
            logger.error(f"Error processing tool result from {tool_name}: {e}")
            # Fallback: create a basic result
            search_result = {
                "title": f"Result from {tool_name}",
                "href": f"mcp://{tool_name}",
                "body": clean_text(str(result)),
                "evidence_score": 0
            }
            search_results.append(search_result)
        
        return search_results

    async def _process_single_item(self, tool_name: str, item: Any, index: int) -> Dict[str, str]:
        """
        Process a single item with evidence evaluation.
        
        Args:
            tool_name: Name of the source tool
            item: The item to process
            index: Index of the item
            
        Returns:
            Dict[str, str]: Processed search result with evidence score
        """
        if isinstance(item, dict):
            # Use the item as is if it has required fields
            title = item.get("title", f"Result from {tool_name}")
            href = item.get("href", item.get("url", f"mcp://{tool_name}/{index}"))
            body = clean_text(item.get("body", item.get("content", str(item))))
        else:
            # Create a search result with a generic title
            title = f"Result from {tool_name}"
            href = f"mcp://{tool_name}/{index}"
            body = clean_text(str(item))
        
        # Evaluate evidence quality using Y/X rubric
        evidence_score = await self._evaluate_evidence_quality(title, body)
        
        search_result = {
            "title": title,
            "href": href,
            "body": body,
            "evidence_score": evidence_score
        }
        
        return search_result

    async def _evaluate_evidence_quality(self, title: str, content: str) -> float:
        """
        Evaluate evidence quality using Y/X framework rubric.
        
        Args:
            title: Title of the content
            content: Content to evaluate
            
        Returns:
            float: Evidence quality score (1-10)
        """
        try:
            evaluation_prompt = f"""<role>
You are an expert evaluator for disruption research using Y/X framework analysis.
</role>

<rubric>
Evaluate this content for evidence quality on a 1-10 scale:

**Y-Axis Evidence (40%)**: Does it contain quantitative performance/capability improvements? 
- Metrics, benchmarks, efficiency gains, technical advances
- Score 1-10 based on specificity and credibility

**X-Axis Evidence (40%)**: Does it contain market/adoption trend data?
- Market growth, user adoption, ecosystem changes, business impacts  
- Score 1-10 based on relevance and recency

**Source Quality (20%)**: Is the source credible and recent?
- Authority of source, publication date (2023-2025 preferred), methodology
- Score 1-10 based on trustworthiness

**Overall Score**: Weighted average (Y*0.4 + X*0.4 + Quality*0.2)
</rubric>

**Title:** {title}
**Content:** {content[:1000]}

Respond with only a single number (1-10) representing the overall evidence quality score."""

            # Try to get LLM evaluation
            try:
                from ..utils.llm import create_chat_completion
                
                messages = [{"role": "user", "content": evaluation_prompt}]
                
                response = await create_chat_completion(
                    model=self.cfg.fast_llm_model,  # Use fast model for evaluation
                    messages=messages,
                    temperature=0.1,
                    max_tokens=50,
                    llm_provider=self.cfg.fast_llm_provider,
                    llm_kwargs=self.cfg.llm_kwargs,
                    cost_callback=self.researcher.add_costs if self.researcher and hasattr(self.researcher, 'add_costs') else None,
                )
                
                # Extract numeric score
                score_match = re.search(r'(\d+\.?\d*)', response.strip())
                if score_match:
                    score = float(score_match.group(1))
                    return min(max(score, 1.0), 10.0)  # Clamp to 1-10 range
                
            except ImportError:
                # Fallback to client call
                from .client import call_llm_fallback
                
                response = await call_llm_fallback(
                    prompt=evaluation_prompt,
                    temperature=0.1,
                    max_tokens=50,
                    cost_callback=self.researcher.add_costs if self.researcher and hasattr(self.researcher, 'add_costs') else None
                )
                
                score_match = re.search(r'(\d+\.?\d*)', response.strip())
                if score_match:
                    score = float(score_match.group(1))
                    return min(max(score, 1.0), 10.0)
            
            # Simple heuristic fallback
            return self._heuristic_evidence_score(title, content)
            
        except Exception as e:
            logger.debug(f"Error in evidence evaluation: {e}")
            return self._heuristic_evidence_score(title, content)

    def _heuristic_evidence_score(self, title: str, content: str) -> float:
        """
        Heuristic-based evidence quality scoring as fallback with enhanced regex patterns.
        
        Args:
            title: Content title
            content: Content text
            
        Returns:
            float: Heuristic evidence score (1-10)
        """
        score = 5.0  # Base score
        text = (title + " " + content).lower()
        
        # Y-axis indicators (performance/capability)
        y_indicators = ['benchmark', 'performance', 'efficiency', 'improvement', 'metric', 'score', 'accuracy', 'speed', 'capability']
        y_score = sum(1 for indicator in y_indicators if indicator in text)
        
        # X-axis indicators (market/adoption)
        x_indicators = ['market', 'adoption', 'growth', 'users', 'revenue', 'investment', 'deployment', 'enterprise', 'commercial']
        x_score = sum(1 for indicator in x_indicators if indicator in text)
        
        # Quality indicators
        quality_indicators = ['study', 'research', 'report', 'analysis', '2023', '2024', '2025']
        quality_score = sum(1 for indicator in quality_indicators if indicator in text)
        
        # *** NEW: Enhanced regex pattern matching for quantitative evidence ***
        import re
        
        # Date patterns (years 2020-2030 for recency)
        if re.search(r'\b202[0-9]\b', text):
            score += 1.5  # Recent dates boost credibility
            
        # Percentage patterns (e.g., "40%", "60%", "increase by 25%")
        percentage_matches = len(re.findall(r'\d+%', text))
        score += min(percentage_matches * 0.8, 2.0)  # Cap at +2.0
        
        # Quantitative metrics (numbers with units/context)
        metric_patterns = [
            r'\$\d+[kmb]?',           # Money: $50M, $2B, $100k
            r'\d+[kmb]?\s*users',     # Users: 10M users, 500k users  
            r'\d+x\s*faster',         # Performance: 3x faster, 10x improvement
            r'\d+\.\d+\s*score',      # Scores: 8.5 score, 0.95 accuracy
            r'\d+\s*fold',            # Multipliers: 5-fold increase
            r'\d+\s*basis\s*points'   # Financial: 50 basis points
        ]
        
        for pattern in metric_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += min(matches * 0.5, 1.5)  # Cap each pattern at +1.5
            
        # Authority/Source patterns
        authority_patterns = [
            r'(mckinsey|bain|bcg|mit|stanford|harvard)',  # Consulting/Academic
            r'(gartner|forrester|idc)',                    # Research firms  
            r'(nature|science|ieee)',                      # Academic journals
            r'(sec filing|10-k|earnings)',                 # Official filings
            r'(cited by \d+)',                            # Citation count
        ]
        
        for pattern in authority_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1.0  # Authority sources get significant boost
                
        # Benchmark/Competition patterns
        benchmark_patterns = [
            r'(outperform|beat|exceed|surpass)',          # Performance comparisons
            r'(benchmark|baseline|state-of-the-art)',     # Benchmark references
            r'(vs\.|versus|compared to)',                 # Direct comparisons
            r'(ranking|leaderboard|top \d+)',            # Rankings
        ]
        
        for pattern in benchmark_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.8
                
        # Temporal/Trend patterns  
        trend_patterns = [
            r'(year-over-year|yoy|quarter-over-quarter)', # Growth periods
            r'(projected|forecast|estimate[ds]?)',         # Forward-looking
            r'(trend|trajectory|momentum)',               # Trend language
            r'(since 202[0-9]|by 202[0-9])',             # Time-bound statements
        ]
        
        for pattern in trend_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.6
        
        # Calculate weighted score with enhanced patterns
        final_score = 5.0 + (y_score * 0.8) + (x_score * 0.8) + (quality_score * 0.4)
        
        # Apply regex boost (already added above, so use current score)
        final_score = score + (y_score * 0.8) + (x_score * 0.8) + (quality_score * 0.4) - 5.0  # Adjust for double base
        
        return min(max(final_score, 1.0), 10.0)

    async def _process_llm_analysis(self, query: str, content: str) -> Dict[str, str]:
        """
        Process LLM analysis content with evidence evaluation.
        
        Args:
            query: Original research query
            content: LLM analysis content
            
        Returns:
            Dict[str, str]: Processed LLM analysis result
        """
        cleaned_content = clean_text(content)
        evidence_score = await self._evaluate_evidence_quality(f"LLM Analysis: {query}", cleaned_content)
        
        return {
            "title": f"LLM Analysis: {query}",
            "href": "mcp://llm_analysis",
            "body": cleaned_content,
            "evidence_score": evidence_score
        } 

    async def _map_to_indicators(self, evidence: str, query: str) -> Dict[str, List[Dict]]:
        """
        Map aggregated evidence to exactly 6 Y and 6 X indicators using self-consistency with caching.
        
        Args:
            evidence: Aggregated evidence from all research results
            query: Original research query for context
            
        Returns:
            Dict with 'y' and 'x' arrays of indicator mappings
        """
        try:
            # Define the 12 specific indicators
            y_indicators = [
                "Task Structure", "Risk", "Contextual Knowledge", 
                "Data Availability", "Process Variability", "Human Workflow"
            ]
            x_indicators = [
                "External Observability", "Industry Standardization", "Proprietary Data",
                "Switching Friction", "Regulatory Barriers", "Agent Protocol"
            ]
            
            # Check cache first
            cached_indicators = self._load_cached_indicators(query)
            if cached_indicators:
                logger.info("Using cached indicator mapping")
                return cached_indicators
            
            # Create base prompt for indicator mapping
            base_prompt = self._create_indicator_mapping_prompt(evidence, query, y_indicators, x_indicators)
            
            # Self-consistency: Generate 3 variants and vote on scores
            variant_results = []
            
            for variant_idx in range(3):
                variant_prompt = f"{base_prompt}\n\n### VARIANT {variant_idx + 1} PERSPECTIVE:\nAnalyze from perspective {variant_idx + 1}: Focus on {'technical feasibility' if variant_idx == 0 else 'market dynamics' if variant_idx == 1 else 'implementation challenges'}."
                
                logger.debug(f"Running indicator mapping variant {variant_idx + 1}/3")
                response = await self._call_llm_for_indicators(variant_prompt)
                
                if response:
                    parsed_indicators = self._parse_indicator_response(response, y_indicators, x_indicators)
                    if parsed_indicators:
                        variant_results.append(parsed_indicators)
                        logger.debug(f"Variant {variant_idx + 1} mapped indicators successfully")
            
            if not variant_results:
                logger.warning("No valid indicator mappings from any variant")
                return self._create_fallback_indicators(y_indicators, x_indicators)
            
            # Vote on indicator scores using median
            final_indicators = self._vote_on_indicators(variant_results, y_indicators, x_indicators)
            
            # Validate that we have exactly 6 Y and 6 X indicators
            assert len(final_indicators['y']) == 6, f"Expected 6 Y indicators, got {len(final_indicators['y'])}"
            assert len(final_indicators['x']) == 6, f"Expected 6 X indicators, got {len(final_indicators['x'])}"
            
            # Verify score ranges
            for indicator_type in ['y', 'x']:
                for ind in final_indicators[indicator_type]:
                    assert 1 <= ind['score'] <= 10, f"Score {ind['score']} out of range 1-10"
                    assert ind['name'] in (y_indicators if indicator_type == 'y' else x_indicators)
            
            logger.info("Successfully mapped evidence to 12 Y/X indicators with validation")
            
            # Save successful mapping to cache
            self._save_cached_indicators(query, final_indicators)
            
            return final_indicators
            
        except Exception as e:
            logger.error(f"Error in indicator mapping: {e}")
            fallback_indicators = self._create_fallback_indicators(y_indicators, x_indicators)
            
            # Save fallback to cache too (prevents repeated failures)
            self._save_cached_indicators(query, fallback_indicators)
            
            return fallback_indicators

    def _get_cache_dir(self) -> str:
        """Get cache directory from config or use default."""
        return getattr(self.cfg, 'mcp_cache_dir', 'cache_dir')

    def _get_indicator_cache_key(self, query: str) -> str:
        """Generate cache key for indicator mapping."""
        import hashlib
        return hashlib.md5(query.encode()).hexdigest() + '_indicators'

    def _load_cached_indicators(self, query: str) -> Dict:
        """
        Load cached indicator mapping if available.
        
        Args:
            query: Research query
            
        Returns:
            Dict: Cached indicator mapping or None if not found
        """
        try:
            import os
            cache_dir = self._get_cache_dir()
            
            if not os.path.exists(cache_dir):
                return None
                
            cache_key = self._get_indicator_cache_key(query)
            cache_path = os.path.join(cache_dir, f"{cache_key}.json")
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                # Validate cache structure
                if (isinstance(cached_data, dict) and 
                    'y' in cached_data and 'x' in cached_data and
                    len(cached_data['y']) == 6 and len(cached_data['x']) == 6):
                    logger.info(f"Loaded cached indicator mapping (cache key: {cache_key[:8]}...)")
                    return cached_data
                else:
                    logger.warning(f"Invalid indicator cache structure in {cache_path}, ignoring")
                    
        except Exception as e:
            logger.warning(f"Error loading cached indicators: {e}")
            
        return None

    def _save_cached_indicators(self, query: str, indicators: Dict):
        """
        Save indicator mapping to cache.
        
        Args:
            query: Research query
            indicators: Indicator mapping to cache
        """
        if not indicators:
            return
            
        try:
            import os
            cache_dir = self._get_cache_dir()
            
            # Ensure cache directory exists
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
                
            cache_key = self._get_indicator_cache_key(query)
            cache_path = os.path.join(cache_dir, f"{cache_key}.json")
            
            # Add metadata to cached data
            cached_data = {
                **indicators,
                'metadata': {
                    'query': query,
                    'timestamp': __import__('time').time(),
                    'y_count': len(indicators.get('y', [])),
                    'x_count': len(indicators.get('x', []))
                }
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, indent=2)
                
            logger.debug(f"Cached indicator mapping: {len(indicators.get('y', []))} Y + {len(indicators.get('x', []))} X indicators (cache key: {cache_key[:8]}...)")
            
        except Exception as e:
            logger.warning(f"Error saving indicators to cache: {e}")

    def _create_indicator_mapping_prompt(self, evidence: str, query: str, y_indicators: List[str], x_indicators: List[str]) -> str:
        """Create the prompt for indicator mapping with few-shot examples and CoT."""
        
        evidence_preview = evidence[:2000] + "..." if len(evidence) > 2000 else evidence
        
        return f"""<role>
You are an expert AI disruption analyst specializing in the Y/X framework for automation potential assessment.
</role>

<plan>
Map the provided evidence to EXACTLY 6 Y-indicators (Automation Potential) and 6 X-indicators (Penetration Potential). Use Chain-of-Thought reasoning and score each 1-10 based on evidence strength.
</plan>

<indicators>
Y-Indicators (Automation Potential):
{', '.join(y_indicators)}

X-Indicators (Penetration Potential): 
{', '.join(x_indicators)}
</indicators>

<few_shot_examples>
Example 1: Vertical Industry (Low Automation)
Query: "Procore AI disruption risk"
Evidence: "Procore’s project management involves high variability and regulatory oversight, limiting full automation while strong data moats protect entry... Bain notes verticals like construction maintain human oversight (Bain 2025 report)."

Y-Task Structure: score=4 (medium-low due to on-site variability and exceptions, cite Bain variability indicator 2025)
X-Proprietary Data Depth: score=2 (deep proprietary project data creates strong moats, cite Bain proprietary depth 2024)

Example 2: Horizontal SaaS (High Penetration)
Query: "monday.com AI workflow disruption"
Evidence: "monday.com’s open APIs enable easy integration but require human oversight for ad-hoc tasks... McKinsey highlights 5-10% churn risk in horizontal productivity tools (McKinsey 2024 analysis)."

Y-Human Workflow/UI: score=3 (heavy dependency on UI for collaborative oversight, cite McKinsey human-loops 2024)
X-External Observability: score=8 (high visibility through exposed APIs and standardization, cite Bain observability 2025)

Example 3: Vertical Industry (High Automation)
Query: "Guidewire AI automation potential"
Evidence: "Guidewire’s claims processing is rules-based and automatable, but HIPAA regulations create barriers... Coatue emphasizes ROI in vertical claims automation (Coatue 2024 supercycle report)."

Y-Task Structure: score=7 (medium-high for structured claims workflows, cite Bain structure indicator 2024)
X-Regulatory Barriers: score=2 (high compliance requirements limit easy entry, cite Bain barriers 2025)
</few_shot_examples>

<chain_of_thought>
Step 1: Identify Y evidence (task automation, risk reduction, context needs, data requirements, process variation, human interaction)
Step 2: Identify X evidence (market visibility, standardization, proprietary barriers, switching costs, regulations, protocols)  
Step 3: Score each indicator 1-10 based on evidence strength and disruption potential
Step 4: Extract supporting sources/quotes for rationale
</chain_of_thought>

**QUERY:** "{query}"

**EVIDENCE:**
{evidence_preview}

**INSTRUCTIONS:**
Provide EXACTLY this JSON structure with all 12 indicators:

{{
    "y": [
        {{
            "name": "Task Structure",
            "score": 1-10,
            "rationale": "Evidence-based explanation with specific quotes",
            "sources": ["Source 1", "Source 2"]
        }},
        // ... exactly 6 Y indicators
    ],
    "x": [
        {{
            "name": "External Observability", 
            "score": 1-10,
            "rationale": "Evidence-based explanation with specific quotes",
            "sources": ["Source 1", "Source 2"]
        }},
        // ... exactly 6 X indicators  
    ]
}}

Score 1=No disruption potential, 10=Maximum disruption potential. Use specific evidence quotes in rationales."""

    async def _call_llm_for_indicators(self, prompt: str) -> str:
        """Call LLM for indicator mapping."""
        try:
            # Try to use existing LLM utilities first
            try:
                from ..utils.llm import create_chat_completion
                
                messages = [{"role": "user", "content": prompt}]
                
                result = await create_chat_completion(
                    model=self.cfg.strategic_llm_model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistency
                    max_tokens=2000,
                    llm_provider=self.cfg.strategic_llm_provider,
                    llm_kwargs=self.cfg.llm_kwargs,
                    cost_callback=self.researcher.add_costs if self.researcher and hasattr(self.researcher, 'add_costs') else None,
                )
                return result
                
            except ImportError:
                # Fallback to direct client call
                from .client import call_llm_fallback
                
                result = await call_llm_fallback(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=2000,
                    cost_callback=self.researcher.add_costs if self.researcher and hasattr(self.researcher, 'add_costs') else None
                )
                return result
                
        except Exception as e:
            logger.error(f"Error calling LLM for indicator mapping: {e}")
            return ""

    def _parse_indicator_response(self, response: str, y_indicators: List[str], x_indicators: List[str]) -> Dict:
        """Parse LLM response for indicator mappings."""
        try:
            # Try to parse as JSON
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    return None
            
            # Validate structure
            if not isinstance(result, dict) or 'y' not in result or 'x' not in result:
                return None
                
            # Validate indicator counts and names
            if len(result['y']) != 6 or len(result['x']) != 6:
                logger.warning(f"Invalid indicator count: Y={len(result['y'])}, X={len(result['x'])}")
                return None
            
            # Validate indicator names match expected lists
            y_names = {ind.get('name') for ind in result['y']}
            x_names = {ind.get('name') for ind in result['x']}
            
            if not y_names.issubset(set(y_indicators)) or not x_names.issubset(set(x_indicators)):
                logger.warning("Indicator names don't match expected lists")
                return None
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing indicator response: {e}")
            return None

    def _vote_on_indicators(self, variant_results: List[Dict], y_indicators: List[str], x_indicators: List[str]) -> Dict:
        """Vote on indicator scores using median across variants with fallback rationales and bias caps."""
        
        final_result = {"y": [], "x": []}
        
        # Detect if this is a vertical industry query for bias application
        industry_type = self._detect_industry_type()
        
        # Process Y indicators
        for y_name in y_indicators:
            scores = []
            rationales = []
            sources = []
            
            for variant in variant_results:
                for y_ind in variant.get('y', []):
                    if y_ind.get('name') == y_name:
                        scores.append(y_ind.get('score', 5))
                        rationales.append(y_ind.get('rationale', ''))
                        sources.extend(y_ind.get('sources', []))
            
            if scores:
                median_score = statistics.median(scores)
                unique_sources = list(set(sources))
                rationale = rationales[0] if rationales else 'No rationale'
                
                # Apply bias caps and fallback rationales
                median_score, rationale = self._apply_score_adjustments(
                    median_score, unique_sources, rationale, industry_type
                )
                
                final_result['y'].append({
                    "name": y_name,
                    "score": int(median_score),
                    "rationale": f"Consensus from {len(scores)} variants: {rationale}",
                    "sources": unique_sources
                })
        
        # Process X indicators  
        for x_name in x_indicators:
            scores = []
            rationales = []
            sources = []
            
            for variant in variant_results:
                for x_ind in variant.get('x', []):
                    if x_ind.get('name') == x_name:
                        scores.append(x_ind.get('score', 5))
                        rationales.append(x_ind.get('rationale', ''))
                        sources.extend(x_ind.get('sources', []))
            
            if scores:
                median_score = statistics.median(scores)
                unique_sources = list(set(sources))
                rationale = rationales[0] if rationales else 'No rationale'
                
                # Apply bias caps and fallback rationales
                median_score, rationale = self._apply_score_adjustments(
                    median_score, unique_sources, rationale, industry_type
                )
                
                final_result['x'].append({
                    "name": x_name,
                    "score": int(median_score),
                    "rationale": f"Consensus from {len(scores)} variants: {rationale}",
                    "sources": unique_sources
                })
        
        return final_result

    def _detect_industry_type(self) -> str:
        """
        Detect if the query relates to a vertical industry for bias application.
        
        Returns:
            str: 'vertical' for industry-specific tools, 'horizontal' for general tools
        """
        # Get the original query from researcher context if available
        query = ""
        if hasattr(self, 'researcher') and self.researcher and hasattr(self.researcher, 'query'):
            query = self.researcher.query.lower()
        
        # Vertical industry indicators
        vertical_patterns = [
            'construction', 'procore', 'healthcare', 'finance', 'banking',
            'manufacturing', 'retail', 'real estate', 'agriculture', 'mining',
            'oil', 'gas', 'energy', 'transportation', 'logistics', 'legal'
        ]
        
        for pattern in vertical_patterns:
            if pattern in query:
                logger.debug(f"Detected vertical industry pattern: {pattern}")
                return 'vertical'
        
        # Default to horizontal for general/tech tools
        return 'horizontal'

    def _apply_score_adjustments(self, median_score: float, sources: List[str], rationale: str, industry_type: str) -> tuple:
        """
        Apply score adjustments for recency, evidence quality, and vertical bias.
        
        Args:
            median_score: Original median score
            sources: List of sources
            rationale: Original rationale
            industry_type: 'vertical' or 'horizontal'
            
        Returns:
            tuple: (adjusted_score, updated_rationale)
        """
        adjusted_score = median_score
        adjustments = []
        
        # Check for recent sources (2023-2025)
        has_recent_sources = any(re.search(r'202[3-9]', str(source)) for source in sources)
        
        # Downgrade if no recent sources found
        if not has_recent_sources:
            adjusted_score -= 2
            adjustments.append("downgraded -2 (no recent 2023+ sources)")
            logger.debug("Applied -2 penalty for lack of recent sources")
        
        # Apply vertical industry bias cap
        if industry_type == 'vertical' and adjusted_score > 5 and len(sources) < 2:
            adjusted_score = 5
            adjustments.append("capped at 5 (vertical industry bias)")
            logger.debug("Applied vertical industry bias cap")
        
        # Add fallback rationale for insufficient evidence
        if len(sources) < 2:
            adjustments.append("Insufficient evidence: Score neutral 5, re-query needed")
            if adjusted_score > 5:
                adjusted_score = 5
        
        # Ensure score stays in valid range
        adjusted_score = max(1, min(10, adjusted_score))
        
        # Update rationale with adjustments
        updated_rationale = rationale
        if adjustments:
            adjustment_text = "; ".join(adjustments)
            updated_rationale += f" [{adjustment_text}]"
        
        return adjusted_score, updated_rationale

    def _create_fallback_indicators(self, y_indicators: List[str], x_indicators: List[str]) -> Dict:
        """Create fallback indicator mapping when LLM fails."""
        fallback = {"y": [], "x": []}
        
        for y_name in y_indicators:
            fallback['y'].append({
                "name": y_name,
                "score": 5,  # Neutral score
                "rationale": "Fallback: Insufficient evidence for reliable scoring",
                "sources": ["Fallback"]
            })
            
        for x_name in x_indicators:
            fallback['x'].append({
                "name": x_name, 
                "score": 5,  # Neutral score
                "rationale": "Fallback: Insufficient evidence for reliable scoring",
                "sources": ["Fallback"]
            })
            
        return fallback 