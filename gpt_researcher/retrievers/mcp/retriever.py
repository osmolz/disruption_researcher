"""
MCP-Based Research Retriever - Optimized for AI Disruption Analysis

A retriever that uses Model Context Protocol (MCP) tools for intelligent research with 
specialized configurations for trusted sources and disruption analysis evaluation.
This retriever implements a three-stage approach:
1. Tool Selection: LLM selects 2-3 most relevant tools with disruption focus
2. Research Execution: LLM uses selected tools with disruption-optimized queries
3. Post-Evaluation: LLM evaluates results against disruption analysis rubric
"""
import asyncio
import logging
import os
import statistics
from typing import List, Dict, Any, Optional
import json

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP_ADAPTERS = True
except ImportError:
    HAS_MCP_ADAPTERS = False

from ...mcp.client import MCPClientManager
from ...mcp.tool_selector import MCPToolSelector
from ...mcp.research import MCPResearchSkill
from ...mcp.streaming import MCPStreamer

logger = logging.getLogger(__name__)


class MCPRetriever:
    """
    Model Context Protocol (MCP) Retriever for GPT Researcher - Optimized for AI Disruption Analysis.
    
    This retriever implements a three-stage approach:
    1. Tool Selection: LLM selects 2-3 most relevant tools with disruption focus
    2. Research Execution: LLM with bound tools conducts intelligent research using disruption-optimized queries
    3. Post-Evaluation: LLM evaluates results against disruption analysis rubric (min 4 sources, recent 2023-2025, relevant)
    
    The retriever requires a researcher instance to access:
    - mcp_configs: List of MCP server configurations
    - cfg: Configuration object with LLM settings and parameters
    - add_costs: Method for tracking research costs
    """

    def __init__(
        self, 
        query: str, 
        headers: Optional[Dict[str, str]] = None,
        query_domains: Optional[List[str]] = None,
        websocket=None,
        researcher=None,
        **kwargs
    ):
        print(f"[MCPRetriever] __init__ called for query: {query}")
        """
        Initialize the MCP Retriever with disruption analysis configurations.
        
        Args:
            query (str): The search query string.
            headers (dict, optional): Headers containing MCP configuration.
            query_domains (list, optional): List of domains to search (not used in MCP).
            websocket: WebSocket for stream logging.
            researcher: Researcher instance containing mcp_configs and cfg.
            **kwargs: Additional arguments (for compatibility).
        """
        self.query = query
        self.headers = headers or {}
        self.query_domains = query_domains or []
        self.websocket = websocket
        self.researcher = researcher
        
        # Initialize disruption analysis configurations
        self.disruption_configs = self._init_disruption_configs()
        
        # Extract mcp_configs and config from the researcher instance
        self.mcp_configs = self._get_mcp_configs()
        self.cfg = self._get_config()
        
        # Initialize modular components
        self.client_manager = MCPClientManager(self.mcp_configs)
        self.tool_selector = MCPToolSelector(self.cfg, self.researcher)
        self.mcp_researcher = MCPResearchSkill(self.cfg, self.researcher)
        self.streamer = MCPStreamer(self.websocket)
        
        # Initialize caching
        self._all_tools_cache = None
        
        # Log initialization
        if self.mcp_configs:
            self.streamer.stream_log_sync(f"🔧 Initializing MCP retriever for disruption analysis: {self.query}")
            self.streamer.stream_log_sync(f"🔧 Found {len(self.mcp_configs)} MCP server configurations")
            self.streamer.stream_log_sync(f"🎯 Loaded {len(self.disruption_configs['trusted_domains'])} trusted domains for analysis")
        else:
            logger.error("No MCP server configurations found. The retriever will fail during search.")
            self.streamer.stream_log_sync("❌ CRITICAL: No MCP server configurations found. Please check documentation.")
            self.streamer.stream_log_sync("🚨 Underspecification: Add MCP server configs for SEC, academic sources, and financial data")
            
        # Edge case validations
        if not self.query or len(self.query.strip()) < 3:
            logger.warning(f"Query too short or empty: '{self.query}' - may produce poor results")
            self.streamer.stream_log_sync("⚠️ Query underspecified - consider adding more context for better results")
            
        if not self.disruption_configs["trusted_domains"]:
            logger.error("No trusted domains configured - analysis quality will be severely impacted")
            self.streamer.stream_log_sync("❌ No trusted domains configured - falling back to default web search")
            
        if not self.cfg:
            logger.error("No configuration provided - cannot proceed with LLM operations")
            raise ValueError("MCPRetriever requires a researcher instance with cfg attribute")
            
        # Validate environment setup
        if not os.getenv('OPENAI_API_KEY') and not HAS_OPENAI:
            logger.warning("No OpenAI API key found and OpenAI package not available - falling back to configured LLM")
            self.streamer.stream_log_sync("⚠️ No OpenAI setup detected - using fallback LLM provider")

    async def call_llm(self, prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        """
        Direct LLM wrapper for GPT-4o with cost tracking and error handling.
        
        Args:
            prompt (str): The prompt to send to the LLM
            temperature (float): Temperature setting (default 0.3 for consistency)
            max_tokens (int): Maximum tokens in response (default 500)
            
        Returns:
            str: LLM response content
        """
        if not HAS_OPENAI:
            logger.error("OpenAI package not available, falling back to existing LLM provider")
            # Fallback to existing provider
            from ...llm_provider.generic.base import GenericLLMProvider
            provider_kwargs = {
                'model': self.cfg.strategic_llm_model,
                **self.cfg.llm_kwargs
            }
            llm_provider = GenericLLMProvider.from_provider(
                self.cfg.strategic_llm_provider, 
                **provider_kwargs
            )
            messages = [{"role": "user", "content": prompt}]
            response = await llm_provider.llm.ainvoke(messages)
            return response.content
        
        try:
            # Estimate tokens for cost tracking
            prompt_tokens = self.estimate_tokens(prompt)
            
            # Edge case: Check for token limits
            if prompt_tokens > 4000:  # Conservative limit for GPT-4o context
                logger.warning(f"Prompt tokens ({prompt_tokens}) exceed safe limit, truncating...")
                # Truncate prompt to fit within limits
                truncated_prompt = prompt[:int(len(prompt) * 0.7)]  # Keep 70% of prompt
                prompt = truncated_prompt + "\n\nIMPORTANT: Return valid JSON only."
                prompt_tokens = self.estimate_tokens(prompt)
                await self.streamer.stream_log(f"⚠️ Prompt truncated to {prompt_tokens} tokens")
            
            # Edge case: Empty or very short prompts
            if len(prompt.strip()) < 50:
                logger.warning("Prompt very short, may produce poor results")
                await self.streamer.stream_log("⚠️ Short prompt detected - results may be limited")
            
            # Create OpenAI client
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Make API call with timeout and retries
            try:
                response = client.chat.completions.create(
                    model='gpt-4o',
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=min(max_tokens, 500),  # Ensure reasonable limit
                    timeout=30  # 30 second timeout
                )
            except Exception as api_error:
                logger.error(f"OpenAI API call failed: {api_error}")
                await self.streamer.stream_error(f"API call failed: {str(api_error)}")
                return "{}"  # Return empty JSON for parsing safety
            
            response_content = response.choices[0].message.content
            
            # Edge case: Empty or malformed response
            if not response_content or len(response_content.strip()) < 10:
                logger.warning("Received empty or very short LLM response")
                await self.streamer.stream_log("⚠️ LLM returned minimal response")
                return "{}"  # Return empty JSON for parsing safety
            
            response_tokens = self.estimate_tokens(response_content)
            
            # Track costs with researcher if available
            total_cost = (prompt_tokens * 0.00001 + response_tokens * 0.00003)  # GPT-4o pricing
            if self.researcher and hasattr(self.researcher, 'add_costs'):
                self.researcher.add_costs({'llm_calls': total_cost})
            
            logger.info(f"🤖 GPT-4o call: {prompt_tokens}+{response_tokens} tokens, ${total_cost:.4f}")
            
            return response_content
            
        except Exception as e:
            logger.error(f"Error in GPT-4o call: {e}")
            await self.streamer.stream_error(f"LLM call failed: {str(e)}")
            return "{}"  # Return empty JSON for parsing safety

    def estimate_tokens(self, text: str) -> int:
        """
        Conservative token estimation for cost tracking.
        
        Args:
            text (str): Text to estimate tokens for
            
        Returns:
            int: Estimated token count
        """
        return int(len(text.split()) * 1.3)  # Conservative estimate

    def _init_disruption_configs(self) -> Dict[str, Any]:
        """
        Initialize disruption analysis configurations with environment-loadable trusted domains and MCP tools.
        
        Returns:
            Dict[str, Any]: Disruption analysis configuration
        """
        # Environment-loadable configurations with sensible defaults
        default_trusted_domains = (
            "mckinsey.com,bain.com,sec.gov,bcg.com,deloitte.com,pwc.com,kpmg.com,"
            "mit.edu,stanford.edu,harvard.edu,brookings.edu,cfr.org,weforum.org,"
            "oecd.org,arxiv.org,nature.com,science.org,ieee.org,acm.org,"
            "forbes.com,reuters.com,ft.com,wsj.com,bloomberg.com"
        )
        
        default_disruption_keywords = (
            "AI disruption,artificial intelligence transformation,digital transformation,"
            "automation impact,machine learning adoption,industry transformation,"
            "business model disruption,technological displacement,AI implementation,"
            "competitive advantage AI"
        )
        
        default_temporal_filters = "since:2023,2023-2025,recent,latest,current trends"
        
        disruption_configs = {
            "trusted_domains": os.getenv('TRUSTED_DOMAINS', default_trusted_domains).split(','),
            "disruption_keywords": os.getenv('DISRUPTION_KEYWORDS', default_disruption_keywords).split(','),
            "temporal_filters": os.getenv('TEMPORAL_FILTERS', default_temporal_filters).split(','),
            "mcp_tools": [
                {
                    "name": "web_search",
                    "command": "python web_search.py",
                    "description": "Web search tool for finding recent disruption analysis content"
                },
                {
                    "name": "browse_page", 
                    "command": "python browse_page.py",
                    "description": "Page browsing tool for detailed content extraction from trusted sources"
                }
            ]
        }
        
        # Clean up any empty strings from environment splits
        disruption_configs["trusted_domains"] = [domain.strip() for domain in disruption_configs["trusted_domains"] if domain.strip()]
        disruption_configs["disruption_keywords"] = [keyword.strip() for keyword in disruption_configs["disruption_keywords"] if keyword.strip()]
        disruption_configs["temporal_filters"] = [filter_term.strip() for filter_term in disruption_configs["temporal_filters"] if filter_term.strip()]
        
        logger.info(f"🌍 Loaded {len(disruption_configs['trusted_domains'])} trusted domains from environment")
        logger.info(f"🔍 Loaded {len(disruption_configs['disruption_keywords'])} disruption keywords from environment")
        
        # Add web_search and browse_page tools to mcp_configs if not already present
        if self.researcher and hasattr(self.researcher, 'mcp_configs'):
            existing_names = {config.get('name', '') for config in self.researcher.mcp_configs}
            for tool_config in disruption_configs["mcp_tools"]:
                if tool_config["name"] not in existing_names:
                    self.researcher.mcp_configs.append(tool_config)
        
        return disruption_configs

    def _get_mcp_configs(self) -> List[Dict[str, Any]]:
        """
        Get MCP configurations from the researcher instance.
        
        Returns:
            List[Dict[str, Any]]: List of MCP server configurations.
        """
        if self.researcher and hasattr(self.researcher, 'mcp_configs'):
            return self.researcher.mcp_configs or []
        return []

    def _get_config(self):
        """
        Get configuration from the researcher instance.
        
        Returns:
            Config: Configuration object with LLM settings.
        """
        if self.researcher and hasattr(self.researcher, 'cfg'):
            return self.researcher.cfg
        
        # If no config available, this is a critical error
        logger.error("No config found in researcher instance. MCPRetriever requires a researcher instance with cfg attribute.")
        raise ValueError("MCPRetriever requires a researcher instance with cfg attribute containing LLM configuration")

    def _optimize_query_for_disruption(self, original_query: str) -> str:
        """
        Optimize query for disruption analysis with comprehensive OR patterns for trusted sources.
        
        Args:
            original_query (str): Original research query
            
        Returns:
            str: Optimized query with disruption analysis focus and trusted source constraints
        """
        # Build site constraints with OR pattern for trusted sources
        site_constraints = " OR ".join([f"site:{domain}" for domain in self.disruption_configs["trusted_domains"]])
        
        # Build disruption keyword alternatives with OR pattern
        disruption_terms = " OR ".join([f'"{keyword}"' for keyword in self.disruption_configs["disruption_keywords"]])
        
        # Build temporal filters with OR pattern
        temporal_terms = " OR ".join(self.disruption_configs["temporal_filters"])
        
        # Construct comprehensive optimized query
        # Pattern: (site:domain1 OR site:domain2 OR ...) original_query ("AI disruption" OR "digital transformation" OR ...) (since:2023 OR recent OR ...)
        optimized_query = f"({site_constraints}) {original_query} ({disruption_terms}) ({temporal_terms})"
        
        # Additional enhancement for Y/X framework analysis
        if "framework" in original_query.lower() or "analysis" in original_query.lower():
            optimized_query += " (framework OR methodology OR strategic analysis)"
        
        # Cap query length to avoid search engine limits
        if len(optimized_query) > 512:
            # Fallback to essential components if too long
            essential_sites = " OR ".join([f"site:{domain}" for domain in self.disruption_configs["trusted_domains"][:10]])
            essential_keywords = " OR ".join([f'"{keyword}"' for keyword in self.disruption_configs["disruption_keywords"][:5]])
            optimized_query = f"({essential_sites}) {original_query} ({essential_keywords}) since:2023"
            logger.warning(f"Query truncated due to length: {len(optimized_query)} chars")
        
        logger.info(f"🎯 Query optimized for disruption analysis:")
        logger.info(f"   Original: '{original_query}'")
        logger.info(f"   Optimized: '{optimized_query[:200]}{'...' if len(optimized_query) > 200 else ''}'")
        
        return optimized_query

    async def _post_eval_results(self, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Post-evaluate research results using self-consistent LLM rubric (3 evaluations + median vote) for disruption analysis quality.
        
        Args:
            results: List of research results to evaluate
            
        Returns:
            List[Dict[str, str]]: Filtered results that meet quality criteria (median avg score ≥7/10)
        """
        if not results:
            return []
            
        if len(results) < 4:
            logger.warning(f"Only {len(results)} results found, minimum 4 required for quality analysis")
            await self.streamer.stream_log("⚠️ Insufficient results for quality analysis (minimum 4 required)")
            return []
        
        try:
            await self.streamer.stream_log("🔍 Running self-consistent evaluation with 3 LLM judges...")
            
            # Run 3 independent evaluations for self-consistency
            evaluations = []
            for variant in range(3):
                evaluation_prompt = self._generate_evaluation_prompt(results, variant_seed=variant)
                response = await self.call_llm(evaluation_prompt, temperature=0.3, max_tokens=500)
                evaluation_result = self._parse_evaluation_response(response)
                evaluations.append(evaluation_result)
                
                await self.streamer.stream_log(f"✅ Evaluation {variant + 1}/3 completed")
            
            # Vote on median scores for each result
            filtered_results = []
            total_score = 0
            valid_scores = 0
            
            for i, result in enumerate(results):
                # Collect scores from all 3 evaluations for this result
                scores_for_result = []
                eval_details = []
                
                for eval_result in evaluations:
                    if i < len(eval_result.get("evaluations", [])):
                        eval_data = eval_result["evaluations"][i]
                        avg_score = eval_data.get("average_score", 0)
                        scores_for_result.append(avg_score)
                        eval_details.append(eval_data)
                
                if len(scores_for_result) >= 2:  # Need at least 2 valid scores
                    median_score = statistics.median(scores_for_result)
                    
                    if median_score >= 7.0:
                        # Add self-consistency evaluation metadata to result
                        result["disruption_eval"] = {
                            "median_score": median_score,
                            "all_scores": scores_for_result,
                            "score_variance": statistics.variance(scores_for_result) if len(scores_for_result) > 1 else 0,
                            "consensus_strength": len([s for s in scores_for_result if s >= 7.0]) / len(scores_for_result),
                            "eval_details": eval_details
                        }
                        
                        filtered_results.append(result)
                        total_score += median_score
                        valid_scores += 1
                        
                        logger.info(f"✅ Result {i+1} passed self-consistent evaluation (median: {median_score:.1f}/10, scores: {scores_for_result})")
                    else:
                        logger.info(f"❌ Result {i+1} failed self-consistent evaluation (median: {median_score:.1f}/10, scores: {scores_for_result})")
                else:
                    logger.warning(f"⚠️ Result {i+1} had insufficient evaluation scores: {len(scores_for_result)}")
            
            # Calculate overall quality metrics
            overall_score = total_score / valid_scores if valid_scores > 0 else 0
            
            # Calculate consensus metrics
            all_individual_scores = [score for result in filtered_results for score in result["disruption_eval"]["all_scores"]]
            consensus_rate = len([s for s in all_individual_scores if s >= 7.0]) / len(all_individual_scores) if all_individual_scores else 0
            
            await self.streamer.stream_log(f"📊 Self-consistent evaluation complete:")
            await self.streamer.stream_log(f"   • {len(filtered_results)}/{len(results)} results passed")
            await self.streamer.stream_log(f"   • Average median score: {overall_score:.1f}/10")
            await self.streamer.stream_log(f"   • Consensus rate: {consensus_rate:.1%}")
            
            if len(filtered_results) < 4:
                logger.warning(f"Only {len(filtered_results)} high-quality results found, returning empty set")
                await self.streamer.stream_log("⚠️ Insufficient high-quality results for disruption analysis")
                return []
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in self-consistent post-evaluation: {e}")
            await self.streamer.stream_log(f"❌ Error in self-consistent evaluation, returning original results: {str(e)}")
            return results

    def _generate_evaluation_prompt(self, results: List[Dict[str, str]], variant_seed: int = 0) -> str:
        """
        Generate LLM prompt for evaluating research results against disruption analysis rubric with variant seeding for self-consistency.
        
        Args:
            results: List of research results to evaluate
            variant_seed: Seed for generating prompt variants (0, 1, 2 for different perspectives)
            
        Returns:
            str: Evaluation prompt
        """
        trusted_domains = ", ".join(self.disruption_configs["trusted_domains"])
        
        results_text = ""
        for i, result in enumerate(results):
            title = result.get("title", "No title")
            url = result.get("href", "No URL")
            content = result.get("body", "")[:500] + "..." if len(result.get("body", "")) > 500 else result.get("body", "")
            
            results_text += f"""
Result {i+1}:
Title: {title}
URL: {url}
Content: {content}

"""
        
        # Generate variant prompts for self-consistency
        if variant_seed == 0:
            perspective = "You are a senior management consultant specializing in AI disruption analysis."
            emphasis = "Focus particularly on business transformation indicators and strategic implications."
        elif variant_seed == 1:
            perspective = "You are an academic researcher studying technology adoption and market disruption."
            emphasis = "Emphasize research methodology, data recency, and analytical rigor."
        else:  # variant_seed == 2
            perspective = "You are a financial analyst evaluating AI investment and disruption trends."
            emphasis = "Prioritize quantitative evidence, market data, and competitive analysis."
        
        return f"""{perspective} Evaluate each research result against the following rubric for AI disruption analysis:

{emphasis}

EVALUATION RUBRIC (Score 1-10 for each criterion):

1. RECENT (2023-2025): Does the content discuss recent developments, trends, or data from 2023-2025?
   - 1-3: Outdated content (pre-2023)
   - 4-6: Some recent references but mostly older content
   - 7-8: Mostly recent content with good 2023-2025 focus
   - 9-10: Exclusively recent, cutting-edge content from 2023-2025

2. TRUSTED SOURCE: Is the source from a trusted domain or authoritative organization?
   Trusted domains: {trusted_domains}
   - 1-3: Unreliable or unknown source
   - 4-6: Somewhat credible source
   - 7-8: Well-known, credible source
   - 9-10: Top-tier trusted source (McKinsey, Bain, SEC, academic institutions, etc.)

3. DISRUPTION RELEVANCE: Does the content provide evidence for AI/technology disruption indicators?
   - 1-3: No clear disruption evidence or indicators
   - 4-6: Limited disruption insights
   - 7-8: Clear disruption patterns and evidence
   - 9-10: Comprehensive disruption analysis with specific indicators and impacts

RESEARCH RESULTS TO EVALUATE:
{results_text}

For each result, provide a JSON evaluation with this exact format:
{{
  "evaluations": [
    {{
      "result_index": 0,
      "recent_score": 8,
      "trusted_score": 9,
      "relevance_score": 7,
      "average_score": 8.0,
      "reasoning": "Detailed explanation of scores and overall assessment"
    }}
  ],
  "overall_assessment": "Summary of evaluation findings and quality recommendations"
}}

IMPORTANT: Average score must be ≥7.0 for a result to qualify for disruption analysis. Return valid JSON only."""

    def _parse_evaluation_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse LLM evaluation response into structured data.
        
        Args:
            response_content: Raw LLM response content
            
        Returns:
            Dict[str, Any]: Parsed evaluation data
        """
        try:
            # Try direct JSON parsing
            return json.loads(response_content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r"\{.*\}", response_content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            logger.warning("Could not parse evaluation response, using fallback")
            # Return fallback structure
            return {
                "evaluations": [],
                "overall_assessment": "Evaluation parsing failed"
            }

    async def search_async(self, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Perform an async search using MCP tools with intelligent three-stage approach optimized for disruption analysis.
        
        Args:
            max_results: Maximum number of results to return.
            
        Returns:
            List[Dict[str, str]]: The search results filtered for disruption analysis quality.
        """
        # Initialize cost tracking
        start_time = asyncio.get_event_loop().time()
        initial_costs = {'llm_calls': 0, 'mcp_operations': 0, 'total_tokens': 0}
        
        # Check if we have any server configurations
        if not self.mcp_configs:
            error_msg = "No MCP server configurations available. Please provide mcp_configs parameter to GPTResearcher."
            logger.error(error_msg)
            await self.streamer.stream_error("MCP retriever cannot proceed without server configurations.")
            return []  # Return empty instead of raising to allow research to continue
            
        # Optimize query for disruption analysis
        optimized_query = self._optimize_query_for_disruption(self.query)
            
        # Log to help debug the integration flow
        logger.info(f"MCPRetriever.search_async called for disruption query: {optimized_query}")
            
        try:
            # Stage 1: Get all available tools
            await self.streamer.stream_stage_start("Stage 1", "Getting available MCP tools for disruption analysis")
            all_tools = await self._get_all_tools()
            
            if not all_tools:
                await self.streamer.stream_warning("No MCP tools available, skipping MCP research")
                return []
            
            # Stage 2: Select most relevant tools with disruption focus
            await self.streamer.stream_stage_start("Stage 2", "Selecting tools optimized for disruption research")
            selected_tools = await self._select_disruption_tools(optimized_query, all_tools, max_tools=3)
            
            if not selected_tools:
                await self.streamer.stream_warning("No relevant tools selected for disruption analysis")
                return []
            
            # Stage 3: Conduct research with selected tools
            await self.streamer.stream_stage_start("Stage 3", "Conducting disruption research with selected tools")
            results = await self.mcp_researcher.conduct_research_with_tools(optimized_query, selected_tools)
            
            # Track MCP operation costs
            mcp_operation_cost = len(selected_tools) * 0.01  # Estimate cost per tool operation
            initial_costs['mcp_operations'] += mcp_operation_cost
            
            # Stage 4: Post-evaluate results with disruption analysis rubric
            await self.streamer.stream_stage_start("Stage 4", "Evaluating results against disruption analysis rubric")
            filtered_results = await self._post_eval_results(results)
            
            # Edge case: Insufficient quality results after evaluation
            if len(filtered_results) < 4:
                logger.warning(f"Quality threshold not met: {len(filtered_results)}/4 minimum high-quality results")
                await self.streamer.stream_log(f"⚠️ Quality threshold not met: {len(filtered_results)}/4 minimum results for reliable analysis")
                await self.streamer.stream_log("💡 Consider: Expanding search terms, relaxing domain constraints, or using additional MCP tools")
                return []  # Return empty for insufficient quality
            
            # Edge case: No results after filtering
            if not filtered_results:
                logger.warning("No results passed quality evaluation")
                await self.streamer.stream_log("❌ No results met disruption analysis quality standards")
                return []
            
            # Slice results to max_results AFTER evaluation (not before)
            original_count = len(filtered_results)
            if len(filtered_results) > max_results:
                logger.info(f"Slicing {len(filtered_results)} quality results to {max_results} max_results")
                filtered_results = filtered_results[:max_results]
                await self.streamer.stream_log(f"✂️ Results sliced from {original_count} to {max_results} for efficiency")
            
            # Edge case: Validate final result structure
            validated_results = []
            for i, result in enumerate(filtered_results):
                if not isinstance(result, dict):
                    logger.warning(f"Result {i} is not a dictionary, skipping")
                    continue
                    
                # Ensure required fields exist
                if not result.get("title") and not result.get("body"):
                    logger.warning(f"Result {i} missing both title and body, skipping")
                    continue
                    
                # Ensure URL exists for citation
                if not result.get("href"):
                    result["href"] = "No URL available"
                    logger.warning(f"Result {i} missing URL, added placeholder")
                
                validated_results.append(result)
            
            if len(validated_results) != len(filtered_results):
                logger.warning(f"Validation removed {len(filtered_results) - len(validated_results)} invalid results")
                await self.streamer.stream_log(f"⚠️ {len(filtered_results) - len(validated_results)} results failed validation")
            
            filtered_results = validated_results
            
            # Final cost tracking and reporting
            end_time = asyncio.get_event_loop().time()
            execution_time = end_time - start_time
            
            # Calculate total tokens processed
            total_content_length = sum(len(result.get("body", "")) for result in filtered_results)
            estimated_total_tokens = self.estimate_tokens(optimized_query) + self.estimate_tokens(str(filtered_results))
            initial_costs['total_tokens'] = estimated_total_tokens
            
            # Report comprehensive costs
            if self.researcher and hasattr(self.researcher, 'add_costs'):
                total_research_cost = initial_costs['llm_calls'] + initial_costs['mcp_operations']
                self.researcher.add_costs({
                    'mcp_retriever_total': total_research_cost,
                    'mcp_execution_time': execution_time,
                    'mcp_results_processed': len(results),
                    'mcp_quality_results': len(filtered_results)
                })
            
            # Log comprehensive cost summary
            logger.info(f"💰 MCP Retriever Cost Summary:")
            logger.info(f"   • LLM calls: ${initial_costs['llm_calls']:.4f}")
            logger.info(f"   • MCP operations: ${initial_costs['mcp_operations']:.4f}")
            logger.info(f"   • Total tokens: {initial_costs['total_tokens']:,}")
            logger.info(f"   • Execution time: {execution_time:.2f}s")
            logger.info(f"   • Quality results: {len(filtered_results)}/{len(results)}")
            
            # Log result summary with disruption analysis focus
            logger.info(f"MCPRetriever returning {len(filtered_results)} high-quality disruption analysis results")
            
            # Stream final cost and result summary
            await self.streamer.stream_research_results(len(filtered_results), total_content_length)
            await self.streamer.stream_log(f"💰 Total research cost: ${initial_costs['llm_calls'] + initial_costs['mcp_operations']:.4f}")
            
            return filtered_results
            
        except Exception as e:
            # Track error costs
            error_time = asyncio.get_event_loop().time() - start_time
            if self.researcher and hasattr(self.researcher, 'add_costs'):
                self.researcher.add_costs({
                    'mcp_retriever_error': 0.001,  # Small error handling cost
                    'mcp_error_time': error_time
                })
            
            logger.error(f"Error in MCP disruption analysis search: {e}")
            await self.streamer.stream_error(f"Error in MCP disruption analysis search: {str(e)}")
            return []
        finally:
            # Ensure client cleanup after search completes
            try:
                await self.client_manager.close_client()
            except Exception as e:
                logger.error(f"Error during client cleanup: {e}")

    async def _select_disruption_tools(self, query: str, all_tools: List, max_tools: int = 3) -> List:
        """
        Select tools with disruption analysis focus, prioritizing web search and browse capabilities.
        
        Args:
            query: Optimized research query 
            all_tools: List of all available tools
            max_tools: Maximum number of tools to select
            
        Returns:
            List: Selected tools optimized for disruption analysis
        """
        # Prioritize disruption-relevant tools
        disruption_tool_patterns = [
            'web_search', 'browse_page', 'search', 'browse', 'web', 'google', 'tavily',
            'get', 'read', 'fetch', 'find', 'query', 'retrieve', 'academic', 'arxiv'
        ]
        
        # First pass: filter tools that match disruption patterns
        relevant_tools = []
        for tool in all_tools:
            tool_name = tool.name.lower()
            tool_description = (tool.description or "").lower()
            
            # Check if tool matches disruption research patterns
            is_relevant = any(pattern in tool_name or pattern in tool_description 
                            for pattern in disruption_tool_patterns)
            
            if is_relevant:
                relevant_tools.append(tool)
        
        if not relevant_tools:
            logger.warning("No disruption-relevant tools found, falling back to standard selection")
            return await self.tool_selector.select_relevant_tools(query, all_tools, max_tools)
        
        # Use standard LLM selection on filtered relevant tools
        selected_tools = await self.tool_selector.select_relevant_tools(query, relevant_tools, max_tools)
        
        # Log disruption-focused selection
        for tool in selected_tools:
            logger.info(f"🎯 Selected disruption analysis tool: {tool.name}")
        
        return selected_tools

    def search(self, max_results: int = 10) -> List[Dict[str, str]]:
        """
        Main search method that orchestrates the MCP research process.
        """
        return asyncio.run(self.search_async(max_results=max_results))

    async def _get_all_tools(self) -> List:
        """
        Get all available tools from MCP servers.
        
        Returns:
            List: All available MCP tools
        """
        if self._all_tools_cache is not None:
            return self._all_tools_cache
            
        try:
            all_tools = await self.client_manager.get_all_tools()
            
            if all_tools:
                await self.streamer.stream_log(f"📋 Loaded {len(all_tools)} total tools from MCP servers")
                self._all_tools_cache = all_tools
                return all_tools
            else:
                await self.streamer.stream_warning("No tools available from MCP servers")
                return []
                
        except Exception as e:
            logger.error(f"Error getting MCP tools: {e}")
            await self.streamer.stream_error(f"Error getting MCP tools: {str(e)}")
            return [] 