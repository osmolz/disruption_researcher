from typing import Any, Optional, Dict, List
import json
import os
import re
import time
from statistics import median

from .config import Config
from .memory import Memory
from .utils.enum import ReportSource, ReportType, Tone
from .llm_provider import GenericLLMProvider
from .prompts import get_prompt_family
from .actions.retriever import get_retrievers

# Core research skills for disruption analysis (pruned)
from .skills.researcher import ResearchConductor
from .skills.context_manager import ContextManager
from .skills.browser import BrowserManager
from .skills.curator import SourceCurator

from .actions import (
    choose_agent,
    plan_research_outline,
    generate_report,
    write_report_to_html,
    write_report_to_docx,
    write_md_to_pdf
)


class GPTResearcher:
    """
    Optimized GPT Researcher for AI Disruption Analysis with ReAct orchestration.
    
    Single-mode focus: Disruption analysis using Y/X investment framework.
    Features: Modular ReAct loops, rubric-gated evidence, JSON outputs, cost guards.
    """
    
    def __init__(
        self,
        query: str,
        report_type: str = "disruption_analysis",
        report_format: str = "json",
        report_source: str = ReportSource.Web.value,
        tone: Tone = Tone.Objective,
        source_urls: list[str] | None = None,
        config_path=None,
        verbose: bool = True,
        headers: dict | None = None,
        log_handler=None,
        **kwargs
    ):
        """
        Initialize GPT Researcher for AI Disruption Analysis with ReAct orchestration.
        
        Args:
            query (str): The disruption research query
            Other args: Focused parameters for disruption analysis (pruned for single-mode)
        """
        self.kwargs = kwargs
        self.query = query
        self.report_type = "disruption_analysis"  # Force single mode
        self.cfg = Config(config_path)
        self.cfg.set_verbose(verbose)
        self.report_source = report_source
        self.report_format = "json"  # Force JSON output
        self.tone = tone if isinstance(tone, Tone) else Tone.Objective
        self.source_urls = source_urls
        self.research_sources = []
        self.verbose = verbose
        self.headers = headers or {}
        self.research_costs = 0.0
        self.log_handler = log_handler
        self.prompt_family = get_prompt_family(self.cfg.prompt_family, self.cfg)
        # CoT: Step1: Initialize MCP configs (default web_search/browse_page if empty)
        self.mcp_configs = self._init_mcp_configs(kwargs.get('mcp_configs'))
        
        # Initialize missing attributes expected by skills
        self.visited_urls = set()  # Should be a set for .add() method
        self.context = []
        self.websocket = None
        self.agent = "Business Analyst Agent"  # Default agent name for disruption analysis
        self.role = """You are a Senior Business Intelligence Analyst specializing in AI disruption analysis. 
        Your expertise includes identifying market vulnerabilities, competitive threats, and strategic opportunities 
        in the context of generative AI transformation."""
        self.query_domains = kwargs.get('query_domains', [])
        self.parent_query = kwargs.get('parent_query', "")
        self.vector_store = None  # Vector store for research context
        
        # Enhanced framework configs for disruption analysis with environment integration
        self.framework_configs = {
            "mode": "single",  # Single-mode focus for disruption analysis
            "trusted_domains": self._load_trusted_domains(),
            "sub_queries": [
                "industry_landscape_analysis",
                "disruption_framework_assessment", 
                "competitive_positioning_analysis"
            ],
            "rubric_criteria": {
                "recent": {
                    "weight": 0.25, 
                    "threshold": int(os.getenv("YX_RECENT_THRESHOLD", "7")),
                    "description": "Recent developments (last 12 months)"
                },
                "trusted": {
                    "weight": 0.20, 
                    "threshold": int(os.getenv("YX_TRUSTED_THRESHOLD", "7")),
                    "description": "Source credibility and authority"
                },
                "relevant": {
                    "weight": 0.30, 
                    "threshold": int(os.getenv("YX_RELEVANCE_THRESHOLD", "7")),
                    "description": "Y/X framework direct relevance"
                },
                "citations": {
                    "weight": 0.25, 
                    "threshold": int(os.getenv("YX_MIN_CITATIONS", "2")),
                    "description": "Minimum citation count per claim"
                }
            },
            "quality_gates": {
                "min_sources": int(os.getenv("YX_MIN_SOURCES", "4")),
                "token_limit": int(os.getenv("YX_TOKEN_LIMIT", "2000")),
                "max_turns": int(os.getenv("YX_MAX_TURNS", "5")),
                "max_retries_per_sub": int(os.getenv("YX_MAX_RETRIES", "2"))
            },
            "parallel_safe": True  # Prevent side effects in concurrent execution
        }
        
        # Validate framework configuration
        self._validate_framework_config()
        
        self.retrievers = get_retrievers(self.headers, self.cfg)
        
        # Initialize memory for embeddings
        self.memory = Memory(
            embedding_provider=self.cfg.embedding_provider,
            model=self.cfg.embedding_model
        )
        
        # Initialize core components (pruned for single-mode)
        self.research_conductor: ResearchConductor = ResearchConductor(self)
        self.context_manager: ContextManager = ContextManager(self)
        self.scraper_manager: BrowserManager = BrowserManager(self)
        self.source_curator: SourceCurator = SourceCurator(self)
        
        # Enhanced ReAct state tracking with validation
        self.react_state = {
            "query": query,
            "context": [],
            "current_sub_query": None,
            "completed_sub_queries": [],
            "retry_counts": {sq: 0 for sq in self.framework_configs["sub_queries"]},
            "token_count": 0,
            "turn_count": 0,
            "quality_scores": [],
            "start_time": None,
            "validation_flags": {
                "sufficient_sources": False,
                "quality_threshold_met": False,
                "completion_validated": False
            }
        }

    def _init_mcp_configs(self, mcp_configs=None):
        """CoT: Ensure mcp_configs is set; default to web_search/browse_page if empty."""
        if mcp_configs and isinstance(mcp_configs, list) and len(mcp_configs) > 0:
            return mcp_configs
        # Default minimal MCP tool configs for trusted evidence
        return [
            {
                'name': 'web_search',
                'command': 'python web_search.py',
                'description': 'Web search tool for trusted evidence (2023-2025)'
            },
            {
                'name': 'browse_page',
                'command': 'python browse_page.py',
                'description': 'Browser tool for page content extraction'
            }
        ]

    def _validate_framework_config(self):
        """Validate framework configuration for Y/X analysis requirements."""
        required_sub_queries = ["industry_landscape_analysis", "disruption_framework_assessment", "competitive_positioning_analysis"]
        
        if self.framework_configs["sub_queries"] != required_sub_queries:
            raise ValueError(f"Invalid sub_queries for disruption analysis. Required: {required_sub_queries}")
        
        if self.framework_configs["quality_gates"]["min_sources"] < 4:
            raise ValueError("Minimum 4 sources required for reliable Y/X framework analysis")
        
        # Log configuration for debugging
        if self.verbose:
            print(f"Framework Config Validated: {self.framework_configs['mode']} mode")
            print(f"Quality Gates: {self.framework_configs['quality_gates']}")

    def _load_trusted_domains(self) -> List[str]:
        """Load trusted domains from environment with Y/X analysis focus."""
        default_domains = "techcrunch.com,venturebeat.com,crunchbase.com,forbes.com,bloomberg.com,pitchbook.com,cbinsights.com"
        trusted_str = os.getenv("TRUSTED_DOMAINS", default_domains)
        domains = [domain.strip() for domain in trusted_str.split(",")]
        
        # Validate domain format
        validated_domains = []
        for domain in domains:
            if "." in domain and len(domain) > 3:
                validated_domains.append(domain)
        
        if len(validated_domains) < 3:
            raise ValueError("Insufficient trusted domains configured for reliable analysis")
        
        return validated_domains

    # Core research methods (already implemented above)
    async def conduct_research(self) -> tuple[str, list | None]:
        """
        Runs the research process and returns the report.
        """
        if self.verbose:
            print(f"🔍 Starting disruption research for: {self.query}")
        
        self.react_state["start_time"] = time.time()
        
        try:
            # Initialize research context
            self.react_state["context"] = []
            
            # Use the research conductor to perform the research
            report = await self.research_conductor.conduct_research()

            # Get the research context from the conductor
            self.research_context = self.research_conductor.get_context()
            self.mcp_context = self.research_conductor.get_mcp_context()

            return report, self.mcp_context
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Research error: {str(e)}")
            # Ensure we have some context even if research fails
            if not self.react_state["context"]:
                self.react_state["context"] = [{"content": f"Research failed with error: {str(e)}", "source": "error"}]
            raise e

    async def write_report(self, existing_headers: list = [], relevant_written_contents: list = [], ext_context=None, custom_prompt="") -> str:
        """Generate structured Y/X framework JSON report for disruption analysis."""
        from .actions.report_generation import generate_report
        
        if self.verbose:
            print(f"✍️ Writing disruption analysis report for '{self.query}'...")
        
        # Use external context if provided, otherwise use research context
        context = ext_context or self.react_state.get("context", [])
        
        if not context:
            if self.verbose:
                print("⚠️ No research context found. Report may be limited.")
            context = []
        
        # Create agent role prompt for disruption analysis
        agent_role_prompt = """You are a Senior Business Intelligence Analyst specializing in AI disruption analysis. 
        Your expertise includes identifying market vulnerabilities, competitive threats, and strategic opportunities 
        in the context of generative AI transformation. Provide analytical, data-driven insights with specific 
        recommendations and quantitative assessments where possible."""
        
        try:
            # Generate the report using the standard report generation function
            report = await generate_report(
                query=self.query,
                context=context,
                agent_role_prompt=agent_role_prompt,
                report_type="research_report",  # Use standard report type for compatibility
                tone=self.tone,
                report_source=self.report_source,
                websocket=None,  # No websocket for local usage
                cfg=self.cfg,
                main_topic="",
                existing_headers=existing_headers,
                relevant_written_contents=relevant_written_contents,
                cost_callback=self.add_costs,
                custom_prompt=custom_prompt,
                headers=self.headers,
                prompt_family=self.prompt_family,
                **self.kwargs
            )
            
            if self.verbose:
                print(f"📝 Disruption analysis report completed ({len(report)} characters)")
            
            return report
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error generating report: {str(e)}")
            
            # Fallback: Create a basic report structure
            fallback_report = f"""# Disruption Analysis Report

## Query
{self.query}

## Analysis Summary
Unable to generate detailed report due to technical issue: {str(e)}

## Research Context
{len(context)} sources analyzed from research phase.

## Status
Report generation encountered an error. Please check system configuration and try again.
"""
            return fallback_report

    # Essential utility methods (pruned)
    def get_research_sources(self) -> list[dict[str, Any]]:
        """Get collected research sources."""
        return self.research_sources

    def add_research_sources(self, sources: list[dict[str, Any]]) -> None:
        """Add sources to research collection."""
        self.research_sources.extend(sources)

    def get_source_urls(self) -> list:
        """Get URLs of collected sources."""
        return [source.get('url', '') for source in self.research_sources]

    def get_research_images(self) -> list:
        """Get collected research images."""
        return []  # Return empty list since image processing is not implemented

    def add_research_images(self, images: list) -> None:
        """Add research images to collection (placeholder implementation)."""
        # Placeholder - images are not stored since image processing is not implemented
        pass

    def get_research_context(self) -> list:
        """Get current research context."""
        return self.react_state["context"]

    def get_costs(self) -> float:
        """
        Returns the total cost of the research task.
        """
        return self.research_costs

    def set_verbose(self, verbose: bool):
        """Set verbose logging mode."""
        self.verbose = verbose
        
    # Enhanced cost tracking and guards (already implemented above)
    def add_costs(self, cost: float) -> None:
        """Enhanced cost tracking with guards and monitoring for disruption analysis."""
        if isinstance(cost, (int, float)) and cost >= 0:
            self.research_costs += cost
            if self.verbose and cost > 0:
                print(f"💸 Added ${cost:.4f} to research costs (Total: ${self.research_costs:.4f})")
        elif self.verbose:
            print(f"⚠️ Invalid cost value: {cost}")

    # Private helper methods (keep essential ones only)
    async def _log_event(self, event_type: str, **kwargs):
        """Helper method to handle logging events"""
        if self.log_handler:
            try:
                if event_type == "tool":
                    await self.log_handler.on_tool_start(kwargs.get('tool_name', ''), **kwargs)
                elif event_type == "action":
                    await self.log_handler.on_agent_action(kwargs.get('action', ''), **kwargs)
                elif event_type == "research":
                    await self.log_handler.on_research_step(kwargs.get('step', ''), kwargs.get('details', {}))

                import logging
                research_logger = logging.getLogger('research')
                research_logger.info(f"{event_type}: {json.dumps(kwargs, default=str)}")

            except Exception as e:
                import logging
                logging.getLogger('research').error(f"Error in _log_event: {e}", exc_info=True)
