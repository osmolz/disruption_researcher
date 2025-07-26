"""
MCP Streaming Utilities Module

Handles websocket streaming and logging for MCP operations with cost tracking and quality metrics.
"""
import asyncio
import logging
from typing import Any, Optional, Dict, List

logger = logging.getLogger(__name__)


class MCPStreamer:
    """
    Handles streaming output for MCP operations with enhanced cost and quality tracking.
    
    Responsible for:
    - Streaming logs to websocket with cost information
    - Synchronous/asynchronous logging with metrics
    - Error handling in streaming
    - Evidence quality and token usage tracking
    """

    def __init__(self, websocket=None):
        """
        Initialize the MCP streamer.
        
        Args:
            websocket: WebSocket for streaming output
        """
        self.websocket = websocket
        self.total_tokens_used = 0
        self.total_cost_estimate = 0.0
        self.evidence_scores = []

    async def stream_log(self, message: str, data: Any = None):
        """Stream a log message to the websocket if available."""
        logger.info(message)
        
        if self.websocket:
            try:
                from ..actions.utils import stream_output
                await stream_output(
                    type="logs", 
                    content="mcp_retriever", 
                    output=message, 
                    websocket=self.websocket,
                    metadata=data
                )
            except Exception as e:
                logger.error(f"Error streaming log: {e}")
                
    def stream_log_sync(self, message: str, data: Any = None):
        """Synchronous version of stream_log for use in sync contexts."""
        logger.info(message)
        
        if self.websocket:
            try:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(self.stream_log(message, data))
                    else:
                        loop.run_until_complete(self.stream_log(message, data))
                except RuntimeError:
                    logger.debug("Could not stream log: no running event loop")
            except Exception as e:
                logger.error(f"Error in sync log streaming: {e}")

    async def stream_stage_start(self, stage: str, description: str):
        """Stream the start of a research stage."""
        await self.stream_log(f"🔧 {stage}: {description}")

    async def stream_stage_complete(self, stage: str, result_count: int = None):
        """Stream the completion of a research stage."""
        if result_count is not None:
            await self.stream_log(f"✅ {stage} completed: {result_count} results")
        else:
            await self.stream_log(f"✅ {stage} completed")

    async def stream_tool_selection(self, selected_count: int, total_count: int):
        """Stream tool selection information with self-consistency details."""
        await self.stream_log(f"🧠 Using LLM with self-consistency to select {selected_count} most relevant tools from {total_count} available")

    async def stream_tool_selection_details(self, tool_name: str, median_score: float, vote_count: int):
        """Stream detailed tool selection results."""
        await self.stream_log(f"🎯 Selected: {tool_name} (score: {median_score:.1f}, consensus: {vote_count}/3)")

    async def stream_tool_execution(self, tool_name: str, step: int, total: int):
        """Stream tool execution progress with parallel indication."""
        if total > 1:
            await self.stream_log(f"🔍 Executing tool {step}/{total} in parallel: {tool_name}")
        else:
            await self.stream_log(f"🔍 Executing tool: {tool_name}")

    async def stream_parallel_execution_start(self, tool_count: int):
        """Stream start of parallel tool execution."""
        await self.stream_log(f"⚡ Starting parallel execution of {tool_count} tools")

    async def stream_parallel_execution_complete(self, successful: int, total: int, duration_ms: int = None):
        """Stream completion of parallel tool execution."""
        if duration_ms:
            await self.stream_log(f"⚡ Parallel execution completed: {successful}/{total} tools successful ({duration_ms}ms)")
        else:
            await self.stream_log(f"⚡ Parallel execution completed: {successful}/{total} tools successful")

    async def stream_evidence_evaluation(self, source: str, score: float):
        """Stream evidence quality evaluation results."""
        if score >= 8.0:
            icon = "🏆"
        elif score >= 7.0:
            icon = "✅"
        elif score >= 5.0:
            icon = "⚠️"
        else:
            icon = "❌"
            
        await self.stream_log(f"{icon} Evidence quality: {source} scored {score:.1f}/10")
        self.evidence_scores.append(score)

    async def stream_cost_update(self, tokens_used: int, estimated_cost: float, operation: str = ""):
        """Stream cost tracking information."""
        self.total_tokens_used += tokens_used
        self.total_cost_estimate += estimated_cost
        
        operation_text = f" ({operation})" if operation else ""
        await self.stream_log(f"💰 Tokens: {tokens_used:,}, Cost: ${estimated_cost:.4f}{operation_text}")

    async def stream_cumulative_cost(self):
        """Stream cumulative cost information."""
        await self.stream_log(f"💰 Total session: {self.total_tokens_used:,} tokens, ${self.total_cost_estimate:.4f}")

    async def stream_research_results(self, result_count: int, total_chars: int = None):
        """Stream research results summary with quality metrics."""
        if total_chars:
            await self.stream_log(f"✅ MCP research completed: {result_count} results obtained ({total_chars:,} chars)")
        else:
            await self.stream_log(f"✅ MCP research completed: {result_count} results obtained")
            
        # Stream evidence quality summary
        if self.evidence_scores:
            avg_quality = sum(self.evidence_scores) / len(self.evidence_scores)
            high_quality_count = sum(1 for score in self.evidence_scores if score >= 7.0)
            await self.stream_log(f"📊 Evidence quality: {avg_quality:.1f}/10 average, {high_quality_count}/{len(self.evidence_scores)} high-quality sources")

    async def stream_yx_framework_summary(self, y_indicators: int, x_indicators: int, quality_sources: int):
        """Stream Y/X framework analysis summary."""
        await self.stream_log(f"📈 Y/X Framework: {y_indicators} Y-axis indicators, {x_indicators} X-axis indicators from {quality_sources} quality sources")

    async def stream_rubric_results(self, tool_name: str, y_score: float, x_score: float, quality_score: float, overall: float):
        """Stream detailed rubric evaluation results."""
        await self.stream_log(f"📋 {tool_name} rubric: Y:{y_score:.1f} X:{x_score:.1f} Q:{quality_score:.1f} → {overall:.1f}/10")

    async def stream_self_consistency_vote(self, variant_count: int, consensus_tools: int, rejected_tools: int):
        """Stream self-consistency voting results."""
        await self.stream_log(f"🗳️ Self-consistency vote: {variant_count} variants, {consensus_tools} consensus tools, {rejected_tools} rejected")

    async def stream_fallback_warning(self, reason: str):
        """Stream fallback mechanism warnings."""
        await self.stream_log(f"⚠️ Fallback activated: {reason}")

    async def stream_quality_threshold_warning(self, tool_name: str, score: float, threshold: float):
        """Stream quality threshold warnings."""
        await self.stream_log(f"⚠️ Quality threshold: {tool_name} scored {score:.1f} < {threshold:.1f}, excluded")

    async def stream_error(self, error_msg: str):
        """Stream error messages."""
        await self.stream_log(f"❌ {error_msg}")

    async def stream_warning(self, warning_msg: str):
        """Stream warning messages."""
        await self.stream_log(f"⚠️ {warning_msg}")

    async def stream_info(self, info_msg: str):
        """Stream informational messages."""
        await self.stream_log(f"ℹ️ {info_msg}")

    async def stream_debug_info(self, component: str, message: str):
        """Stream debug information for development."""
        await self.stream_log(f"🔧 {component}: {message}")

    async def stream_research_strategy(self, strategy: str):
        """Stream research strategy information."""
        await self.stream_log(f"🎯 Research strategy: {strategy}")

    async def stream_source_validation(self, source: str, is_valid: bool, reason: str = ""):
        """Stream source validation results."""
        if is_valid:
            await self.stream_log(f"✅ Source validated: {source}")
        else:
            reason_text = f" ({reason})" if reason else ""
            await self.stream_log(f"❌ Source rejected: {source}{reason_text}")

    async def stream_progress_update(self, current: int, total: int, operation: str):
        """Stream progress updates for long operations."""
        percentage = (current / total * 100) if total > 0 else 0
        await self.stream_log(f"📊 {operation}: {current}/{total} ({percentage:.1f}%)")

    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current session metrics.
        
        Returns:
            Dict[str, Any]: Session summary with costs and quality metrics
        """
        summary = {
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost_estimate,
            "evidence_count": len(self.evidence_scores),
            "average_evidence_quality": sum(self.evidence_scores) / len(self.evidence_scores) if self.evidence_scores else 0,
            "high_quality_sources": sum(1 for score in self.evidence_scores if score >= 7.0),
            "low_quality_sources": sum(1 for score in self.evidence_scores if score < 5.0)
        }
        return summary

    async def stream_session_summary(self):
        """Stream final session summary with quality warnings."""
        summary = self.get_session_summary()
        
        await self.stream_log("📊 Session Summary:")
        await self.stream_log(f"💰 Total cost: ${summary['total_cost']:.4f} ({summary['total_tokens']:,} tokens)")
        await self.stream_log(f"📋 Evidence: {summary['evidence_count']} sources, avg quality {summary['average_evidence_quality']:.1f}/10")
        await self.stream_log(f"🏆 Quality breakdown: {summary['high_quality_sources']} high, {summary['low_quality_sources']} low quality")
        
        # *** NEW: Quality-based warnings and recommendations ***
        avg_quality = summary['average_evidence_quality']
        evidence_count = summary['evidence_count']
        
        if avg_quality < 7.0:
            await self.stream_warning(f"Low evidence quality detected (avg: {avg_quality:.1f}/10)")
            await self.stream_warning("Recommendation: Re-run research with more specialized tools")
            
            # Specific recommendations based on quality issues
            if evidence_count < 5:
                await self.stream_info("💡 Suggestion: Increase tool selection to gather more sources")
            
            high_quality_ratio = summary['high_quality_sources'] / evidence_count if evidence_count > 0 else 0
            if high_quality_ratio < 0.3:
                await self.stream_info("💡 Suggestion: Focus on tools that access authoritative sources (academic, industry reports)")
            
            if summary['low_quality_sources'] > summary['high_quality_sources']:
                await self.stream_info("💡 Suggestion: Review tool selection criteria - prioritize quality over quantity")
                
        elif avg_quality >= 8.0:
            await self.stream_log("🎯 Excellent evidence quality achieved!")
            
        # Additional quality insights
        if evidence_count > 0:
            if summary['high_quality_sources'] == 0:
                await self.stream_warning("No high-quality sources found - consider different research strategy")
            elif summary['low_quality_sources'] == 0 and evidence_count >= 5:
                await self.stream_log("🌟 Perfect quality: All sources meet high standards!")
                
        # Cost efficiency insights
        if summary['total_cost'] > 0.50:  # Threshold for cost warning
            cost_per_source = summary['total_cost'] / evidence_count if evidence_count > 0 else 0
            await self.stream_info(f"💰 Cost efficiency: ${cost_per_source:.3f} per source")
            
            if cost_per_source > 0.10:
                await self.stream_warning("High cost per source - consider optimizing tool selection")
                
        await self.stream_log("📈 Session analysis complete")

    def reset_session_metrics(self):
        """Reset session tracking metrics."""
        self.total_tokens_used = 0
        self.total_cost_estimate = 0.0
        self.evidence_scores = [] 