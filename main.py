#!/usr/bin/env python3

import asyncio
import argparse
from gpt_researcher.agent import GPTResearcher
from gpt_researcher.utils.enum import ReportSource, ReportType, Tone

async def main():
    parser = argparse.ArgumentParser(description='GPT Researcher')
    parser.add_argument('--query', type=str, required=True, help='Research query')
    parser.add_argument('--report_type', type=str, default='research_report', help='Type of report')
    parser.add_argument('--retriever', type=str, default='tavily', help='Retriever to use (tavily, mcp, hybrid)')
    
    args = parser.parse_args()
    
    # Configure MCP if requested
    mcp_configs = None
    if args.retriever in ['mcp', 'hybrid']:
        mcp_configs = [
            {
                "name": "langchain_mcp_server",
                "command": "http://localhost:8001/mcp",
                "description": "LangChain MCP server for research tools"
            }
        ]
        print(f"🔧 Using {args.retriever} mode with LangChain MCP server")
    
    # Create researcher instance
    researcher = GPTResearcher(
        query=args.query,
        report_type=args.report_type,
        mcp_configs=mcp_configs
    )
    
    # Configure retrievers based on mode
    if args.retriever == 'mcp':
        # MCP only mode (original behavior)
        from gpt_researcher.retrievers.mcp.retriever import MCPRetriever
        researcher.retrievers = [MCPRetriever]
        print(f"🔍 MCP-only mode: {[r.__name__ for r in researcher.retrievers]}")
    elif args.retriever == 'hybrid':
        # Hybrid mode: Add MCP alongside existing retrievers
        from gpt_researcher.retrievers.mcp.retriever import MCPRetriever
        researcher.retrievers.append(MCPRetriever)
        print(f"🔍 Hybrid mode: {[r.__name__ for r in researcher.retrievers]}")
    else:
        # Default mode: Use standard retrievers only
        print(f"🔍 Standard mode: {[r.__name__ for r in researcher.retrievers]}")
    
    # Run research
    print(f"🚀 Starting research on: {args.query}")
    result = await researcher.conduct_research()
    
    if isinstance(result, tuple):
        report, mcp_context = result
        print(f"📊 MCP Context: {len(mcp_context) if mcp_context else 0} sources")
        print(f"📊 Total Research Cost: ${researcher.get_costs()}")
    else:
        report = result
        print("📊 No MCP context returned")
    
    print("\n" + "="*80)
    print("📝 RESEARCH REPORT")
    print("="*80)
    print(report)
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
