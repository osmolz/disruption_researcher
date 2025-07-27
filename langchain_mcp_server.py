#!/usr/bin/env python3
"""
LangChain MCP Tool Server for GPT-Researcher
A simple FastAPI server that exposes LangChain tools via MCP-compatible endpoints.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from datetime import datetime

# LangChain imports
try:
    from langchain.tools import DuckDuckGoSearchRun, WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchResults
    from langchain.tools import Tool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    print("⚠️ LangChain not installed. Install with: pip install langchain langchain-community duckduckgo-search wikipedia")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangChain MCP Tool Server", version="1.0.0")

# Request/Response models
class MCPRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    tool_name: Optional[str] = None

class MCPResult(BaseModel):
    title: str
    href: str
    body: str
    source: str
    date: str

class MCPResponse(BaseModel):
    results: List[MCPResult]
    tool_used: str
    query: str
    timestamp: str

class LangChainMCPServer:
    """LangChain-based MCP tool server for research tasks."""
    
    def __init__(self):
        self.tools = {}
        self._init_tools()
    
    def _init_tools(self):
        """Initialize LangChain tools for research."""
        if not HAS_LANGCHAIN:
            logger.error("LangChain not available - server will have limited functionality")
            return
        
        try:
            # DuckDuckGo Search tool
            ddg_search = DuckDuckGoSearchResults(
                max_results=10,
                output_format="list"
            )
            self.tools["web_search"] = Tool(
                name="web_search",
                description="Search the web using DuckDuckGo for recent information and trusted sources",
                func=self._wrap_ddg_search(ddg_search)
            )
            
            # Wikipedia tool
            wikipedia = WikipediaQueryRun(
                api_wrapper=WikipediaAPIWrapper(
                    top_k_results=3,
                    doc_content_chars_max=2000
                )
            )
            self.tools["wikipedia"] = Tool(
                name="wikipedia",
                description="Search Wikipedia for authoritative information on topics",
                func=self._wrap_wikipedia(wikipedia)
            )
            
            # Browse page tool (simplified - uses requests)
            self.tools["browse_page"] = Tool(
                name="browse_page",
                description="Browse and extract content from a specific webpage URL",
                func=self._browse_page_tool
            )
            
            logger.info(f"✅ Initialized {len(self.tools)} LangChain tools")
            
        except Exception as e:
            logger.error(f"Error initializing tools: {e}")
            # Fallback dummy tool
            self.tools["dummy"] = Tool(
                name="dummy",
                description="Dummy tool for testing when other tools fail",
                func=self._dummy_tool
            )
    
    def _wrap_ddg_search(self, ddg_tool):
        """Wrap DuckDuckGo search to return structured results."""
        def search_wrapper(query: str) -> List[Dict[str, str]]:
            try:
                results = ddg_tool.run(query)
                if isinstance(results, str):
                    # Parse string results
                    return [{
                        'title': f'DuckDuckGo Search: {query}',
                        'href': 'https://duckduckgo.com',
                        'body': results[:1000],
                        'source': 'DuckDuckGo',
                        'date': datetime.now().strftime('%Y-%m-%d')
                    }]
                elif isinstance(results, list):
                    # Parse list results
                    formatted_results = []
                    for item in results[:5]:  # Limit to 5 results
                        if isinstance(item, dict):
                            formatted_results.append({
                                'title': item.get('title', 'No title'),
                                'href': item.get('link', item.get('url', 'No URL')),
                                'body': item.get('snippet', item.get('body', 'No content'))[:1000],
                                'source': 'DuckDuckGo',
                                'date': datetime.now().strftime('%Y-%m-%d')
                            })
                    return formatted_results if formatted_results else self._dummy_results(query, "DuckDuckGo")
                else:
                    return self._dummy_results(query, "DuckDuckGo")
            except Exception as e:
                logger.error(f"DuckDuckGo search error: {e}")
                return self._dummy_results(query, "DuckDuckGo (Error)")
        
        return search_wrapper
    
    def _wrap_wikipedia(self, wiki_tool):
        """Wrap Wikipedia search to return structured results."""
        def wiki_wrapper(query: str) -> List[Dict[str, str]]:
            try:
                result = wiki_tool.run(query)
                return [{
                    'title': f'Wikipedia: {query}',
                    'href': f'https://en.wikipedia.org/wiki/{query.replace(" ", "_")}',
                    'body': result[:1500] if result else 'No Wikipedia content found',
                    'source': 'Wikipedia',
                    'date': datetime.now().strftime('%Y-%m-%d')
                }]
            except Exception as e:
                logger.error(f"Wikipedia search error: {e}")
                return self._dummy_results(query, "Wikipedia (Error)")
        
        return wiki_wrapper
    
    def _browse_page_tool(self, url: str) -> List[Dict[str, str]]:
        """Simple page browsing tool using requests."""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else 'No title'
            
            # Extract main content
            content = ""
            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'div'], limit=20):
                text = tag.get_text().strip()
                if len(text) > 50:  # Only include substantial text
                    content += text + " "
            
            return [{
                'title': title_text,
                'href': url,
                'body': content[:2000],  # Limit content length
                'source': 'Web Browse',
                'date': datetime.now().strftime('%Y-%m-%d')
            }]
            
        except Exception as e:
            logger.error(f"Browse page error for {url}: {e}")
            return [{
                'title': f'Browse Error: {url}',
                'href': url,
                'body': f'Failed to browse page: {str(e)}',
                'source': 'Web Browse (Error)',
                'date': datetime.now().strftime('%Y-%m-%d')
            }]
    
    def _dummy_tool(self, query: str) -> List[Dict[str, str]]:
        """Dummy tool for testing and fallback."""
        return self._dummy_results(query, "Dummy Tool")
    
    def _dummy_results(self, query: str, source: str) -> List[Dict[str, str]]:
        """Generate dummy results for testing/fallback."""
        return [{
            'title': f'LangChain MCP Result: {query}',
            'href': 'https://langchain-mcp.local/result',
            'body': f'This is a LangChain MCP server result for query: {query}. The tool {source} was used to generate this response.',
            'source': source,
            'date': datetime.now().strftime('%Y-%m-%d')
        }]
    
    async def execute_tool(self, tool_name: str, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Execute a specific tool with the given query."""
        if tool_name not in self.tools:
            logger.warning(f"Tool '{tool_name}' not found. Available: {list(self.tools.keys())}")
            # Use first available tool as fallback
            tool_name = list(self.tools.keys())[0] if self.tools else "dummy"
        
        if tool_name not in self.tools:
            return self._dummy_results(query, "No Tools Available")
        
        try:
            tool = self.tools[tool_name]
            logger.info(f"🔧 Executing tool '{tool_name}' for query: {query}")
            
            # Execute tool (some tools are async, some are sync)
            if asyncio.iscoroutinefunction(tool.func):
                results = await tool.func(query)
            else:
                results = tool.func(query)
            
            # Ensure results are in the expected format
            if not isinstance(results, list):
                results = [results] if results else []
            
            # Limit results
            results = results[:max_results]
            
            logger.info(f"✅ Tool '{tool_name}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            return self._dummy_results(query, f"{tool_name} (Error)")

# Initialize server
mcp_server = LangChainMCPServer()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "LangChain MCP Tool Server is running",
        "tools": list(mcp_server.tools.keys()),
        "version": "1.0.0"
    }

@app.get("/tools")
async def list_tools():
    """List available tools."""
    tools_info = []
    for name, tool in mcp_server.tools.items():
        tools_info.append({
            "name": name,
            "description": tool.description
        })
    return {"tools": tools_info}

@app.post("/mcp", response_model=MCPResponse)
async def mcp_endpoint(request: MCPRequest):
    """Main MCP endpoint that GPT-Researcher will call."""
    try:
        logger.info(f"🔍 MCP request: {request.query} (tool: {request.tool_name})")
        
        # Determine which tool to use
        tool_name = request.tool_name or "web_search"  # Default to web search
        
        # Execute the tool
        results = await mcp_server.execute_tool(
            tool_name=tool_name,
            query=request.query,
            max_results=request.max_results
        )
        
        # Format response
        mcp_results = []
        for result in results:
            mcp_results.append(MCPResult(
                title=result.get('title', 'No title'),
                href=result.get('href', 'No URL'),
                body=result.get('body', 'No content'),
                source=result.get('source', 'Unknown'),
                date=result.get('date', datetime.now().strftime('%Y-%m-%d'))
            ))
        
        response = MCPResponse(
            results=mcp_results,
            tool_used=tool_name,
            query=request.query,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ MCP response: {len(mcp_results)} results for '{request.query}'")
        return response
        
    except Exception as e:
        logger.error(f"MCP endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"MCP processing error: {str(e)}")

@app.post("/search")
async def search_endpoint(request: MCPRequest):
    """Direct search endpoint (alternative to /mcp)."""
    return await mcp_endpoint(request)

if __name__ == "__main__":
    print("🚀 Starting LangChain MCP Tool Server...")
    print("📋 Available tools:")
    for name, tool in mcp_server.tools.items():
        print(f"  • {name}: {tool.description}")
    print("\n🌐 Server will be available at: http://localhost:8001")
    print("🔧 MCP endpoint: http://localhost:8001/mcp")
    print("📖 API docs: http://localhost:8001/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    ) 