"""
MCP Client Management Module

Handles MCP client creation, configuration conversion, and connection management.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP_ADAPTERS = True
except ImportError:
    HAS_MCP_ADAPTERS = False

# Fallback OpenAI import for when MCP adapters aren't available
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


async def call_llm_fallback(
    prompt: str, 
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 500,
    cost_callback: callable = None
) -> str:
    """
    Fallback LLM call when langchain-mcp-adapters are not available.
    
    Args:
        prompt: The prompt to send
        model: OpenAI model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        cost_callback: Optional callback for cost tracking
        
    Returns:
        str: The LLM response
    """
    if not HAS_OPENAI:
        logger.error("Neither langchain-mcp-adapters nor openai library available")
        return ""
        
    try:
        client = openai.AsyncOpenAI()
        
        messages = [{"role": "user", "content": prompt}]
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        content = response.choices[0].message.content
        
        # Track costs if callback provided
        if cost_callback and response.usage:
            # Estimate cost based on token usage (rough calculation)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            estimated_cost = (input_tokens * 0.000005) + (output_tokens * 0.000015)
            cost_callback(estimated_cost)
        
        return content or ""
        
    except Exception as e:
        logger.error(f"Error in fallback LLM call: {e}")
        return ""


class MCPClientManager:
    """
    Manages MCP client lifecycle and configuration.
    
    Responsible for:
    - Converting GPT Researcher MCP configs to langchain format
    - Creating and managing MultiServerMCPClient instances
    - Handling client cleanup and resource management
    - Providing fallback functionality when adapters unavailable
    """

    def __init__(self, mcp_configs: List[Dict[str, Any]]):
        """
        Initialize the MCP client manager.
        
        Args:
            mcp_configs: List of MCP server configurations from GPT Researcher
        """
        self.mcp_configs = mcp_configs or []
        self._client = None
        self._client_lock = asyncio.Lock()

    def convert_configs_to_langchain_format(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert GPT Researcher MCP configs to langchain-mcp-adapters format.
        
        Returns:
            Dict[str, Dict[str, Any]]: Server configurations for MultiServerMCPClient
        """
        server_configs = {}
        
        for i, config in enumerate(self.mcp_configs):
            # Generate server name
            server_name = config.get("name", f"mcp_server_{i+1}")
            
            # Build the server config
            server_config = {}
            
            # Auto-detect transport type from URL if provided
            connection_url = config.get("connection_url")
            if connection_url:
                if connection_url.startswith(("wss://", "ws://")):
                    server_config["transport"] = "websocket"
                    server_config["url"] = connection_url
                elif connection_url.startswith(("https://", "http://")):
                    server_config["transport"] = "streamable_http"
                    server_config["url"] = connection_url
                else:
                    # Fallback to specified connection_type or stdio
                    connection_type = config.get("connection_type", "stdio")
                    server_config["transport"] = connection_type
                    if connection_type in ["websocket", "streamable_http", "http"]:
                        server_config["url"] = connection_url
            else:
                # No URL provided, use stdio (default) or specified connection_type
                connection_type = config.get("connection_type", "stdio")
                server_config["transport"] = connection_type
            
            # Handle stdio transport configuration
            if server_config.get("transport") == "stdio":
                if config.get("command"):
                    server_config["command"] = config["command"]
                    
                    # Handle server_args
                    server_args = config.get("args", [])
                    if isinstance(server_args, str):
                        server_args = server_args.split()
                    server_config["args"] = server_args
                    
                    # Handle environment variables
                    server_env = config.get("env", {})
                    if server_env:
                        server_config["env"] = server_env
                        
            # Add authentication if provided
            if config.get("connection_token"):
                server_config["token"] = config["connection_token"]
                
            server_configs[server_name] = server_config
            
        return server_configs

    async def get_or_create_client(self) -> Optional[object]:
        """
        Get or create a MultiServerMCPClient with proper lifecycle management.
        
        Returns:
            MultiServerMCPClient: The client instance or None if creation fails
        """
        async with self._client_lock:
            if self._client is not None:
                return self._client
                
            if not HAS_MCP_ADAPTERS:
                logger.error("langchain-mcp-adapters not installed - fallback mode enabled")
                return None
                
            if not self.mcp_configs:
                logger.error("No MCP server configurations found")
                return None
                
            try:
                # Convert configs to langchain format
                server_configs = self.convert_configs_to_langchain_format()
                logger.info(f"Creating MCP client for {len(server_configs)} server(s)")
                
                # Initialize the MultiServerMCPClient
                self._client = MultiServerMCPClient(server_configs)
                
                return self._client
                
            except Exception as e:
                logger.error(f"Error creating MCP client: {e}")
                return None

    async def close_client(self):
        """
        Properly close the MCP client and clean up resources with explicit cleanup.
        """
        async with self._client_lock:
            if self._client is not None:
                try:
                    # Check if the client has an explicit close method
                    if hasattr(self._client, 'close'):
                        logger.debug("Explicitly closing MCP client")
                        if asyncio.iscoroutinefunction(self._client.close):
                            await self._client.close()
                        else:
                            self._client.close()
                    elif hasattr(self._client, 'aclose'):
                        logger.debug("Explicitly closing MCP client with aclose")
                        await self._client.aclose()
                    else:
                        logger.debug("No explicit close method found, relying on garbage collection")
                        
                    # Additional cleanup for any open connections
                    if hasattr(self._client, '_clients'):
                        for client_name, client in self._client._clients.items():
                            if hasattr(client, 'close'):
                                try:
                                    if asyncio.iscoroutinefunction(client.close):
                                        await client.close()
                                    else:
                                        client.close()
                                    logger.debug(f"Closed individual client: {client_name}")
                                except Exception as e:
                                    logger.warning(f"Error closing client {client_name}: {e}")
                                    
                except Exception as e:
                    logger.error(f"Error during MCP client cleanup: {e}")
                finally:
                    # Always clear the reference
                    self._client = None
                    logger.debug("MCP client reference cleared")

    async def get_all_tools(self) -> List:
        """
        Get all available tools from MCP servers.
        
        Returns:
            List: All available MCP tools
        """
        client = await self.get_or_create_client()
        if not client:
            return []
            
        try:
            # Get tools from all servers
            all_tools = await client.get_tools()
            
            if all_tools:
                logger.info(f"Loaded {len(all_tools)} total tools from MCP servers")
                return all_tools
            else:
                logger.warning("No tools available from MCP servers")
                return []
                
        except Exception as e:
            logger.error(f"Error getting MCP tools: {e}")
            return [] 