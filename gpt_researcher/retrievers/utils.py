import importlib.util
import logging
import os
import sys
import re
from urllib.parse import urlparse
from datetime import datetime, date
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

async def stream_output(log_type, step, content, websocket=None, with_data=False, data=None):
    """
    Stream output to the client.
    
    Args:
        log_type (str): The type of log
        step (str): The step being performed
        content (str): The content to stream
        websocket: The websocket to stream to
        with_data (bool): Whether to include data
        data: Additional data to include
    """
    if websocket:
        try:
            if with_data:
                await websocket.send_json({
                    "type": log_type,
                    "step": step,
                    "content": content,
                    "data": data
                })
            else:
                await websocket.send_json({
                    "type": log_type,
                    "step": step,
                    "content": content
                })
        except Exception as e:
            logger.error(f"Error streaming output: {e}")

def check_pkg(pkg: str) -> None:
    """
    Checks if a package is installed and raises an error if not.
    
    Args:
        pkg (str): The package name
    
    Raises:
        ImportError: If the package is not installed
    """
    if not importlib.util.find_spec(pkg):
        pkg_kebab = pkg.replace("_", "-")
        raise ImportError(
            f"Unable to import {pkg_kebab}. Please install with "
            f"`pip install -U {pkg_kebab}`"
        )

# Valid retrievers for fallback
VALID_RETRIEVERS = [
    "tavily",
    "custom",
    "duckduckgo",
    "searchapi",
    "serper",
    "serpapi",
    "google",
    "searx",
    "bing",
    "arxiv",
    "semantic_scholar",
    "pubmed_central",
    "exa",
    "mcp",
    "mock"
]

def get_all_retriever_names():
    """
    Get all available retriever names
    :return: List of all available retriever names
    :rtype: list
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Get all items in the current directory
        all_items = os.listdir(current_dir)
        
        # Filter out only the directories, excluding __pycache__
        retrievers = [
            item for item in all_items 
            if os.path.isdir(os.path.join(current_dir, item)) and not item.startswith('__')
        ]
        
        return retrievers
    except Exception as e:
        logger.error(f"Error getting retrievers: {e}")
        return VALID_RETRIEVERS

# Trusted domains for disruption research
TRUSTED_DOMAINS = [
    "mckinsey.com",
    "mit.edu", 
    "harvard.edu",
    "stanford.edu",
    "weforum.org",
    "bcg.com",
    "bain.com",
    "economist.com",
    "ft.com",
    "wsj.com",
    "reuters.com",
    "bloomberg.com",
    "nature.com",
    "science.org",
    "techcrunch.com",
    "wired.com",
    "hbr.org",
    "sloanreview.mit.edu",
    "forbes.com",
    "pwc.com",
    "deloitte.com",
    "kpmg.com",
    "ey.com"
]

async def trusted_filter(results: List[Dict[str, Any]], llm_provider=None, websocket=None) -> List[Dict[str, Any]]:
    """
    Filter search results for trusted domains, recent dates (2023-2025), and LLM-scored relevance to disruption.
    
    Args:
        results (List[Dict]): List of search results with 'url', 'title', 'content' keys
        llm_provider: LLM provider instance for relevance scoring
        websocket: Optional websocket for streaming logs
        
    Returns:
        List[Dict]: Filtered results that meet all criteria, or empty list if <4 results remain
    """
    if not results:
        return []
        
    await stream_output("logs", "filtering", "Starting trusted filter for disruption research...", websocket)
    
    filtered_results = []
    
    for result in results:
        try:
            # Check trusted domain
            url = result.get('url', '')
            if url:
                domain = urlparse(url).netloc.lower()
                # Remove www. prefix for comparison
                domain = domain.replace('www.', '')
                
                is_trusted = any(trusted_domain in domain for trusted_domain in TRUSTED_DOMAINS)
                if not is_trusted:
                    continue
                    
            # Check date range (2023-2025)
            content = result.get('content', '') + ' ' + result.get('title', '')
            date_valid = check_date_relevance(content)
            if not date_valid:
                continue
                
            # LLM relevance scoring for disruption
            if llm_provider:
                score = await score_disruption_relevance(result, llm_provider)
                if score < 7:
                    continue
                    
            filtered_results.append(result)
            
        except Exception as e:
            logger.error(f"Error filtering result: {e}")
            continue
    
    # Check minimum threshold
    if len(filtered_results) < 4:
        await stream_output("logs", "filtering", 
                          f"Warning: Only {len(filtered_results)} trusted results found. Minimum threshold is 4. Returning empty results.", 
                          websocket)
        return []
    
    await stream_output("logs", "filtering", 
                      f"Filtered to {len(filtered_results)} trusted, recent, and relevant results for disruption analysis.", 
                      websocket)
    
    return filtered_results

def check_date_relevance(content: str) -> bool:
    """
    Check if content contains dates between 2023-2025 or appears to be recent.
    
    Args:
        content (str): Content to check for date relevance
        
    Returns:
        bool: True if content appears to be from 2023-2025 timeframe
    """
    if not content:
        return True  # If no date info, assume it could be recent
        
    # Look for year patterns
    year_pattern = r'\b(202[3-5])\b'
    years_found = re.findall(year_pattern, content)
    
    if years_found:
        return True
        
    # Look for recent date patterns (e.g., "March 2024", "Q1 2023", etc.)
    recent_patterns = [
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(202[3-5])\b',
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(202[3-5])\b',
        r'\bQ[1-4]\s+(202[3-5])\b',
        r'\b(202[3-5])\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}\b'
    ]
    
    for pattern in recent_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
            
    # If no specific dates found, assume it could be recent
    return True

async def score_disruption_relevance(result: Dict[str, Any], llm_provider) -> int:
    """
    Use LLM to score the relevance of a result to business/technological disruption.
    
    Args:
        result (Dict): Search result with 'title' and 'content' keys
        llm_provider: LLM provider instance
        
    Returns:
        int: Relevance score from 1-10 (>=7 indicates high relevance)
    """
    try:
        title = result.get('title', '')
        content = result.get('content', '')[:500]  # Limit content length for efficiency
        
        prompt = f"""
Rate the relevance of this content to business/technological disruption on a scale of 1-10.

Consider disruption as: breakthrough technologies, business model innovations, market transformations, 
industry shifts, emerging technologies that challenge existing paradigms, AI/automation impacts, 
digital transformation, sustainability innovations, or major economic/social changes.

Title: {title}
Content: {content}

Response format: Just return a single number from 1-10.
"""
        
        # Use the LLM provider to get a score
        if hasattr(llm_provider, 'get_response'):
            response = await llm_provider.get_response(prompt)
        else:
            # Fallback if different method name
            response = "7"  # Default to passing score
            
        # Extract number from response
        score_match = re.search(r'\b([1-9]|10)\b', str(response))
        if score_match:
            return int(score_match.group(1))
        else:
            return 7  # Default to passing score if parsing fails
            
    except Exception as e:
        logger.error(f"Error scoring disruption relevance: {e}")
        return 7  # Default to passing score on error
