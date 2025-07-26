from typing import Any, Dict, List, Optional
import requests
import os


class BrowsePageRetriever:
    """
    Browse Page Retriever for detailed content extraction
    Focuses on deep analysis of business disruption content from trusted sources
    """

    def __init__(self, query: str, query_domains=None):
        self.query = query
        self.urls = []  # URLs to browse
        self.trusted_domains = query_domains or [
            "mckinsey.com",
            "bain.com", 
            "sec.gov",
            "bcg.com"
        ]

    def add_url(self, url: str) -> None:
        """
        Add URL to browse list if from trusted domain
        """
        if any(domain in url for domain in self.trusted_domains):
            self.urls.append(url)
        else:
            print(f"Skipping untrusted domain: {url}")

    def _is_business_disruption_content(self, content: str) -> bool:
        """
        Check if content is relevant to business disruption analysis
        """
        disruption_keywords = [
            "artificial intelligence", "AI", "machine learning", "automation",
            "digital transformation", "disruption", "innovation", "technology impact",
            "competitive advantage", "market share", "business model", "efficiency"
        ]
        
        content_lower = content.lower()
        return any(keyword.lower() in content_lower for keyword in disruption_keywords)

    def search(self, max_results: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Browse pages and extract relevant content for disruption analysis
        
        :param max_results: Maximum number of results to return
        :return: List of page content with url and raw_content
        """
        try:
            results = []
            
            for url in self.urls[:max_results]:
                # In a real implementation, this would call the browse_page tool
                # For now, return a structured format that matches expected output
                print(f"BrowsePageRetriever: Browsing {url}")
                
                # Placeholder for actual browse_page tool integration
                result = {
                    "url": url,
                    "raw_content": f"Content from {url} related to: {self.query}"
                }
                
                # In actual implementation, would filter based on disruption relevance
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"BrowsePageRetriever failed: {e}")
            return None

    def browse_url(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Browse a specific URL and extract content
        
        :param url: URL to browse
        :return: Page content with url and raw_content
        """
        try:
            if not any(domain in url for domain in self.trusted_domains):
                print(f"Skipping untrusted domain: {url}")
                return None
                
            # In actual implementation, would use browse_page tool
            print(f"BrowsePageRetriever: Browsing single URL {url}")
            
            return {
                "url": url,
                "raw_content": f"Detailed content from {url}"
            }
            
        except Exception as e:
            print(f"Failed to browse {url}: {e}")
            return None 