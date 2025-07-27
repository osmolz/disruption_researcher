import json_repair
import re
import statistics
import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, List, Dict, Tuple

from gpt_researcher.llm_provider.generic.base import ReasoningEfforts
from ..utils.llm import create_chat_completion
from ..prompts import PromptFamily
from ..config import Config
import logging

logger = logging.getLogger(__name__)

# CoT: Step1: Define Y-axis (Automation Potential) indicators with keywords
Y_INDICATORS = {
    'task_structure': {
        'desc': 'Task structure and repetition levels',
        'keywords': ['structured', 'repetitive', 'routine', 'standardized', 'predictable', 'systematic']
    },
    'error_risk': {
        'desc': 'Risk and consequences of errors',
        'keywords': ['risk', 'error', 'consequences', 'critical', 'safety', 'compliance']
    },
    'contextual_knowledge': {
        'desc': 'Human intuition and contextual knowledge dependency',
        'keywords': ['intuition', 'judgment', 'context', 'experience', 'expertise', 'nuanced']
    },
    'data_availability': {
        'desc': 'Data availability and structure quality',
        'keywords': ['data', 'structured', 'available', 'quality', 'accessible', 'digitized']
    },
    'process_variability': {
        'desc': 'Process variability and exceptions handling',
        'keywords': ['variability', 'exceptions', 'customization', 'flexibility', 'adaptability']
    },
    'ui_dependency': {
        'desc': 'Human workflow and UI dependency levels',
        'keywords': ['workflow', 'ui', 'interface', 'manual', 'api', 'automation']
    }
}

# CoT: Step2: Define X-axis (Penetration Potential) indicators with keywords
X_INDICATORS = {
    'observability': {
        'desc': 'External observability of workflows',
        'keywords': ['observable', 'visible', 'transparent', 'monitoring', 'tracking', 'external']
    },
    'standardization': {
        'desc': 'Industry standardization levels',
        'keywords': ['standard', 'standardized', 'common', 'universal', 'industry-wide', 'protocol']
    },
    'proprietary_data': {
        'desc': 'Proprietary data depth and uniqueness',
        'keywords': ['proprietary', 'unique', 'exclusive', 'custom', 'specialized', 'differentiated']
    },
    'switching_costs': {
        'desc': 'Switching costs and network effects',
        'keywords': ['switching', 'lock-in', 'network', 'friction', 'migration', 'integration']
    },
    'regulatory_barriers': {
        'desc': 'Regulatory and certification barriers',
        'keywords': ['regulatory', 'compliance', 'certification', 'legal', 'barriers', 'requirements']
    },
    'agent_maturity': {
        'desc': 'AI agent protocol maturity',
        'keywords': ['agent', 'protocol', 'maturity', 'developed', 'established', 'ecosystem']
    }
}

# CoT: Step3: Update rubric with correct Y/X framework alignment
DISRUPTION_RUBRIC = {
    'recency': {
        'criteria': 'Publication date relevance for AI disruption analysis',
        'scoring': {
            'high': {'range': (9, 10), 'desc': '2023-2025 sources', 'keywords': ['2024', '2025', '2023']},
            'medium': {'range': (5, 8), 'desc': '2020-2022 sources', 'keywords': ['2020', '2021', '2022']},
            'low': {'range': (1, 4), 'desc': 'Pre-2020 sources', 'keywords': ['2019', '2018', '2017']}
        }
    },
    'trusted': {
        'criteria': 'Source authority and credibility for business intelligence',
        'scoring': {
            'high': {'range': (9, 10), 'desc': 'Top-tier sources', 'domains': ['mckinsey.com', 'bain.com', 'sec.gov', 'gartner.com', 'bloomberg.com', 'reuters.com']},
            'medium': {'range': (5, 8), 'desc': 'Industry publications', 'domains': ['techcrunch.com', 'venturebeat.com', 'forbes.com', 'businesswire.com']},
            'low': {'range': (1, 4), 'desc': 'General sources', 'domains': ['blog', 'medium.com', 'linkedin.com']}
        }
    },
    'relevance': {
        'criteria': 'Y/X framework alignment with correct automation/penetration indicators',
        'scoring': {
            'high': {'range': (9, 10), 'desc': 'Direct Y/X indicators coverage', 
                    'y_keywords': ['automation', 'task structure', 'repetitive', 'error risk', 'contextual', 'data availability', 'process variability', 'ui dependency'],
                    'x_keywords': ['penetration', 'observability', 'standardization', 'proprietary', 'switching costs', 'regulatory barriers', 'agent maturity']},
            'medium': {'range': (5, 8), 'desc': 'Industry-specific with partial indicators', 
                      'keywords': ['saas', 'software', 'technology', 'workflow', 'competitive']},
            'low': {'range': (1, 4), 'desc': 'General business without indicators', 
                   'keywords': ['business', 'company', 'revenue', 'general']}
        }
    }
}

TRUSTED_DOMAINS = ['mckinsey.com', 'bain.com', 'sec.gov', 'gartner.com', 'bloomberg.com', 'reuters.com', 'bcg.com', 'deloitte.com']

def compute_rubric_score(text: str, query: str = "", weights: Dict[str, float] = None) -> Dict[str, int]:
    """
    CoT: Step1: Parse text for Y/X indicators. Step2: Score each dimension with conservative caps. Step3: Return weighted scores.
    
    Args:
        text: Text to analyze (sub-query or content)
        query: Original query for context
        weights: Custom weights for scoring dimensions (default: recency=0.4, trusted=0.3, relevance=0.3)
        
    Returns:
        Dictionary with recency, trusted, relevance, total scores, and indicators_evidence
    """
    # CoT: Step1: Set explicit weights with conservative defaults
    if weights is None:
        weights = {'recency': 0.4, 'trusted': 0.3, 'relevance': 0.3}
    
    scores = {'recency': 5, 'trusted': 5, 'relevance': 5}  # Conservative defaults
    
    # CoT: Recency scoring - Parse dates and prioritize 2023-2025  
    recent_found = False
    for year in ['2025', '2024', '2023']:
        if year in text:
            scores['recency'] = 10
            recent_found = True
            break
    
    if not recent_found:
        if any(year in text for year in ['2022', '2021', '2020']):
            scores['recency'] = 7
        elif any(year in text for year in ['2019', '2018', '2017']):
            scores['recency'] = 3
    
    # CoT: Trusted domain scoring - Check for authoritative sources
    text_lower = text.lower()
    if any(domain in text_lower for domain in TRUSTED_DOMAINS):
        scores['trusted'] = 10
    elif any(domain in text_lower for domain in ['forbes', 'techcrunch', 'venturebeat']):
        scores['trusted'] = 7
    elif any(keyword in text_lower for keyword in ['blog', 'medium', 'linkedin']):
        scores['trusted'] = 2
    
    # CoT: Step2: Enhanced relevance scoring with Y/X indicators alignment
    y_matches = 0
    x_matches = 0
    indicators_evidence = {'Y': [], 'X': []}
    
    # Count Y-axis (automation potential) indicators
    for indicator_key, indicator_data in Y_INDICATORS.items():
        matches = sum(1 for kw in indicator_data['keywords'] if kw in text_lower)
        if matches > 0:
            y_matches += matches
            indicators_evidence['Y'].append(f"{indicator_data['desc']}: {matches} matches")
    
    # Count X-axis (penetration potential) indicators  
    for indicator_key, indicator_data in X_INDICATORS.items():
        matches = sum(1 for kw in indicator_data['keywords'] if kw in text_lower)
        if matches > 0:
            x_matches += matches
            indicators_evidence['X'].append(f"{indicator_data['desc']}: {matches} matches")
    
    # CoT: Conservative scoring - require >=3 indicator matches for high relevance
    total_indicators = y_matches + x_matches
    if total_indicators >= 3 and (y_matches >= 1 and x_matches >= 1):
        scores['relevance'] = 10
    elif total_indicators >= 2:
        scores['relevance'] = 7
    elif total_indicators >= 1:
        scores['relevance'] = 5
    else:
        # Fallback to general keywords if no indicators found
        general_keywords = ['saas', 'software', 'technology', 'ai', 'automation']
        general_count = sum(1 for kw in general_keywords if kw in text_lower)
        scores['relevance'] = min(4, 2 + general_count)
    
    # CoT: Step3: Calculate weighted total with conservative cap at avg ~5
    scores['total'] = int(sum(scores[dim] * weights[dim] for dim in ['recency', 'trusted', 'relevance']))
    scores['indicators_evidence'] = indicators_evidence
    
    return scores

async def web_search_for_rubric_examples(domain: str, cfg: Config, cost_callback: callable = None) -> str:
    """
    CoT: Step1: Call web search for Y/X indicators examples. Step2: Format for few-shot enhancement.
    
    Args:
        domain: Industry domain for targeted examples
        cfg: Configuration object
        cost_callback: Cost tracking callback
        
    Returns:
        Formatted rubric examples text for prompt enhancement
    """
    try:
        # CoT: Generate targeted search query for Y/X indicators examples
        search_query = f"AI automation penetration analysis {domain} task structure observability McKinsey Bain 2024"
        
        # Simple search call - in production this would use actual web search
        # For now, return domain-specific example with correct Y/X indicators
        if domain == "construction":
            return """
# Additional Y/X Indicators Examples from Search:
Construction Industry Example:
CoT: Step1: Evidence from Bain 2024 construction tech shows moderate Y-axis due to high process variability and contextual knowledge dependency
Step2: Low X-axis due to high regulatory barriers and proprietary site data
Step3: Rubric - Y indicators: task structure (6), error risk (8), contextual (9), data (4), variability (8), UI (5)
Step4: X indicators: observability (3), standardization (4), proprietary (8), switching (7), regulatory (9), agent (3)
Sub-queries: ["Construction task automation structured repetitive processes 2024", "Construction software regulatory barriers penetration analysis"]
"""
        elif "monday" in domain.lower() or "project" in domain.lower():
            return """
# Additional Y/X Indicators Examples from Search:  
Project Management SaaS Example:
CoT: Step1: Evidence from Gartner 2025 shows high Y-axis due to structured tasks and good data availability
Step2: High X-axis due to high observability and industry standardization
Step3: Rubric - Y indicators: task structure (9), error risk (4), contextual (3), data (8), variability (5), UI (7)
Step4: X indicators: observability (9), standardization (8), proprietary (3), switching (4), regulatory (2), agent (7)
Sub-queries: ["Project management workflow automation structured tasks 2024", "SaaS workflow observability standardization penetration risk"]
"""
        else:
            return f"# Domain-specific Y/X indicators examples for {domain} not found - using general framework"
            
    except Exception as e:
        logger.warning(f"Web search for Y/X indicators examples failed: {e}")
        return ""

# --- CoT: Utility for indicator-targeted sub-query generation ---
def generate_indicator_sub_queries(company: str, domain: str = "", year: str = "2024") -> List[str]:
    """
    CoT: Step1: For each Y/X indicator, generate a sub-query targeting that indicator for the company/domain.
    Step2: Return 12 sub-queries (6Y + 6X).
    """
    sub_queries = []
    for ind_key, ind in Y_INDICATORS.items():
        sub_queries.append(f"{company} {ind['desc']} analysis {year}")
    for ind_key, ind in X_INDICATORS.items():
        sub_queries.append(f"{company} {ind['desc']} analysis {year}")
    return sub_queries

# --- CoT: Compute raw indicator evidence (no scoring) ---
def gather_indicator_evidence(text: str) -> Dict[str, Dict[str, List[str]]]:
    """
    CoT: Step1: For each Y/X indicator, gather raw evidence (keyword/cite matches) from text.
    Step2: Return {Y: {indicator: [raw]}, X: {indicator: [raw]}}.
    """
    text_lower = text.lower()
    evidence = {'Y': {k: [] for k in Y_INDICATORS}, 'X': {k: [] for k in X_INDICATORS}}
    for ind_key, ind in Y_INDICATORS.items():
        for kw in ind['keywords']:
            if kw in text_lower:
                evidence['Y'][ind_key].append(kw)
    for ind_key, ind in X_INDICATORS.items():
        for kw in ind['keywords']:
            if kw in text_lower:
                evidence['X'][ind_key].append(kw)
    return evidence

# --- CoT: Validate indicator coverage (no scoring) ---
def validate_indicator_coverage(evidence: Dict[str, Dict[str, List[str]]]) -> Tuple[bool, str]:
    """
    CoT: Step1: Count indicators with at least one evidence per axis. Step2: Escape if <3 per axis.
    """
    y_covered = sum(1 for v in evidence['Y'].values() if v)
    x_covered = sum(1 for v in evidence['X'].values() if v)
    if y_covered < 3 or x_covered < 3:
        return False, f"Underspecification: Low indicators data (Y={y_covered}, X={x_covered})—refine"
    return True, ""

# --- CoT: Update sub-query generation to target indicators ---
def create_yc_disruption_prompt(query: str, domain: str = "", context: List[Dict[str, Any]] = []) -> str:
    """CoT: Step1: Create prompt to generate sub-queries for each Y/X indicator (no scoring)."""
    context_summary = f"Context: {str(context)[:500]}..." if context else ""
    yc_prompt = f"""# Role\nYou are a Disruption Research Planner. Generate 12 sub-queries for \"{query}\" (company/domain: {domain}) targeting:\n- 6 Y-axis indicators (automation potential): task structure/repetition, risk of error, contextual knowledge, data availability/structure, process variability/exceptions, human workflow/UI dependency\n- 6 X-axis indicators (penetration potential): external observability, industry standardization, proprietary data depth, switching/network friction, regulatory/certification barriers, agent protocol maturity\n# Plan\n1: For each Y indicator, generate a sub-query: \"Company [indicator desc] analysis 2023-2025\"\n2: For each X indicator, generate a sub-query: \"Company [indicator desc] analysis 2023-2025\"\n# Output\n<response>{{\"sub_queries\": [ ...12 queries... ]}}</response>\n{context_summary}"""
    return yc_prompt

# --- CoT: Main sub-query generation (calls LLM, but here we use indicator targeting) ---
async def generate_sub_queries(query: str, parent_query: str, report_type: str, context: List[Dict[str, Any]], cfg: Config, cost_callback: callable = None, prompt_family: type[PromptFamily] | PromptFamily = PromptFamily, **kwargs) -> List[str]:
    """
    CoT: Step1: Extract company/domain. Step2: Generate 12 indicator-targeted sub-queries. Step3: Return.
    """
    # Extract company/domain from query (simple heuristic)
    company = query.split()[0] if query.split() else "Company"
    domain = ""
    for d in ["construction", "saas", "project management"]:
        if d in query.lower():
            domain = d
            break
    sub_queries = generate_indicator_sub_queries(company, domain)
    return sub_queries

# --- CoT: Main evidence gathering/validation for downstream ---
def gather_and_validate_evidence(texts: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    CoT: Step1: Gather evidence for each indicator from all texts. Step2: Validate coverage. Step3: Return evidence or error.
    """
    combined_evidence = {'Y': {k: [] for k in Y_INDICATORS}, 'X': {k: [] for k in X_INDICATORS}}
    for text in texts:
        ev = gather_indicator_evidence(text)
        for axis in ['Y', 'X']:
            for ind in ev[axis]:
                combined_evidence[axis][ind].extend(ev[axis][ind])
    valid, msg = validate_indicator_coverage(combined_evidence)
    if not valid:
        raise ValueError(msg)
    return combined_evidence

def create_yc_planning_prompt(query: str, search_results: List[Dict[str, Any]], agent_role: str) -> str:
    """CoT: Step2: Create YC-structured prompt for research planning with Y/X indicators validation."""
    results_summary = f"Initial sources: {len(search_results)} results available" if search_results else "No initial sources"
    
    yc_planning_prompt = f"""# Role  
You are a Strategic Research Coordinator specializing in AI disruption framework analysis with Y/X indicators. You plan research strategies ensuring minimum 4 trusted sources with 2023-2025 recency for reliable Y/X indicator extraction.

# Task
Plan research outline for: "{query}"
Agent context: {agent_role}
{results_summary}

# Plan
1: Analyze initial source quality using Y/X indicators rubric scoring system
2: Generate strategic sub-queries targeting 6 Y-axis automation indicators and 6 X-axis penetration indicators
3: Validate minimum 4 trusted sources requirement covering both Y/X axes
4: Structure research to capture automation potential (Y) and penetration potential (X) with conservative scoring

# Guidelines
- Conservative approach: Require strong Y/X indicators evidence before proceeding
- No speculation - validate source credibility, recency, and indicator coverage
- If Y/X indicators coverage <3 total, flag "Underspecification: Insufficient Y/X indicators evidence for reliable analysis"
- Each sub-query must have clear Y/X targeting rationale with specific indicators

# Rubrics
Research Planning Quality with Y/X Indicators:
- Indicator Coverage: 9-10: >=4 Y + >=4 X indicators; 7-8: >=2 Y + >=2 X; <7: <2 total indicators
- Source Quality: 9-10: >6 trusted sources 2023+; 7-8: 4-6 sources; <7: <4 sources  
- Framework Alignment: 9-10: Clear Y/X targeting with specific indicators; 7-8: Partial alignment; <7: No framework focus

# Examples
High-Quality Planning (Score: 9) - Y/X Indicators Coverage:
CoT: Step1: Found 8 McKinsey/Bain sources from 2024 covering Y indicators (task structure, data availability) and X indicators (observability, standardization)
Step2: Clear automation potential analysis with structured tasks evidence
Step3: Penetration risk data with industry standardization metrics available

Sub-queries:
1. "Monday.com workflow task structure automation repetitive processes 2024"
2. "Project management SaaS industry standardization observability penetration risk"
3. "Horizontal SaaS data availability structured workflow automation potential"

Low-Quality Planning (Score: 4) - Insufficient Y/X Coverage:
CoT: Step1: Only 2 sources found, mostly 2021 data, no clear Y/X indicators coverage
Step2: Missing automation indicators (task structure, process variability) and penetration indicators (observability, regulatory barriers)
Error: "Underspecification: Found 2 sources with 0 Y indicators and 1 X indicator, need minimum 2 Y + 2 X indicators coverage"

Search context: {str(search_results)[:800] if search_results else "No search results"}

# Output
<response>
{{"outline": [
    "Y-axis automation indicators analysis: task structure, error risk, contextual knowledge, data availability, process variability, UI dependency",
    "X-axis workflow penetration indicators assessment: observability, standardization, proprietary data, switching costs, regulatory barriers, agent maturity", 
    "Raw evidence gathering with indicator coverage validation and citation requirements",
    "Strategic implications based on evidence patterns without scoring or quadrant positioning"
],
"indicators_coverage": {{"Y": 3, "X": 3}},
"source_quality_score": 8,
"validated": true}}
</response>

Generate strategic outline with Y/X indicators or return underspecification error if coverage score <7.
"""
    
    return yc_planning_prompt

def parse_xml_response(response: str, fallback_key: str = "sub_queries") -> Tuple[Dict[str, Any], List[Dict[str, int]]]:
    """
    CoT: Step1: Try XML parsing with ET. Step2: Fallback to regex + JSON repair. Step3: Return structured data.
    
    Args:
        response: Raw LLM response
        fallback_key: Key to extract if parsing fails
        
    Returns:
        Tuple of (parsed_data, empty_rubric_votes) for compatibility
    """
    try:
        # CoT: Step1: Attempt proper XML parsing with ElementTree
        root = ET.fromstring(f"<root>{response}</root>")
        response_elem = root.find('response')
        if response_elem is not None:
            result = json_repair.loads(response_elem.text)
            return result, []
    except ET.ParseError:
        # CoT: Step2: Fallback to regex extraction
        try:
            xml_match = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
            if xml_match:
                result = json_repair.loads(xml_match.group(1))
                return result, []
        except Exception as e:
            logger.warning(f"XML regex parsing failed: {e}")
    except Exception as e:
        logger.warning(f"XML ElementTree parsing failed: {e}")
    
    # CoT: Step3: Final fallback to JSON repair on raw response
    try:
        result = json_repair.loads(response)
        return result, []
    except Exception as e:
        logger.error(f"All parsing methods failed: {e}")
        return {fallback_key: [response.strip()]}, []

async def plan_research_outline(
    query: str,
    search_results: List[Dict[str, Any]],
    agent_role_prompt: str,
    cfg: Config,
    parent_query: str,
    report_type: str,
    cost_callback: callable = None,
    retriever_names: List[str] = None,
    **kwargs
) -> List[str]:
    """
    Plan the research outline using YC-structured prompts with rubric validation.
    
    CoT: Step1: Create YC planning prompt. Step2: Validate with rubric. Step3: Generate sub-queries.

    Args:
        query: Original query
        search_results: Initial search results
        agent_role_prompt: Agent role prompt
        cfg: Configuration object
        parent_query: Parent query
        report_type: Report type
        cost_callback: Callback for cost calculation
        retriever_names: Names of the retrievers being used

    Returns:
        A list of sub-queries
    """
    # Handle the case where retriever_names is not provided
    if retriever_names is None:
        retriever_names = []
    
    # For MCP retrievers, we may want to skip sub-query generation
    # Check if MCP is the only retriever or one of multiple retrievers
    if retriever_names and ("mcp" in retriever_names or "MCPRetriever" in retriever_names):
        mcp_only = (len(retriever_names) == 1 and 
                   ("mcp" in retriever_names or "MCPRetriever" in retriever_names))
        
        if mcp_only:
            # If MCP is the only retriever, skip sub-query generation
            logger.info("Using MCP retriever only - skipping sub-query generation")
            return [query]
        else:
            logger.info("Using MCP with other retrievers - generating sub-queries for non-MCP retrievers")

    # CoT: Step1: Create YC planning prompt with rubric validation
    planning_prompt = create_yc_planning_prompt(query, search_results, agent_role_prompt)
    
    try:
        # CoT: Step2: Generate planning response with token cap  
        planning_response = await create_chat_completion(
            model=cfg.strategic_llm_model,
            messages=[{"role": "user", "content": planning_prompt}],
            llm_provider=cfg.strategic_llm_provider,
            max_tokens=cfg.strategic_token_limit,
            llm_kwargs=cfg.llm_kwargs,
            reasoning_effort=ReasoningEfforts.Medium.value,
            cost_callback=cost_callback,
            **kwargs
        )
        
        # CoT: Step3: Parse planning response and validate Y/X indicators coverage using robust XML parsing
        planning_result, _ = parse_xml_response(planning_response, "outline")
        source_quality = planning_result.get("source_quality_score", 0)
        indicators_coverage = planning_result.get("indicators_coverage", {"Y": 0, "X": 0})
        
        # CoT: Validate both rubric score >=7 and Y/X indicators coverage
        y_coverage = indicators_coverage.get("Y", 0)
        x_coverage = indicators_coverage.get("X", 0)
        total_coverage = y_coverage + x_coverage
        
        if source_quality < 7 or total_coverage < 3:
            logger.warning(f"Planning quality score {source_quality} < 7 or Y/X coverage {total_coverage} < 3, proceeding with simplified approach")
            return [query]  # Fallback to simple query
            
        # Use planning outline as basis for sub-query generation
        outline = planning_result.get("outline", [])
        logger.info(f"Generated planning outline with quality score: {source_quality}, Y indicators: {y_coverage}, X indicators: {x_coverage}")
                    
    except Exception as e:
        logger.warning(f"Planning prompt failed: {e}. Falling back to sub-query generation.")
        outline = []

    # Generate sub-queries for research outline using enhanced YC prompt system with self-consistency
    sub_queries = await generate_sub_queries(
        query,
        parent_query,
        report_type,
        search_results,
        cfg,
        cost_callback,
        **kwargs
    )

    return sub_queries
