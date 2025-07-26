"""
GPT-Researcher Prompts - Optimized for AI Disruption Analysis Framework

This module contains YC-style prompts optimized for AI disruption analysis with focus on:
- Y-axis (Automation Potential): Task complexity, AI tool availability, implementation feasibility
- X-axis (Market Penetration): Adoption rates, competitive dynamics, customer acceptance

Key Framework Components:
1. Planning Phase: Strategic sub-query generation with Y/X targeting
2. Execution Phase: Evidence extraction with 2-3 citations from 2023-2025 sources  
3. Publishing Phase: Quadrant positioning with self-consistency validation

Quality Controls:
- Conservative scoring: Cap >5 only with strong evidence and sufficient citations
- Source validation: Minimum 4 sources, prioritize recent/trusted/relevant
- Self-consistency: Generate 3 variants, vote on median scores
- Error handling: Return underspecification errors for insufficient evidence

Output Formats:
- XML for evidence: <evidence><indicator>...</indicator><score>X</score><cites>[...]</cites></evidence>
- JSON for final analysis with y_scores, x_scores, quadrant, rationale, tailwinds, headwinds
"""

import warnings
from datetime import date, datetime, timezone

from langchain.docstore.document import Document

from .config import Config
from .utils.enum import ReportSource, ReportType, Tone
from .utils.enum import PromptFamily as PromptFamilyEnum
from typing import Callable, List, Dict, Any


## Prompt Families #############################################################

class PromptFamily:
    """General purpose class for prompt formatting.

    This may be overwritten with a derived class that is model specific. The
    methods are broken down into two groups:

    1. Prompt Generators: These follow a standard format and are correlated with
        the ReportType enum. They should be accessed via
        get_prompt_by_report_type

    2. Prompt Methods: These are situation-specific methods that do not have a
        standard signature and are accessed directly in the agent code.

    All derived classes must retain the same set of method names, but may
    override individual methods.
    """

    def __init__(self, config: Config):
        """Initialize with a config instance. This may be used by derived
        classes to select the correct prompting based on configured models and/
        or providers
        """
        self.cfg = config

    # DISRUPTION ANALYSIS FRAMEWORK PROMPTS ####################################
    
    @staticmethod
    def generate_disruption_planning_prompt(
        query: str,
        domain: str = "",
        max_subqueries: int = 5,
        context: List[Dict[str, Any]] = [],
    ) -> str:
        """
        YC-style planning prompt for AI disruption analysis sub-query generation.
        Focuses on Y-axis (automation potential) and X-axis (market penetration).
        """
        
        context_info = f"Context: {context}" if context else ""
        
        return f"""# Role: Strategic Planning Specialist for AI Disruption Analysis
You are an expert strategist specializing in identifying AI disruption patterns across industries, with deep knowledge of automation potential and market penetration dynamics.

### Task: Generate Strategic Sub-Queries for Disruption Analysis
Generate {max_subqueries} focused sub-queries to analyze: "{query}" in domain: "{domain}"
Target Y-axis (automation potential) and X-axis (market penetration) evidence from 2023-2025 sources.

### Plan:
Step 1: Industry automation assessment - Current task complexity and AI readiness
Step 2: Technology framework analysis - Available tools and implementation barriers  
Step 3: Market penetration evaluation - Adoption rates and competitive dynamics
Step 4: Evidence validation - Ensure 2-3 recent citations per sub-query

### Guidelines:
- **Rubric**: Score relevance 1-10 (1-3: Irrelevant/outdated, 7-10: Advances Y/X analysis)
- **Evidence Requirement**: Each sub-query must target sources with 2-3 citations from trusted sources (2023-2025)
- **Self-Consistency**: Generate 3 variants, vote on median relevance scores ≥7
- **Quality Control**: If <4 relevant sources available, flag for underspecification

### Examples:
**High-Quality Sub-Query (Score: 9)**
- "Construction project management automation tools adoption rates 2024-2025"
- Targets: Y-axis (automation of scheduling/resource allocation), X-axis (market penetration in construction)

**Low-Quality Sub-Query (Score: 3)** 
- "General construction industry trends"
- Issue: Too broad, no Y/X focus, lacks specificity for evidence extraction

{context_info}

### Output Format:
<plan>
<step>
<sub_query>Industry automation trends in {domain} focusing on task complexity reduction 2024-2025</sub_query>
<rubric_score>8</rubric_score>
<target_axis>Y-axis automation potential</target_axis>
</step>
<step>
<sub_query>Market penetration rates for AI tools in {domain} competitive landscape</sub_query>
<rubric_score>7</rubric_score>
<target_axis>X-axis market penetration</target_axis>
</step>
</plan>

Generate exactly {max_subqueries} sub-queries with scores ≥7 or return underspecification error.
Current date context: {datetime.now(timezone.utc).strftime('%B %d, %Y')}
"""

    @staticmethod
    def generate_disruption_execution_prompt(
        query: str,
        sources_data: str,
        max_evidence_items: int = 8,
    ) -> str:
        """
        YC-style execution prompt for evidence extraction with Y/X disruption framework.
        """
        
        return f"""# Role: Evidence Extraction Specialist for AI Disruption Analysis
You are a senior analyst specializing in extracting automation and market penetration indicators from research sources with rigorous citation standards.

### Task: Extract Y/X Disruption Evidence
Analyze sources for query: "{query}"
Extract automation potential (Y-axis) and market penetration (X-axis) indicators with 2-3 citations each.

### Plan:
Step 1: Identify Y-axis automation evidence (task complexity, AI tool availability, implementation feasibility)
Step 2: Extract X-axis penetration data (adoption rates, market share, competitive positioning)  
Step 3: Validate evidence quality with recent citations (2023-2025)
Step 4: Score each indicator 1-10 based on strength and relevance

### Guidelines:
- **Evidence Scoring**: 1-3 (Low/weak evidence), 4-6 (Moderate), 7-10 (High/strong evidence)
- **Citation Requirement**: Minimum 2-3 citations from trusted sources (2023-2025)
- **Quality Cap**: Score >5 ONLY with strong evidence and sufficient citations
- **Self-Consistency**: Generate 3 evidence variants, vote on median scores
- **Token Limit**: Summaries <500 tokens total

### Scoring Rubric:
- **Recent**: Published 2023-2025 (+2 points)
- **Trusted**: Authoritative source (McKinsey, Bain, industry leaders) (+2 points)  
- **Relevant**: Directly addresses Y/X framework (+3 points)
- **Citations**: 2-3 quality citations (+3 points)

### Examples:
**High-Quality Evidence (Score: 8)**
```
<evidence>
<indicator>Task Automation Potential</indicator>
<score>8</score>
<axis>Y-axis</axis>
<cites>[McKinsey 2024: Construction Automation Report, Bain 2025: AI in Project Management]</cites>
<rationale>Strong evidence for 40% task automation in project scheduling with AI tools showing proven ROI</rationale>
</evidence>
```

**Low-Quality Evidence (Score: 3)**  
```
<evidence>
<indicator>General Industry Growth</indicator>
<score>3</score>
<axis>Neither</axis>
<cites>[Generic blog 2022]</cites>
<rationale>Outdated, non-specific, lacks Y/X relevance</rationale>
</evidence>
```

### Source Data:
{sources_data}

### Output Format:
Extract up to {max_evidence_items} pieces of evidence. If <4 quality sources or average score <4:

<if_block condition="insufficient_evidence">
{{"error": "Underspecification: Insufficient quality sources for reliable Y/X analysis. Found X sources, need minimum 4 with recent citations."}}
</if_block>

Otherwise, provide evidence in XML format as shown in examples above.
"""

    @staticmethod
    def generate_disruption_publishing_prompt(
        query: str,
        evidence_data: str,
        company_name: str = "",
    ) -> str:
        """
        YC-style publishing prompt for final disruption analysis aggregation into JSON output.
        """
        
        return f"""# Role: Strategic Publisher for AI Disruption Analysis
You are a senior strategist responsible for synthesizing evidence into authoritative disruption analysis with quadrant positioning and strategic recommendations.

### Task: Aggregate Evidence into Disruption Analysis
Synthesize evidence for: "{query}" {f"(Company: {company_name})" if company_name else ""}
Generate final Y/X scores, quadrant positioning, and strategic implications.

### Plan:
Step 1: Aggregate Y-axis automation scores (median of evidence scores)
Step 2: Aggregate X-axis penetration scores (median of evidence scores)  
Step 3: Position in disruption quadrant (Sustaining/Low-End/New-Market/Big Bang)
Step 4: Identify strategic tailwinds and headwinds
Step 5: Self-consistency check via 3 analysis variants

### Scoring Framework:
- **Y-Axis (Automation Potential)**: 1-10 scale
  - 1-3: Manual/complex tasks, limited AI applicability
  - 4-6: Moderate automation potential, some AI tools available
  - 7-10: High automation potential, proven AI solutions
  
- **X-Axis (Market Penetration)**: 1-10 scale  
  - 1-3: Early/niche adoption, high barriers
  - 4-6: Growing adoption, moderate penetration
  - 7-10: Mainstream adoption, established market

### Quadrant Mapping:
- **Sustaining** (High Y, High X): Incremental improvements to existing solutions
- **Low-End Disruptive** (Low Y, High X): Simpler, more accessible alternatives
- **New-Market Disruptive** (High Y, Low X): Novel solutions for underserved segments  
- **Big Bang Disruptive** (High Y, High X): Simultaneous better and cheaper solutions

### Guidelines:
- **Self-Consistency**: Generate 3 analysis variants, vote on median final scores
- **Evidence Weighting**: Prioritize recent, trusted sources with strong citations
- **Conservative Scoring**: Cap scores >7 only with overwhelming evidence
- **Token Limit**: Total analysis <2000 tokens

### Evidence Data:
{evidence_data}

### Output Format:
<analysis>
<y_scores>[automation_score_1, automation_score_2, ...]</y_scores>
<x_scores>[penetration_score_1, penetration_score_2, ...]</x_scores>
<median_y_score>7</median_y_score>
<median_x_score>4</median_x_score>
<quadrant>New-Market Disruptive</quadrant>
<confidence_level>High</confidence_level>
<rationale>Evidence-based reasoning for quadrant positioning with specific citations and score justification</rationale>
<tailwinds>
- Positive factor 1 with supporting evidence
- Positive factor 2 with citations
</tailwinds>
<headwinds>
- Challenge 1 with evidence
- Challenge 2 with market data
</headwinds>
<strategic_implications>Key recommendations for stakeholders based on quadrant position</strategic_implications>
</analysis>

If evidence is insufficient for reliable analysis, return:
{{"error": "Underspecification: Insufficient evidence quality for reliable disruption analysis. Recommend additional research on [specific gaps]."}}
"""

    @staticmethod
    def generate_self_consistency_check(
        original_analysis: str,
        variant_count: int = 3,
    ) -> str:
        """
        Generate self-consistency check prompt for validation of disruption analysis.
        """
        
        return f"""# Role: Analysis Validation Specialist
You are responsible for ensuring consistency and reliability in disruption analysis through multi-variant validation.

### Task: Self-Consistency Validation
Review the original analysis and generate {variant_count} alternative interpretations using the same evidence.
Vote on median scores for final validation.

### Original Analysis:
{original_analysis}

### Validation Process:
1. Generate {variant_count} alternative score interpretations
2. Identify score variance and reasoning differences  
3. Vote on median scores for Y-axis and X-axis
4. Flag significant discrepancies (>2 point variance)
5. Provide consensus recommendation

### Output Format:
<validation>
<variant_scores>
<variant_1><y_score>X</y_score><x_score>Y</x_score></variant_1>
<variant_2><y_score>X</y_score><x_score>Y</x_score></variant_2>
<variant_3><y_score>X</y_score><x_score>Y</x_score></variant_3>
</variant_scores>
<median_consensus><y_score>X</y_score><x_score>Y</x_score></median_consensus>
<variance_flag>{"High" if variance >2 else "Low"}</variance_flag>
<consensus_confidence>{"High" if variance <1 else "Medium" if variance <2 else "Low"}</consensus_confidence>
<recommendation>Final validated analysis or flag for additional research</recommendation>
</validation>
"""

    # MCP-specific prompts (existing)
    @staticmethod
    def generate_mcp_tool_selection_prompt(query: str, tools_info: List[Dict], max_tools: int = 3) -> str:
        """
        Generate prompt for LLM-based MCP tool selection.
        
        Args:
            query: The research query
            tools_info: List of available tools with their metadata
            max_tools: Maximum number of tools to select
            
        Returns:
            str: The tool selection prompt
        """
        import json
        
        return f"""You are a research assistant helping to select the most relevant tools for a research query.

RESEARCH QUERY: "{query}"

AVAILABLE TOOLS:
{json.dumps(tools_info, indent=2)}

TASK: Analyze the tools and select EXACTLY {max_tools} tools that are most relevant for researching the given query.

SELECTION CRITERIA:
- Choose tools that can provide information, data, or insights related to the query
- Prioritize tools that can search, retrieve, or access relevant content
- Consider tools that complement each other (e.g., different data sources)
- Exclude tools that are clearly unrelated to the research topic

Return a JSON object with this exact format:
{{
  "selected_tools": [
    {{
      "index": 0,
      "name": "tool_name",
      "relevance_score": 9,
      "reason": "Detailed explanation of why this tool is relevant"
    }}
  ],
  "selection_reasoning": "Overall explanation of the selection strategy"
}}

Select exactly {max_tools} tools, ranked by relevance to the research query.
"""

    @staticmethod
    def generate_mcp_research_prompt(query: str, selected_tools: List) -> str:
        """
        Generate prompt for MCP research execution with selected tools.
        
        Args:
            query: The research query
            selected_tools: List of selected MCP tools
            
        Returns:
            str: The research execution prompt
        """
        # Handle cases where selected_tools might be strings or objects with .name attribute
        tool_names = []
        for tool in selected_tools:
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            else:
                tool_names.append(str(tool))
        
        return f"""You are a research assistant with access to specialized tools. Your task is to research the following query and provide comprehensive, accurate information.

RESEARCH QUERY: "{query}"

INSTRUCTIONS:
1. Use the available tools to gather relevant information about the query
2. Call multiple tools if needed to get comprehensive coverage
3. If a tool call fails or returns empty results, try alternative approaches
4. Synthesize information from multiple sources when possible
5. Focus on factual, relevant information that directly addresses the query

AVAILABLE TOOLS: {tool_names}

Please conduct thorough research and provide your findings. Use the tools strategically to gather the most relevant and comprehensive information."""

    @staticmethod
    def generate_search_queries_prompt(
        question: str,
        parent_query: str,
        report_type: str,
        max_iterations: int = 3,
        context: List[Dict[str, Any]] = [],
    ):
        """Generates YC-style search queries prompt optimized for disruption analysis.
        Args:
            question (str): The question to generate the search queries prompt for
            parent_query (str): The main question (only relevant for detailed reports)
            report_type (str): The report type
            max_iterations (int): The maximum number of search queries to generate
            context (str): Context for better understanding of the task with realtime web information

        Returns: str: The search queries prompt for the given question
        """

        if (
            report_type == ReportType.DetailedReport.value
            or report_type == ReportType.SubtopicReport.value
        ):
            task = f"{parent_query} - {question}"
        else:
            task = question

        context_prompt = f"""
### Context Intelligence:
{context}

Use this real-time context to enhance search query specificity, focusing on recent developments (2023-2025) that impact automation potential and market penetration dynamics.
""" if context else ""

        dynamic_example = ", ".join([f'"query {i+1}"' for i in range(max_iterations)])

        return f"""# Role: Strategic Search Query Specialist for Disruption Analysis
You are an expert research strategist specializing in identifying AI disruption patterns through targeted search queries that uncover automation potential and market penetration evidence.

### Task: Generate Disruption-Focused Search Queries
Generate {max_iterations} strategic search queries for: "{task}"
Focus on Y-axis (automation potential) and X-axis (market penetration) evidence from 2023-2025 sources.

### Plan:
Step 1: Industry automation assessment queries - Target task complexity and AI tool adoption
Step 2: Market penetration evaluation queries - Focus on adoption rates and competitive dynamics
Step 3: Evidence validation queries - Ensure access to recent, trusted sources with citations

### Guidelines:
- **Temporal Focus**: Prioritize 2023-2025 sources for current disruption landscape
- **Authority Targeting**: Include terms for trusted sources (McKinsey, Bain, industry reports)
- **Y-X Framework**: Each query should target either automation potential OR market penetration
- **Citation Readiness**: Structure queries to find sources with quantitative data and citations
- **Specificity**: Avoid generic terms, focus on measurable disruption indicators

### Search Query Types:
1. **Automation Evidence**: "[industry] AI automation tools adoption rates 2024", "[task] complexity reduction AI 2025"
2. **Penetration Evidence**: "[industry] AI market penetration 2024", "[technology] competitive landscape report 2025"  
3. **Authority Sources**: "McKinsey [industry] AI disruption 2024", "Bain automation report [domain] 2025"

### Quality Standards:
- Each query must target recent sources (2023-2025)
- Include industry-specific terminology for precision
- Structure for quantitative evidence discovery
- Enable citation-rich source identification

{context_prompt}

### Output Format:
You must respond with a list of strings in the following format: [{dynamic_example}].
The response should contain ONLY the list.

Current date context: {datetime.now(timezone.utc).strftime('%B %d, %Y')}
"""

    @staticmethod
    def generate_report_prompt(
        question: str,
        context,
        report_source: str,
        report_format="apa",
        total_words=1000,
        tone=None,
        language="english",
    ):
        """Generates YC-style report prompt optimized for disruption analysis framework.
        Args: question (str): The disruption analysis question to generate the report for
                context: The research context containing Y/X evidence and citations
        Returns: str: The disruption-focused report prompt
        """

        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, prioritizing 2023-2025 sources and authoritative publications.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report:

eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [url website](url)
"""
        else:
            reference_prompt = f"""
You MUST write all used source document names at the end of the report as references, prioritizing recent and authoritative sources.
"""

        tone_prompt = f"Write the report in a {tone.value} tone with emphasis on evidence-based analysis." if tone else ""

        return f"""# Role: Strategic Analyst for AI Disruption Reporting
You are a senior strategic analyst specializing in AI disruption analysis, responsible for synthesizing research into authoritative reports with Y/X framework positioning and evidence-based recommendations.

### Task: Generate Disruption Analysis Report
Using the research context below, generate a comprehensive disruption analysis report for: "{question}"
Focus on Y-axis (automation potential) and X-axis (market penetration) evidence with quadrant positioning.

### Research Context:
"{context}"

### Plan:
Step 1: Analyze automation potential evidence (Y-axis indicators)
Step 2: Evaluate market penetration evidence (X-axis indicators)  
Step 3: Position findings in disruption quadrant framework
Step 4: Synthesize strategic implications and recommendations
Step 5: Validate with evidence citations and quantitative support

### Report Guidelines:
**Structure Requirements**:
- **Executive Summary**: Y/X positioning with confidence level
- **Automation Analysis** (Y-Axis): Task complexity, AI tool availability, implementation feasibility
- **Market Penetration** (X-Axis): Adoption rates, competitive dynamics, customer acceptance
- **Disruption Quadrant**: Sustaining/Low-End/New-Market/Big Bang positioning with rationale
- **Strategic Implications**: Actionable recommendations based on quadrant position
- **Evidence Base**: Citation-rich support with quantitative data emphasis

**Quality Standards**:
- Minimum {total_words} words with comprehensive evidence integration
- Prioritize 2023-2025 sources and authoritative publications (McKinsey, Bain, industry leaders)
- Include quantitative data, statistics, and concrete metrics throughout
- Provide evidence-based conclusions with specific Y/X score rationale
- Use markdown tables for comparative data and framework positioning

**Citation Requirements**:
- In-text citations in {report_format} format with markdown hyperlinks: ([in-text citation](url))
- Prioritize recent, trusted sources over older or less reliable ones
- Include 2-3 citations minimum per major claim or evidence point
- {reference_prompt}

**Framework Application**:
- **Conservative Scoring**: High scores (>7) only with overwhelming evidence
- **Evidence Weighting**: Recent + Trusted + Relevant + Citations ≥2 for credibility
- **Quadrant Logic**: Clear positioning based on Y/X evidence aggregation
- **Strategic Focus**: Actionable insights for stakeholders based on disruption position

**Analysis Requirements**:
- Determine concrete, evidence-based position - avoid generic conclusions
- Integrate multiple source perspectives when supported by data
- Highlight quantitative evidence and measurable indicators
- Address implementation barriers and market dynamics
- Provide confidence levels for key findings

{tone_prompt}

### Output Language: {language}

### Critical Success Factors:
This analysis will inform strategic decision-making. Ensure evidence-based rigor, recent source prioritization, and actionable strategic insights based on disruption quadrant positioning.

Current analysis date: {date.today()}
"""

    @staticmethod
    def curate_sources(query, sources, max_results=10):
        return f"""# Role: Source Curation Specialist for Disruption Analysis
You are an expert research curator specializing in identifying high-quality sources for AI disruption analysis with focus on automation potential and market penetration evidence.

### Task: Curate Sources for Y/X Disruption Framework
Evaluate and curate sources for: "{query}"
Prioritize sources containing Y-axis (automation potential) and X-axis (market penetration) indicators with quantitative data and recent citations.

### Plan:
Step 1: Assess source relevance to Y/X disruption framework
Step 2: Evaluate source authority and recency (prioritize 2023-2025)
Step 3: Prioritize quantitative data and citation-rich sources
Step 4: Select up to {max_results} highest-quality sources

### Curation Guidelines:
**Primary Criteria** (Required for inclusion):
- **Y/X Relevance**: Contains automation potential OR market penetration indicators
- **Quantitative Value**: Includes statistics, numbers, market data, or concrete metrics
- **Citation Quality**: From authoritative sources (McKinsey, Bain, industry reports, academic)
- **Temporal Relevance**: Recent sources (2023-2025) prioritized, older only if essential

**Secondary Criteria** (Enhancement factors):
- **Source Authority**: Favor McKinsey, Bain, BCG, academic institutions, industry leaders
- **Data Depth**: Comprehensive analysis over surface-level information  
- **Multiple Perspectives**: Include diverse viewpoints if they add Y/X insights
- **Implementation Evidence**: Real-world case studies and adoption examples

### Scoring Framework:
**High Priority (Keep)**: 
- Recent authoritative sources with Y/X relevant quantitative data
- Industry reports with adoption rates, automation metrics, market penetration data
- Case studies with measurable automation/penetration outcomes

**Medium Priority** (Keep if space):
- Older sources (2020-2022) with unique Y/X insights
- Opinion pieces from recognized industry experts
- Sources with qualitative insights supporting quantitative trends

**Low Priority (Exclude)**:
- Generic/non-specific content without Y/X relevance
- Sources without quantitative data or concrete evidence
- Outdated information (pre-2020) without historical significance
- Clearly unreliable or non-authoritative sources

### Content Retention Standards:
- **Preserve Original**: DO NOT rewrite, summarize, or condense source content
- **Clean Only**: Remove formatting issues and obvious errors
- **Evidence Focus**: Retain ALL quantitative data, statistics, and metrics
- **Citation Preservation**: Maintain source attribution and publication details

### Sources to Evaluate:
{sources}

### Output Requirements:
Return response in EXACT sources JSON list format as original sources.
Select up to {max_results} sources prioritizing Y/X framework relevance and quantitative evidence.
Response MUST contain NO markdown formatting or additional text - JSON list only!

If fewer than 4 quality sources meet Y/X criteria, include warning in source metadata.
"""

    @staticmethod
    def generate_resource_report_prompt(
        question, context, report_source: str, report_format="apa", tone=None, total_words=1000, language="english"
    ):
        """Generates the resource report prompt for the given question and research summary.

        Args:
            question (str): The question to generate the resource report prompt for.
            context (str): The research summary to generate the resource report prompt for.

        Returns:
            str: The resource report prompt for the given question and research summary.
        """

        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
            You MUST include all relevant source urls.
            Every url should be hyperlinked: [url website](url)
            """
        else:
            reference_prompt = f"""
            You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each."
        """

        return (
            f'"""{context}"""\n\nBased on the above information, generate a bibliography recommendation report for the following'
            f' question or topic: "{question}". The report should provide a detailed analysis of each recommended resource,'
            " explaining how each source can contribute to finding answers to the research question.\n"
            "Focus on the relevance, reliability, and significance of each source.\n"
            "Ensure that the report is well-structured, informative, in-depth, and follows Markdown syntax.\n"
            "Use markdown tables and other formatting features when appropriate to organize and present information clearly.\n"
            "Include relevant facts, figures, and numbers whenever available.\n"
            f"The report should have a minimum length of {total_words} words.\n"
            f"You MUST write the report in the following language: {language}.\n"
            "You MUST include all relevant source urls."
            "Every url should be hyperlinked: [url website](url)"
            f"{reference_prompt}"
        )

    @staticmethod
    def generate_custom_report_prompt(
        query_prompt, context, report_source: str, report_format="apa", tone=None, total_words=1000, language: str = "english"
    ):
        return f'"{context}"\n\n{query_prompt}'

    @staticmethod
    def generate_outline_report_prompt(
        question, context, report_source: str, report_format="apa", tone=None,  total_words=1000, language: str = "english"
    ):
        """Generates the outline report prompt for the given question and research summary.
        Args: question (str): The question to generate the outline report prompt for
                research_summary (str): The research summary to generate the outline report prompt for
        Returns: str: The outline report prompt for the given question and research summary
        """

        return (
            f'"""{context}""" Using the above information, generate an outline for a research report in Markdown syntax'
            f' for the following question or topic: "{question}". The outline should provide a well-structured framework'
            " for the research report, including the main sections, subsections, and key points to be covered."
            f" The research report should be detailed, informative, in-depth, and a minimum of {total_words} words."
            " Use appropriate Markdown syntax to format the outline and ensure readability."
            " Consider using markdown tables and other formatting features where they would enhance the presentation of information."
        )

    @staticmethod
    def generate_deep_research_prompt(
        question: str,
        context: str,
        report_source: str,
        report_format="apa",
        tone=None,
        total_words=2000,
        language: str = "english"
    ):
        """Generates the deep research report prompt, specialized for handling hierarchical research results.
        Args:
            question (str): The research question
            context (str): The research context containing learnings with citations
            report_source (str): Source of the research (web, etc.)
            report_format (str): Report formatting style
            tone: The tone to use in writing
            total_words (int): Minimum word count
            language (str): Output language
        Returns:
            str: The deep research report prompt
        """
        reference_prompt = ""
        if report_source == ReportSource.Web.value:
            reference_prompt = f"""
You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.
Every url should be hyperlinked: [url website](url)
Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report:

eg: Author, A. A. (Year, Month Date). Title of web page. Website Name. [url website](url)
"""
        else:
            reference_prompt = f"""
You MUST write all used source document names at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each."
"""

        tone_prompt = f"Write the report in a {tone.value} tone." if tone else ""

        return f"""
Using the following hierarchically researched information and citations:

"{context}"

Write a comprehensive research report answering the query: "{question}"

The report should:
1. Synthesize information from multiple levels of research depth
2. Integrate findings from various research branches
3. Present a coherent narrative that builds from foundational to advanced insights
4. Maintain proper citation of sources throughout
5. Be well-structured with clear sections and subsections
6. Have a minimum length of {total_words} words
7. Follow {report_format} format with markdown syntax
8. Use markdown tables, lists and other formatting features when presenting comparative data, statistics, or structured information

Additional requirements:
- Prioritize insights that emerged from deeper levels of research
- Highlight connections between different research branches
- Include relevant statistics, data, and concrete examples
- You MUST determine your own concrete and valid opinion based on the given information. Do NOT defer to general and meaningless conclusions.
- You MUST prioritize the relevance, reliability, and significance of the sources you use. Choose trusted sources over less reliable ones.
- You must also prioritize new articles over older articles if the source can be trusted.
- Use in-text citation references in {report_format} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).
- {tone_prompt}
- Write in {language}

{reference_prompt}

Please write a thorough, well-researched report that synthesizes all the gathered information into a cohesive whole.
Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')}.
"""

    @staticmethod
    def auto_agent_instructions():
        return """
This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a specific server, defined by its type and role, with each server requiring distinct instructions.
Agent
The server is determined by the field of the topic and the specific name of the server that could be utilized to research the topic provided. Agents are categorized by their area of expertise, and each server type is associated with a corresponding emoji.

examples:
task: "should I invest in apple stocks?"
response:
{
    "server": "💰 Finance Agent",
    "agent_role_prompt: "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
}
task: "could reselling sneakers become profitable?"
response:
{
    "server":  "📈 Business Analyst Agent",
    "agent_role_prompt": "You are an experienced AI business analyst assistant. Your main objective is to produce comprehensive, insightful, impartial, and systematically structured business reports based on provided business data, market trends, and strategic analysis."
}
task: "what are the most interesting sites in Tel Aviv?"
response:
{
    "server":  "🌍 Travel Agent",
    "agent_role_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
}
"""

    @staticmethod
    def generate_summary_prompt(query, data):
        """Generates YC-style summary prompt optimized for disruption evidence extraction.
        Args: query (str): The disruption analysis query to extract evidence for
                data (str): The source text to analyze for Y/X indicators
        Returns: str: The disruption-focused summary prompt
        """

        return f"""# Role: Evidence Extraction Specialist for Disruption Analysis
You are a senior analyst specializing in extracting automation potential and market penetration indicators from research sources with rigorous evidence standards.

### Task: Extract Disruption Evidence Summary
Analyze the provided text for query: "{query}"
Extract key evidence related to Y-axis (automation potential) and X-axis (market penetration) with citations.

### Plan:
Step 1: Identify automation evidence (task complexity, AI tools, implementation feasibility)
Step 2: Extract market penetration data (adoption rates, market share, competitive positioning)
Step 3: Preserve quantitative data, statistics, and concrete evidence
Step 4: Maintain source attribution and citation readiness

### Guidelines:
- **Evidence Focus**: Prioritize Y/X framework indicators over general information
- **Quantitative Priority**: Include ALL numbers, stats, percentages, and quantifiable data
- **Citation Preservation**: Maintain source attribution and publication details
- **Temporal Relevance**: Flag publication dates, especially 2023-2025 sources
- **Quality Assessment**: Note source authority (McKinsey, Bain, industry leaders)
- **Token Efficiency**: Summarize <500 tokens while preserving critical evidence

### Source Text:
{data}

### Output Requirements:
If the query cannot be answered using the text, YOU MUST summarize the text focusing on disruption-relevant elements.
Include ALL factual information such as:
- Numbers, statistics, percentages
- Market data and adoption rates  
- Automation capabilities and tool availability
- Implementation timelines and feasibility
- Competitive positioning and market share
- Industry expert quotes and citations

### Evidence Extraction Focus:
**Y-Axis Indicators**: Task automation potential, AI tool sophistication, implementation barriers
**X-Axis Indicators**: Market adoption rates, competitive penetration, customer acceptance

Preserve exact citations and quantitative data for downstream analysis.
"""

    @staticmethod
    def pretty_print_docs(docs: list[Document], top_n: int | None = None) -> str:
        """Compress the list of documents into a context string"""
        return f"\n".join(f"Source: {d.metadata.get('source')}\n"
                          f"Title: {d.metadata.get('title')}\n"
                          f"Content: {d.page_content}\n"
                          for i, d in enumerate(docs)
                          if top_n is None or i < top_n)

    @staticmethod
    def join_local_web_documents(docs_context: str, web_context: str) -> str:
        """Joins local web documents with context scraped from the internet"""
        return f"Context from local documents: {docs_context}\n\nContext from web sources: {web_context}"

    # DISRUPTION ANALYSIS UTILITIES ############################################
    
    @staticmethod
    def validate_evidence_quality(evidence_list: List[Dict], min_sources: int = 4) -> Dict[str, Any]:
        """
        Validate evidence quality for disruption analysis with underspecification detection.
        
        Args:
            evidence_list: List of evidence items with scores and citations
            min_sources: Minimum number of quality sources required
            
        Returns:
            Dict with validation results and error handling
        """
        
        if len(evidence_list) < min_sources:
            return {
                "valid": False,
                "error": f"Underspecification: Found {len(evidence_list)} sources, need minimum {min_sources} for reliable Y/X analysis."
            }
        
        # Check for recent citations (2023-2025)
        recent_sources = [e for e in evidence_list if any("2023" in str(cite) or "2024" in str(cite) or "2025" in str(cite) for cite in e.get("cites", []))]
        
        if len(recent_sources) < min_sources * 0.5:  # At least 50% recent sources
            return {
                "valid": False,
                "error": f"Underspecification: Insufficient recent sources (2023-2025). Found {len(recent_sources)} recent, need minimum {min_sources//2}."
            }
        
        # Check average evidence quality scores
        scores = [e.get("score", 0) for e in evidence_list if e.get("score")]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        if avg_score < 4:
            return {
                "valid": False,
                "error": f"Underspecification: Low evidence quality. Average score {avg_score:.1f}, need minimum 4.0 for reliable analysis."
            }
        
        return {
            "valid": True,
            "quality_score": avg_score,
            "recent_sources": len(recent_sources),
            "total_sources": len(evidence_list)
        }

    @staticmethod
    def generate_disruption_variants_prompt(base_analysis: str, variant_count: int = 3) -> str:
        """
        Generate prompt for creating analysis variants for self-consistency validation.
        """
        
        return f"""# Role: Multi-Perspective Analysis Specialist
You are responsible for generating alternative interpretations of disruption analysis to ensure consistency and reliability through variant validation.

### Task: Generate Analysis Variants
Create {variant_count} alternative interpretations of the base analysis using the same evidence but with different analytical perspectives.

### Base Analysis:
{base_analysis}

### Variant Generation Guidelines:
1. **Maintain Evidence Base**: Use only the same evidence from base analysis
2. **Alternative Perspectives**: Apply different analytical lenses (conservative, optimistic, sector-specific)
3. **Score Variance**: Allow reasonable score variations (±1-2 points) based on interpretation
4. **Consistent Framework**: Maintain Y/X disruption framework throughout all variants

### Variant Types:
**Variant 1 - Conservative Analysis**: 
- Emphasize implementation barriers and adoption challenges
- Weight evidence conservatively, favor lower scores for uncertain indicators
- Focus on market maturity and established player advantages

**Variant 2 - Balanced Analysis**:
- Neutral perspective weighing evidence objectively  
- Standard interpretation of Y/X indicators
- Consider both opportunities and challenges equally

**Variant 3 - Progressive Analysis**:
- Emphasize innovation potential and early adoption signals
- Weight recent evidence and forward-looking indicators heavily
- Focus on disruption potential and emerging trends

### Output Format:
<variants>
<variant_1>
<y_score>X</y_score>
<x_score>Y</x_score>
<perspective>Conservative</perspective>
<rationale>Evidence interpretation emphasizing challenges and barriers</rationale>
</variant_1>
<variant_2>
<y_score>X</y_score>
<x_score>Y</x_score>
<perspective>Balanced</perspective>
<rationale>Neutral evidence interpretation with equal weighting</rationale>
</variant_2>
<variant_3>
<y_score>X</y_score>
<x_score>Y</x_score>
<perspective>Progressive</perspective>
<rationale>Evidence interpretation emphasizing innovation potential</rationale>
</variant_3>
</variants>

Generate exactly {variant_count} variants with scores and rationale for each perspective.
"""

    @staticmethod
    def calculate_consensus_scores(variants: List[Dict]) -> Dict[str, Any]:
        """
        Calculate consensus scores from analysis variants using median voting.
        
        Args:
            variants: List of variant analyses with y_score and x_score
            
        Returns:
            Dict with consensus scores and confidence metrics
        """
        
        y_scores = [v.get("y_score", 0) for v in variants if v.get("y_score")]
        x_scores = [v.get("x_score", 0) for v in variants if v.get("x_score")]
        
        if not y_scores or not x_scores:
            return {"error": "Insufficient variant scores for consensus calculation"}
        
        # Calculate median scores
        y_median = sorted(y_scores)[len(y_scores)//2]
        x_median = sorted(x_scores)[len(x_scores)//2]
        
        # Calculate variance for confidence assessment
        y_variance = max(y_scores) - min(y_scores)
        x_variance = max(x_scores) - min(x_scores)
        
        # Determine confidence level
        max_variance = max(y_variance, x_variance)
        if max_variance <= 1:
            confidence = "High"
        elif max_variance <= 2:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            "y_consensus": y_median,
            "x_consensus": x_median,
            "y_variance": y_variance,
            "x_variance": x_variance,
            "confidence_level": confidence,
            "requires_additional_research": max_variance > 2
        }

    @staticmethod
    def get_disruption_workflow_example() -> Dict[str, str]:
        """
        Provide example workflow for complete disruption analysis.
        
        Returns:
            Dict with example usage of optimized prompts for disruption analysis
        """
        
        return {
            "step_1_planning": """
# Example: Planning Phase
query = "Procore construction project management automation"
domain = "construction"

prompt = PromptFamily.generate_disruption_planning_prompt(
    query=query, 
    domain=domain, 
    max_subqueries=3
)

# Expected sub-queries:
# 1. "Construction project scheduling automation tools adoption rates 2024-2025"
# 2. "Procore market penetration construction industry competitive analysis"  
# 3. "AI task automation construction project management feasibility study"
""",
            
            "step_2_execution": """
# Example: Execution Phase  
query = "Procore construction automation potential"
sources_data = "[scraped content from research]"

prompt = PromptFamily.generate_disruption_execution_prompt(
    query=query,
    sources_data=sources_data,
    max_evidence_items=6
)

# Expected evidence format:
# <evidence>
# <indicator>Task Automation Potential</indicator>
# <score>3</score>
# <axis>Y-axis</axis>
# <cites>[McKinsey 2024: Construction Tech Report]</cites>
# <rationale>Limited automation in core workflows, manual scheduling dominates</rationale>
# </evidence>
""",
            
            "step_3_publishing": """
# Example: Publishing Phase
query = "Procore disruption analysis"
evidence_data = "[extracted evidence with Y/X scores]"

prompt = PromptFamily.generate_disruption_publishing_prompt(
    query=query,
    evidence_data=evidence_data,
    company_name="Procore"
)

# Expected analysis format:
# <analysis>
# <y_scores>[3, 4, 3]</y_scores>
# <x_scores>[7, 8, 6]</x_scores>
# <median_y_score>3</median_y_score>
# <median_x_score>7</median_x_score>
# <quadrant>Low-End Disruptive</quadrant>
# <confidence_level>Medium</confidence_level>
# <rationale>High market penetration but limited automation potential...</rationale>
# </analysis>
""",
            
            "step_4_validation": """
# Example: Self-Consistency Validation
base_analysis = "[initial disruption analysis]"

variants_prompt = PromptFamily.generate_disruption_variants_prompt(
    base_analysis=base_analysis,
    variant_count=3
)

# Calculate consensus
consensus = PromptFamily.calculate_consensus_scores([
    {"y_score": 3, "x_score": 7},
    {"y_score": 4, "x_score": 6}, 
    {"y_score": 3, "x_score": 8}
])

# Result: {"y_consensus": 3, "x_consensus": 7, "confidence_level": "High"}
"""
        }

    ################################################################################################

    # DETAILED REPORT PROMPTS

    @staticmethod
    def generate_subtopics_prompt() -> str:
        return """
Provided the main topic:

{task}

and research data:

{data}

- Construct a list of subtopics which indicate the headers of a report document to be generated on the task.
- These are a possible list of subtopics : {subtopics}.
- There should NOT be any duplicate subtopics.
- Limit the number of subtopics to a maximum of {max_subtopics}
- Finally order the subtopics by their tasks, in a relevant and meaningful order which is presentable in a detailed report

"IMPORTANT!":
- Every subtopic MUST be relevant to the main topic and provided research data ONLY!

{format_instructions}
"""

    @staticmethod
    def generate_subtopic_report_prompt(
        current_subtopic,
        existing_headers: list,
        relevant_written_contents: list,
        main_topic: str,
        context,
        report_format: str = "apa",
        max_subsections=5,
        total_words=800,
        tone: Tone = Tone.Objective,
        language: str = "english",
    ) -> str:
        return f"""
Context:
"{context}"

Main Topic and Subtopic:
Using the latest information available, construct a detailed report on the subtopic: {current_subtopic} under the main topic: {main_topic}.
You must limit the number of subsections to a maximum of {max_subsections}.

Content Focus:
- The report should focus on answering the question, be well-structured, informative, in-depth, and include facts and numbers if available.
- Use markdown syntax and follow the {report_format.upper()} format.
- When presenting data, comparisons, or structured information, use markdown tables to enhance readability.

IMPORTANT:Content and Sections Uniqueness:
- This part of the instructions is crucial to ensure the content is unique and does not overlap with existing reports.
- Carefully review the existing headers and existing written contents provided below before writing any new subsections.
- Prevent any content that is already covered in the existing written contents.
- Do not use any of the existing headers as the new subsection headers.
- Do not repeat any information already covered in the existing written contents or closely related variations to avoid duplicates.
- If you have nested subsections, ensure they are unique and not covered in the existing written contents.
- Ensure that your content is entirely new and does not overlap with any information already covered in the previous subtopic reports.

"Existing Subtopic Reports":
- Existing subtopic reports and their section headers:

    {existing_headers}

- Existing written contents from previous subtopic reports:

    {relevant_written_contents}

"Structure and Formatting":
- As this sub-report will be part of a larger report, include only the main body divided into suitable subtopics without any introduction or conclusion section.

- You MUST include markdown hyperlinks to relevant source URLs wherever referenced in the report, for example:

    ### Section Header

    This is a sample text ([in-text citation](url)).

- Use H2 for the main subtopic header (##) and H3 for subsections (###).
- Use smaller Markdown headers (e.g., H2 or H3) for content structure, avoiding the largest header (H1) as it will be used for the larger report's heading.
- Organize your content into distinct sections that complement but do not overlap with existing reports.
- When adding similar or identical subsections to your report, you should clearly indicate the differences between and the new content and the existing written content from previous subtopic reports. For example:

    ### New header (similar to existing header)

    While the previous section discussed [topic A], this section will explore [topic B]."

"Date":
Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.

"IMPORTANT!":
- You MUST write the report in the following language: {language}.
- The focus MUST be on the main topic! You MUST Leave out any information un-related to it!
- Must NOT have any introduction, conclusion, summary or reference section.
- You MUST use in-text citation references in {report_format.upper()} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).
- You MUST mention the difference between the existing content and the new content in the report if you are adding the similar or same subsections wherever necessary.
- The report should have a minimum length of {total_words} words.
- Use an {tone.value} tone throughout the report.

Do NOT add a conclusion section.
"""

    @staticmethod
    def generate_draft_titles_prompt(
        current_subtopic: str,
        main_topic: str,
        context: str,
        max_subsections: int = 5
    ) -> str:
        return f"""
"Context":
"{context}"

"Main Topic and Subtopic":
Using the latest information available, construct a draft section title headers for a detailed report on the subtopic: {current_subtopic} under the main topic: {main_topic}.

"Task":
1. Create a list of draft section title headers for the subtopic report.
2. Each header should be concise and relevant to the subtopic.
3. The header should't be too high level, but detailed enough to cover the main aspects of the subtopic.
4. Use markdown syntax for the headers, using H3 (###) as H1 and H2 will be used for the larger report's heading.
5. Ensure the headers cover main aspects of the subtopic.

"Structure and Formatting":
Provide the draft headers in a list format using markdown syntax, for example:

### Header 1
### Header 2
### Header 3

"IMPORTANT!":
- The focus MUST be on the main topic! You MUST Leave out any information un-related to it!
- Must NOT have any introduction, conclusion, summary or reference section.
- Focus solely on creating headers, not content.
"""

    @staticmethod
    def generate_report_introduction(question: str, research_summary: str = "", language: str = "english", report_format: str = "apa") -> str:
        return f"""{research_summary}\n
Using the above latest information, Prepare a detailed report introduction on the topic -- {question}.
- The introduction should be succinct, well-structured, informative with markdown syntax.
- As this introduction will be part of a larger report, do NOT include any other sections, which are generally present in a report.
- The introduction should be preceded by an H1 heading with a suitable topic for the entire report.
- You must use in-text citation references in {report_format.upper()} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).
Assume that the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.
- The output must be in {language} language.
"""


    @staticmethod
    def generate_report_conclusion(query: str, report_content: str, language: str = "english", report_format: str = "apa") -> str:
        """
        Generate a concise conclusion summarizing the main findings and implications of a research report.

        Args:
            query (str): The research task or question.
            report_content (str): The content of the research report.
            language (str): The language in which the conclusion should be written.

        Returns:
            str: A concise conclusion summarizing the report's main findings and implications.
        """
        prompt = f"""
    Based on the research report below and research task, please write a concise conclusion that summarizes the main findings and their implications:

    Research task: {query}

    Research Report: {report_content}

    Your conclusion should:
    1. Recap the main points of the research
    2. Highlight the most important findings
    3. Discuss any implications or next steps
    4. Be approximately 2-3 paragraphs long

    If there is no "## Conclusion" section title written at the end of the report, please add it to the top of your conclusion.
    You must use in-text citation references in {report_format.upper()} format and make it with markdown hyperlink placed at the end of the sentence or paragraph that references them like this: ([in-text citation](url)).

    IMPORTANT: The entire conclusion MUST be written in {language} language.

    Write the conclusion:
    """

        return prompt


class GranitePromptFamily(PromptFamily):
    """Prompts for IBM's granite models"""


    def _get_granite_class(self) -> type[PromptFamily]:
        """Get the right granite prompt family based on the version number"""
        if "3.3" in self.cfg.smart_llm:
            return Granite33PromptFamily
        if "3" in self.cfg.smart_llm:
            return Granite3PromptFamily
        # If not a known version, return the default
        return PromptFamily

    def pretty_print_docs(self, *args, **kwargs) -> str:
        return self._get_granite_class().pretty_print_docs(*args, **kwargs)

    def join_local_web_documents(self, *args, **kwargs) -> str:
        return self._get_granite_class().join_local_web_documents(*args, **kwargs)


class Granite3PromptFamily(PromptFamily):
    """Prompts for IBM's granite 3.X models (before 3.3)"""

    _DOCUMENTS_PREFIX = "<|start_of_role|>documents<|end_of_role|>\n"
    _DOCUMENTS_SUFFIX = "\n<|end_of_text|>"

    @classmethod
    def pretty_print_docs(cls, docs: list[Document], top_n: int | None = None) -> str:
        if not docs:
            return ""
        all_documents = "\n\n".join([
            f"Document {doc.metadata.get('source', i)}\n" + \
            f"Title: {doc.metadata.get('title')}\n" + \
            doc.page_content
            for i, doc in enumerate(docs)
            if top_n is None or i < top_n
        ])
        return "".join([cls._DOCUMENTS_PREFIX, all_documents, cls._DOCUMENTS_SUFFIX])

    @classmethod
    def join_local_web_documents(cls, docs_context: str | list, web_context: str | list) -> str:
        """Joins local web documents using Granite's preferred format"""
        if isinstance(docs_context, str) and docs_context.startswith(cls._DOCUMENTS_PREFIX):
            docs_context = docs_context[len(cls._DOCUMENTS_PREFIX):]
        if isinstance(web_context, str) and web_context.endswith(cls._DOCUMENTS_SUFFIX):
            web_context = web_context[:-len(cls._DOCUMENTS_SUFFIX)]
        all_documents = "\n\n".join([docs_context, web_context])
        return "".join([cls._DOCUMENTS_PREFIX, all_documents, cls._DOCUMENTS_SUFFIX])


class Granite33PromptFamily(PromptFamily):
    """Prompts for IBM's granite 3.3 models"""

    _DOCUMENT_TEMPLATE = """<|start_of_role|>document {{"document_id": "{document_id}"}}<|end_of_role|>
{document_content}<|end_of_text|>
"""

    @staticmethod
    def _get_content(doc: Document) -> str:
        doc_content = doc.page_content
        if title := doc.metadata.get("title"):
            doc_content = f"Title: {title}\n{doc_content}"
        return doc_content.strip()

    @classmethod
    def pretty_print_docs(cls, docs: list[Document], top_n: int | None = None) -> str:
        return "\n".join([
            cls._DOCUMENT_TEMPLATE.format(
                document_id=doc.metadata.get("source", i),
                document_content=cls._get_content(doc),
            )
            for i, doc in enumerate(docs)
            if top_n is None or i < top_n
        ])

    @classmethod
    def join_local_web_documents(cls, docs_context: str | list, web_context: str | list) -> str:
        """Joins local web documents using Granite's preferred format"""
        return "\n\n".join([docs_context, web_context])

## Factory ######################################################################

# This is the function signature for the various prompt generator functions
PROMPT_GENERATOR = Callable[
    [
        str,        # question
        str,        # context
        str,        # report_source
        str,        # report_format
        str | None, # tone
        int,        # total_words
        str,        # language
    ],
    str,
]

report_type_mapping = {
    ReportType.ResearchReport.value: "generate_report_prompt",
    ReportType.ResourceReport.value: "generate_resource_report_prompt",
    ReportType.OutlineReport.value: "generate_outline_report_prompt",
    ReportType.CustomReport.value: "generate_custom_report_prompt",
    ReportType.SubtopicReport.value: "generate_subtopic_report_prompt",
    ReportType.DeepResearch.value: "generate_deep_research_prompt",
    "DisruptionAnalysis": "generate_disruption_publishing_prompt",
}


def get_prompt_by_report_type(
    report_type: str,
    prompt_family: type[PromptFamily] | PromptFamily,
):
    prompt_by_type = getattr(prompt_family, report_type_mapping.get(report_type, ""), None)
    default_report_type = ReportType.ResearchReport.value
    if not prompt_by_type:
        warnings.warn(
            f"Invalid report type: {report_type}.\n"
            f"Please use one of the following: {', '.join([enum_value for enum_value in report_type_mapping.keys()])}\n"
            f"Using default report type: {default_report_type} prompt.",
            UserWarning,
        )
        prompt_by_type = getattr(prompt_family, report_type_mapping.get(default_report_type))
    return prompt_by_type


prompt_family_mapping = {
    PromptFamilyEnum.Default.value: PromptFamily,
    PromptFamilyEnum.Granite.value: GranitePromptFamily,
    PromptFamilyEnum.Granite3.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite31.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite32.value: Granite3PromptFamily,
    PromptFamilyEnum.Granite33.value: Granite33PromptFamily,
}


def get_prompt_family(
    prompt_family_name: PromptFamilyEnum | str, config: Config,
) -> PromptFamily:
    """Get a prompt family by name or value."""
    if isinstance(prompt_family_name, PromptFamilyEnum):
        prompt_family_name = prompt_family_name.value
    if prompt_family := prompt_family_mapping.get(prompt_family_name):
        return prompt_family(config)
    warnings.warn(
        f"Invalid prompt family: {prompt_family_name}.\n"
        f"Please use one of the following: {', '.join([enum_value for enum_value in prompt_family_mapping.keys()])}\n"
        f"Using default prompt family: {PromptFamilyEnum.Default.value} prompt.",
        UserWarning,
    )
    return PromptFamily()
