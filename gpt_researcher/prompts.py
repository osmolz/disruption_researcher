"""
GPT-Researcher Prompts - Optimized for Raw Evidence Gathering Framework

This module contains YC-style prompts optimized for AI disruption analysis with focus on:
- Y-axis (Automation Potential): How easy can AI automate tasks WITHIN the workflow - within the application software how easy is it for the task being done to be automated or streamlined with AI (emphasis on risk of error and repetitiveness)
- X-axis (Workflow Penetration): How easy is it for generative or agentic AI to siphon value from the software by mimicking the core logic and process the software provides (NOT market penetration)

Key Framework Components:
1. Planning Phase: Strategic sub-query generation targeting specific indicators
2. Execution Phase: Raw evidence extraction with 2-3 citations from 2023-2025 sources  
3. Publishing Phase: Evidence synthesis without scoring or quadrant positioning

Quality Controls:
- Conservative approach: Raw data only, no scoring/bucketing for accurate downstream analysis
- Source validation: Minimum 4 sources, prioritize recent/trusted/relevant
- Self-consistency: Generate 3 variants, vote on coverage >=7 indicators
- Error handling: Return underspecification errors for insufficient evidence coverage

Output Formats:
- XML for evidence: <evidence><indicator>...</indicator><raw>...</raw><cites>[...]</cites></evidence>
- JSON for final analysis with indicators_evidence: {Y: {indicator1: [raw/cites]}, X: {indicator1: [raw/cites]}}
"""

import warnings
from datetime import date, datetime, timezone

from langchain.docstore.document import Document

from .config import Config
from .utils.enum import ReportSource, ReportType, Tone
from .utils.enum import PromptFamily as PromptFamilyEnum
from typing import Callable, List, Dict, Any

# DISRUPTION INDICATORS CONSTANTS ##############################################

# CoT: Define 6 Y-axis indicators for automation potential within workflow - how easy can AI automate tasks WITHIN the application software (emphasis on risk of error and repetitiveness)
Y_AUTOMATION_INDICATORS = {
    "Y1_TASK_STRUCTURE": "Task Structure & Repetition Analysis - patterns of repetitive tasks within workflow",
    "Y2_ERROR_RISK": "Risk of Error & Quality Control Requirements - crucial factor for automation feasibility", 
    "Y3_CONTEXTUAL_KNOWLEDGE": "Contextual Knowledge & Domain Expertise Needs - complexity of knowledge required within workflow",
    "Y4_DATA_AVAILABILITY": "Data Availability & Structure for Automation - internal workflow data accessibility",
    "Y5_PROCESS_VARIABILITY": "Process Variability & Exception Handling - consistency of workflow processes",
    "Y6_HUMAN_WORKFLOW": "Human Workflow & UI Dependency Assessment - human interaction requirements within application"
}

# CoT: Define 6 X-axis indicators for workflow penetration - how easy is it for generative or agentic AI to siphon value by mimicking core logic and processes (NOT market penetration)
X_WORKFLOW_PENETRATION_INDICATORS = {
    "X1_EXTERNAL_OBSERVABILITY": "External Observability of Workflow Logic - how visible are the core processes to external AI systems",
    "X2_INDUSTRY_STANDARDIZATION": "Industry Standardization & Protocol Adoption - standardized workflows easier to mimic",
    "X3_PROPRIETARY_DATA": "Proprietary Data Depth & Competitive Moats - exclusive workflow data protecting against AI mimicry", 
    "X4_SWITCHING_FRICTION": "Switching Cost & Network Effect Barriers - barriers preventing workflow replication",
    "X5_REGULATORY_BARRIERS": "Regulatory & Certification Requirements - legal protections against workflow mimicry",
    "X6_AGENT_PROTOCOL": "Agent Protocol Maturity & Integration Standards - ease of AI systems integrating with workflow"
}

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
        YC-style planning prompt for raw evidence gathering sub-query generation.
        Targets 6 Y-axis (automation) and 6 X-axis (penetration) indicators.
        
        CoT: Step 1 - Generate targeted sub-queries for each indicator
        CoT: Step 2 - Focus on raw evidence gathering, not scoring
        CoT: Step 3 - Ensure 2023-2025 source targeting for relevance
        """
        
        context_info = f"Context: {context}" if context else ""
        
        # CoT: Create indicator-specific query examples for targeting
        y_indicators_list = "\n".join([f"- {key}: {desc}" for key, desc in Y_AUTOMATION_INDICATORS.items()])
        x_indicators_list = "\n".join([f"- {key}: {desc}" for key, desc in X_WORKFLOW_PENETRATION_INDICATORS.items()])
        
        return f"""# Role: Strategic Planning Specialist for Raw Evidence Collection
You are an expert strategist specializing in generating targeted sub-queries to gather raw evidence for AI disruption indicators, with focus on collecting unprocessed data for downstream analysis.

### Task: Generate Evidence-Gathering Sub-Queries  
Generate {max_subqueries} focused sub-queries to collect raw evidence for: "{query}" in domain: "{domain}"
Target specific Y-axis (automation) and X-axis (penetration) indicators from 2023-2025 sources.

### Y-Axis Automation Indicators (How easy can AI automate tasks WITHIN the workflow):
{y_indicators_list}

### X-Axis Workflow Penetration Indicators (How easy for AI to siphon value by mimicking core logic):  
{x_indicators_list}

### Plan:
Step 1: Target specific indicators - Generate sub-queries for Y1-Y6 and X1-X6 evidence
Step 2: Raw evidence focus - Seek task complexity data, adoption metrics, barrier analysis
Step 3: Source specification - Target authoritative sources (McKinsey, Bain, industry reports)
Step 4: Coverage validation - Ensure minimum 3 indicators covered per axis

### Guidelines:
- **Raw Evidence Focus**: Generate queries to extract data/facts per indicator, NOT scores
- **Indicator Targeting**: Each sub-query should target 1-2 specific indicators explicitly
- **Source Quality**: Target trusted sources from 2023-2025 with quantitative data
- **Coverage Requirement**: Aim for minimum 3 Y-axis and 3 X-axis indicators covered
- **Conservative Approach**: If <3 indicators per axis targetable, flag underspecification

### Few-Shot Examples:
**High-Quality Sub-Query (Targets Y1 + Y4)**
- "Company task structure repetition analysis {domain} automation data availability 2023-2025"
- Evidence Target: Y1 (task patterns), Y4 (data structure for automation)

**High-Quality Sub-Query (Targets X2 + X5)**  
- "{domain} industry standardization protocols regulatory certification requirements 2024"
- Evidence Target: X2 (standardization), X5 (regulatory barriers)

**Low-Quality Sub-Query**
- "General {domain} industry trends 2024"
- Issue: No specific indicator targeting, too broad for evidence extraction

{context_info}

### Output Format:
<plan>
<step>
<sub_query>{domain} task structure repetition automation evidence Y1 indicator analysis 2024-2025</sub_query>
<target_indicators>Y1_TASK_STRUCTURE</target_indicators>
<evidence_focus>Raw data on task patterns, repetition rates, automation feasibility</evidence_focus>
</step>
<step>
<sub_query>{domain} workflow penetration external observability X1 AI visibility core processes 2024</sub_query>
<target_indicators>X1_EXTERNAL_OBSERVABILITY</target_indicators>
<evidence_focus>Raw workflow logic visibility data, AI observability of core processes</evidence_focus>
</step>
</plan>

Generate exactly {max_subqueries} sub-queries covering minimum 3 Y and 3 X indicators.
If insufficient indicator coverage possible, return: "Underspecification: Cannot target minimum 3 Y/X indicators - refine query scope"

Current date context: {datetime.now(timezone.utc).strftime('%B %d, %Y')}
"""

    @staticmethod
    def generate_disruption_execution_prompt(
        query: str,
        sources_data: str,
        max_evidence_items: int = 12,
    ) -> str:
        """
        YC-style execution prompt for raw evidence extraction per indicator.
        
        CoT: Step 1 - Extract raw evidence per Y/X indicator without scoring
        CoT: Step 2 - Preserve citations and quantitative data  
        CoT: Step 3 - Validate coverage >=7 indicators or escape
        """
        
        # CoT: Create searchable indicator reference for evidence extraction
        y_indicators_ref = "\n".join([f"- {key}: {desc}" for key, desc in Y_AUTOMATION_INDICATORS.items()])
        x_indicators_ref = "\n".join([f"- {key}: {desc}" for key, desc in X_WORKFLOW_PENETRATION_INDICATORS.items()])
        
        return f"""# Role: Raw Evidence Extraction Specialist
You are a senior analyst specializing in extracting raw automation and penetration indicator evidence from research sources with rigorous citation standards, focusing on unprocessed data for downstream analysis.

### Task: Extract Raw Indicator Evidence
Analyze sources for query: "{query}"
Extract raw evidence for Y-axis (automation) and X-axis (penetration) indicators with 2-3 citations each.

### Y-Axis Indicators (How easy can AI automate tasks WITHIN the workflow):
{y_indicators_ref}

### X-Axis Indicators (How easy for AI to siphon value by mimicking core logic):
{x_indicators_ref}

### Plan:
Step 1: Identify indicator-specific evidence - Map source content to Y1-Y6, X1-X6 indicators
Step 2: Extract raw data - Preserve numbers, percentages, quotes, concrete facts (NO scoring)
Step 3: Validate citations - Ensure 2-3 trusted sources (2023-2025) per evidence item
Step 4: Coverage check - Minimum 7 indicators covered or flag underspecification

### Guidelines:
- **Raw Evidence Only**: Extract facts, data, quotes, metrics - NO scoring or interpretation
- **Citation Requirement**: Minimum 2-3 citations from trusted sources (2023-2025)
- **Indicator Mapping**: Each evidence item must map to specific Y or X indicator
- **Coverage Validation**: Minimum 7 indicators total (mix of Y and X), escape if <7
- **Conservative Quality**: Raw data only for accurate downstream scoring/bucketing

### Evidence Extraction Standards:
- **Quantitative Priority**: Numbers, percentages, adoption rates, cost data
- **Authoritative Sources**: McKinsey, Bain, industry leaders, academic research
- **Recent Focus**: 2023-2025 sources prioritized for current landscape
- **Context Preservation**: Keep raw context around data points for reliability

### Source Data:
{sources_data}

### Output Format:
Extract up to {max_evidence_items} pieces of raw evidence. If <7 indicators covered:

<if_block condition="insufficient_coverage">
{{"error": "Underspecification: Low indicators evidence coverage. Found X indicators, need minimum 7 - refine query or expand sources."}}
</if_block>

Otherwise, provide raw evidence in XML format:

<evidence>
<indicator>Y1_TASK_STRUCTURE</indicator>
<raw>Construction project scheduling involves 73% repetitive tasks according to industry analysis, with standardized workflows in planning, resource allocation, and progress tracking phases</raw>
<cites>[McKinsey 2024: Construction Automation Report, Bain 2025: Project Management AI Adoption]</cites>
</evidence>

<evidence>
<indicator>X2_INDUSTRY_STANDARDIZATION</indicator>
<raw>Construction industry shows 42% adoption of standardized project management protocols, with limited integration between major platforms like Procore, Autodesk, and Oracle</raw>
<cites>[Construction Tech Report 2024, Industry Week 2025: Platform Integration Study]</cites>
</evidence>

Extract raw evidence for minimum 7 different indicators with quality citations.
"""

    @staticmethod
    def generate_disruption_publishing_prompt(
        query: str,
        evidence_data: str,
        company_name: str = "",
    ) -> str:
        """
        YC-style publishing prompt for raw evidence synthesis into JSON output.
        
        CoT: Step 1 - Organize raw evidence by indicator categories  
        CoT: Step 2 - Synthesize implications without scoring/quadrants
        CoT: Step 3 - Output structured JSON for downstream analysis
        """
        
        return f"""# Role: Evidence Synthesis Specialist  
You are a senior strategist responsible for organizing raw evidence into structured analysis without scoring or quadrant positioning, preserving data integrity for downstream scoring systems.

### Task: Synthesize Raw Evidence into Structured Output
Synthesize evidence for: "{query}" {f"(Company: {company_name})" if company_name else ""}
Organize raw indicator evidence and implications without scoring or bucketing.

### Plan:
Step 1: Organize evidence by Y-axis indicators (Y1-Y6 automation potential within workflow)
Step 2: Organize evidence by X-axis indicators (X1-X6 workflow penetration by AI mimicking)
Step 3: Identify strategic implications based on raw evidence patterns
Step 4: Preserve all quantitative data and citations for downstream analysis

### Evidence Data:
{evidence_data}

### Guidelines:
- **Raw Evidence Focus**: Organize and preserve evidence without scoring or interpretation
- **No Quadrant Positioning**: Avoid Low-End/New-Market/Big Bang classifications
- **Citation Preservation**: Maintain all source attribution and quantitative data
- **Implication Analysis**: Identify patterns and strategic implications from raw evidence
- **Downstream Ready**: Structure for accurate scoring/bucketing by downstream systems

### Output Format:
Provide analysis in JSON structure:

```json
{{
  "query": "{query}",
  "company": "{company_name}",
  "indicators_evidence": {{
    "Y_automation_potential": {{
      "Y1_task_structure": {{
        "raw_evidence": ["Task repetition within workflow", "Workflow standardization", "..."],
        "citations": ["Source 1", "Source 2", "..."],
        "key_metrics": ["73% repetitive tasks within app", "Average 4.2 complexity score", "..."]
      }},
      "Y2_error_risk": {{
        "raw_evidence": ["Error risk within workflow tasks", "Quality control for automation", "..."],
        "citations": ["Source 1", "Source 2", "..."],
        "key_metrics": ["12% error rate impact", "$50K average error cost", "..."]
      }}
    }},
    "X_workflow_penetration": {{
      "X1_external_observability": {{
        "raw_evidence": ["Workflow logic visibility to external AI", "Process observability", "..."],
        "citations": ["Source 1", "Source 2", "..."],
        "key_metrics": ["42% workflow visibility", "8 observable core processes", "..."]
      }},
      "X2_industry_standardization": {{
        "raw_evidence": ["Standardized workflow protocols", "AI mimicry ease via standards", "..."],
        "citations": ["Source 1", "Source 2", "..."],
        "key_metrics": ["35% protocol standardization", "14 workflow standards", "..."]
      }}
    }}
  }},
  "strategic_implications": {{
    "automation_readiness": "Evidence-based assessment of AI automation potential within workflow",
    "workflow_vulnerability": "Assessment of workflow penetration risk by AI mimicking core logic",
    "implementation_barriers": "Concrete barriers identified from evidence",
    "opportunity_areas": "Data-driven opportunity identification for workflow automation and protection"
  }},
  "evidence_quality": {{
    "total_indicators_covered": 8,
    "y_indicators_covered": 4,
    "x_indicators_covered": 4,
    "recent_sources_count": 12,
    "authoritative_sources_count": 8
  }}
}}
```

If evidence coverage is insufficient (<7 indicators total):
{{"error": "Underspecification: Insufficient evidence quality for reliable analysis. Covered X indicators, need minimum 7. Recommend additional research on [specific indicator gaps]."}}

Focus on raw evidence organization and strategic implications without scoring or quadrant positioning.
"""

    @staticmethod
    def generate_self_consistency_check(
        original_analysis: str,
        variant_count: int = 3,
    ) -> str:
        """
        Generate self-consistency check prompt for coverage validation (not scoring).
        
        CoT: Step 1 - Generate variants focused on evidence coverage
        CoT: Step 2 - Vote on coverage completeness >=7 indicators  
        CoT: Step 3 - Validate evidence quality and consistency
        """
        
        return f"""# Role: Evidence Coverage Validation Specialist
You are responsible for ensuring consistency and reliability in evidence coverage through multi-variant validation, focusing on indicator completeness rather than score agreement.

### Task: Self-Consistency Coverage Validation
Review the original analysis and generate {variant_count} alternative evidence interpretations.
Vote on coverage completeness for final validation (minimum 7 indicators).

### Original Analysis:
{original_analysis}

### Validation Process:
1. Generate {variant_count} alternative evidence coverage assessments using same sources
2. Count indicators covered in each variant (Y1-Y6, X1-X6)
3. Vote on median coverage count for validation
4. Flag coverage gaps if <7 indicators consistently identified
5. Provide consensus recommendation on evidence quality

### Coverage Standards:
- **Minimum Coverage**: 7 indicators total (mix of Y and X indicators)
- **Quality Threshold**: Each indicator needs 2+ citations from 2023-2025 sources
- **Consistency Check**: All variants should identify similar indicator coverage
- **Gap Analysis**: Flag missing indicators that should be researched further

### Output Format:
<validation>
<variant_coverage>
<variant_1>
<y_indicators_covered>4</y_indicators_covered>
<x_indicators_covered>3</x_indicators_covered>
<total_coverage>7</total_coverage>
<covered_indicators>["Y1", "Y3", "Y4", "Y6", "X1", "X2", "X5"]</covered_indicators>
</variant_1>
<variant_2>
<y_indicators_covered>3</y_indicators_covered>
<x_indicators_covered>4</x_indicators_covered>
<total_coverage>7</total_coverage>
<covered_indicators>["Y1", "Y2", "Y4", "X1", "X2", "X3", "X6"]</covered_indicators>
</variant_2>
<variant_3>
<y_indicators_covered>4</y_indicators_covered>
<x_indicators_covered>4</x_indicators_covered>
<total_coverage>8</total_coverage>
<covered_indicators>["Y1", "Y2", "Y4", "Y5", "X1", "X2", "X4", "X6"]</covered_indicators>
</variant_3>
</variant_coverage>
<coverage_consensus>
<median_coverage>7</median_coverage>
<coverage_confidence>{"High" if median >=7 else "Low"}</coverage_confidence>
<consistently_covered>["Y1", "Y4", "X1", "X2"]</consistently_covered>
<gaps_identified>["Y3", "Y5", "X3"]</gaps_identified>
</coverage_consensus>
<recommendation>{"Evidence coverage adequate for analysis" if median >=7 else "Insufficient coverage - expand research on gaps identified"}</recommendation>
</validation>

Focus on coverage validation and evidence quality, not score consistency.
"""

    @staticmethod
    def generate_disruption_variants_prompt(base_analysis: str, variant_count: int = 3) -> str:
        """
        Generate prompt for creating analysis variants for self-consistency validation.
        
        CoT: Step 1 - Generate evidence coverage variants, not score variants
        CoT: Step 2 - Focus on indicator identification and evidence completeness
        CoT: Step 3 - Maintain raw evidence focus for downstream analysis
        """
        
        return f"""# Role: Multi-Perspective Evidence Assessment Specialist
You are responsible for generating alternative evidence coverage interpretations to ensure consistency and reliability through variant validation, focusing on indicator identification rather than scoring.

### Task: Generate Evidence Coverage Variants
Create {variant_count} alternative evidence coverage assessments of the base analysis using the same source data but with different analytical perspectives for indicator identification.

### Base Analysis:
{base_analysis}

### Variant Generation Guidelines:
1. **Maintain Evidence Base**: Use only the same evidence from base analysis
2. **Alternative Perspectives**: Apply different analytical lenses for indicator identification
3. **Coverage Focus**: Count indicators covered (Y1-Y6, X1-X6) rather than generating scores
4. **Consistent Framework**: Maintain raw evidence approach throughout all variants

### Variant Types:
**Variant 1 - Conservative Assessment**: 
- Emphasize strict indicator matching criteria
- Count only clear, well-supported indicator evidence
- Focus on high-confidence indicator identification

**Variant 2 - Balanced Assessment**:
- Standard perspective for indicator identification
- Consider both direct and indirect evidence for indicators
- Balanced approach to evidence-to-indicator mapping

**Variant 3 - Comprehensive Assessment**:
- Broader interpretation of indicator evidence
- Include emerging patterns and forward-looking indicators
- Emphasize evidence breadth and indicator coverage

### Output Format:
<variants>
<variant_1>
<y_evidence_coverage>
<Y1_task_structure>Evidence found and citations preserved</Y1_task_structure>
<Y4_data_availability>Evidence found and citations preserved</Y4_data_availability>
</y_evidence_coverage>
<x_evidence_coverage>
<X1_external_observability>Evidence found and citations preserved</X1_external_observability>
<X2_industry_standardization>Evidence found and citations preserved</X2_industry_standardization>
</x_evidence_coverage>
<perspective>Conservative</perspective>
<total_indicators>4</total_indicators>
<rationale>Strict evidence-to-indicator mapping with high confidence threshold</rationale>
</variant_1>
<variant_2>
<y_evidence_coverage>
<Y1_task_structure>Evidence found and citations preserved</Y1_task_structure>
<Y2_error_risk>Evidence found and citations preserved</Y2_error_risk>
<Y4_data_availability>Evidence found and citations preserved</Y4_data_availability>
</y_evidence_coverage>
<x_evidence_coverage>
<X1_external_observability>Evidence found and citations preserved</X1_external_observability>
<X2_industry_standardization>Evidence found and citations preserved</X2_industry_standardization>
<X5_regulatory_barriers>Evidence found and citations preserved</X5_regulatory_barriers>
</x_evidence_coverage>
<perspective>Balanced</perspective>
<total_indicators>6</total_indicators>
<rationale>Balanced evidence interpretation with standard indicator matching</rationale>
</variant_2>
<variant_3>
<y_evidence_coverage>
<Y1_task_structure>Evidence found and citations preserved</Y1_task_structure>
<Y2_error_risk>Evidence found and citations preserved</Y2_error_risk>
<Y4_data_availability>Evidence found and citations preserved</Y4_data_availability>
<Y6_human_workflow>Evidence found and citations preserved</Y6_human_workflow>
</y_evidence_coverage>
<x_evidence_coverage>
<X1_external_observability>Evidence found and citations preserved</X1_external_observability>
<X2_industry_standardization>Evidence found and citations preserved</X2_industry_standardization>
<X4_switching_friction>Evidence found and citations preserved</X4_switching_friction>
<X5_regulatory_barriers>Evidence found and citations preserved</X5_regulatory_barriers>
</x_evidence_coverage>
<perspective>Comprehensive</perspective>
<total_indicators>8</total_indicators>
<rationale>Comprehensive evidence interpretation emphasizing coverage breadth</rationale>
</variant_3>
</variants>

Generate exactly {variant_count} variants with indicator coverage counts and preserved evidence for each perspective.
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
        """Generates YC-style search queries prompt optimized for raw evidence gathering.
        
        CoT: Step 1 - Target specific Y/X indicators in search queries
        CoT: Step 2 - Focus on raw data extraction, not scoring preparation
        CoT: Step 3 - Structure queries for evidence-rich source discovery
        
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

Use this real-time context to enhance search query specificity, focusing on recent developments (2023-2025) that provide raw evidence for the 12 disruption indicators.
""" if context else ""

        dynamic_example = ", ".join([f'"query {i+1}"' for i in range(max_iterations)])

        return f"""# Role: Strategic Search Query Specialist for Raw Evidence Collection
You are an expert research strategist specializing in generating targeted search queries that uncover raw evidence for the 6 Y-axis (automation) and 6 X-axis (penetration) disruption indicators.

### Task: Generate Evidence-Gathering Search Queries
Generate {max_iterations} strategic search queries for: "{task}"
Focus on collecting raw data and facts for specific Y1-Y6 and X1-X6 indicators from 2023-2025 sources.

### Y-Axis Indicators (AI automation potential WITHIN workflow):
- Y1: Task structure & repetition patterns within application
- Y2: Error risk & quality control data (crucial for automation within workflow)
- Y3: Contextual knowledge requirements within workflow tasks
- Y4: Data availability & structure metrics for internal automation
- Y5: Process variability & exception rates within application
- Y6: Human workflow & UI dependencies within application

### X-Axis Indicators (AI workflow penetration by mimicking core logic):
- X1: External observability of workflow logic & core processes
- X2: Industry standardization enabling AI to mimic workflows
- X3: Proprietary data depth protecting against workflow mimicry
- X4: Switching costs & barriers preventing workflow replication
- X5: Regulatory & certification requirements protecting workflows
- X6: Agent protocol maturity enabling AI integration with workflows

### Plan:
Step 1: Indicator-specific queries - Target Y1-Y6 automation evidence with quantitative data
Step 2: Penetration-focused queries - Target X1-X6 market evidence with adoption metrics
Step 3: Authority source queries - Ensure access to McKinsey, Bain, industry reports with citations

### Guidelines:
- **Indicator Targeting**: Each query should target 1-2 specific indicators explicitly
- **Raw Evidence Focus**: Structure queries to find concrete data, not opinions or scores
- **Temporal Focus**: Prioritize 2023-2025 sources for current landscape evidence
- **Authority Targeting**: Include terms for trusted sources with quantitative research
- **Citation Readiness**: Structure queries to find sources with data and proper attribution

### Search Query Types (Indicator-Focused):
1. **Y-Indicator Queries**: "[industry] task repetition within workflow Y1 automation analysis 2024", "[domain] error rates workflow tasks Y2 automation risk 2025"
2. **X-Indicator Queries**: "[company] workflow logic observability X1 external visibility 2024", "[industry] workflow standardization X2 AI mimicry ease 2025"  
3. **Authority Evidence**: "McKinsey [industry] workflow automation indicators 2024", "Bain [domain] AI workflow penetration report 2025"

### Quality Standards:
- Each query must target specific indicators and recent sources (2023-2025)
- Include industry-specific terminology for precise evidence discovery
- Structure for quantitative data and metric identification
- Enable citation-rich source access for validation

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

        return f"""# Role: Strategic Analyst for Raw Evidence Reporting
You are a senior strategic analyst specializing in synthesizing raw indicator evidence into comprehensive reports that preserve data integrity for downstream analysis, avoiding premature scoring or bucketing.

### Task: Generate Raw Evidence Analysis Report
Using the research context below, generate a comprehensive evidence-based report for: "{question}"
Focus on organizing and presenting raw evidence for Y-axis (automation) and X-axis (penetration) indicators without scoring or quadrant positioning.

### Research Context:
"{context}"

### Plan:
Step 1: Organize automation evidence by Y1-Y6 indicators (automation within workflow) with quantitative data
Step 2: Present workflow penetration evidence by X1-X6 indicators (AI mimicking core logic) with metrics
Step 3: Synthesize strategic implications based on evidence patterns (no scoring)
Step 4: Identify implementation considerations and workflow protection dynamics
Step 5: Preserve all quantitative data and citations for reliability

### Report Guidelines:
**Structure Requirements**:
- **Executive Summary**: Evidence coverage overview with confidence assessment
- **Automation Evidence** (Y-Axis): Organize findings by Y1-Y6 indicators (AI automation within workflow) with raw data
- **Workflow Penetration Evidence** (X-Axis): Present findings by X1-X6 indicators (AI mimicking core logic) with metrics
- **Strategic Implications**: Evidence-based recommendations without quadrant positioning
- **Implementation Considerations**: Concrete barriers and enablers for workflow automation and protection
- **Evidence Quality Assessment**: Coverage analysis and source reliability metrics

**Quality Standards**:
- Minimum {total_words} words with comprehensive evidence presentation
- Prioritize 2023-2025 sources and authoritative publications (McKinsey, Bain, industry leaders)
- Include ALL quantitative data, statistics, and concrete metrics from evidence
- Present raw facts and findings without interpretation or scoring
- Use markdown tables for evidence organization and metric presentation

**Citation Requirements**:
- In-text citations in {report_format} format with markdown hyperlinks: ([in-text citation](url))
- Prioritize recent, trusted sources over older or less reliable ones
- Include 2-3 citations minimum per major evidence point
- {reference_prompt}

**Evidence Presentation Framework**:
- **Raw Data Focus**: Present facts, metrics, and findings without scoring or interpretation
- **Indicator Organization**: Structure evidence by specific Y1-Y6 and X1-X6 indicators
- **Citation Preservation**: Maintain exact source attribution for all quantitative claims
- **Strategic Focus**: Derive actionable insights from evidence patterns, not quadrant positioning

**Analysis Requirements**:
- Present concrete, evidence-based findings - avoid generic conclusions
- Integrate multiple source perspectives when supported by data
- Highlight ALL quantitative evidence and measurable indicators
- Address implementation barriers and market dynamics from evidence
- Assess evidence coverage and identify gaps for future research

{tone_prompt}

### Output Language: {language}

### Critical Success Factors:
This analysis will inform downstream scoring and strategic decision-making. Ensure evidence-based rigor, raw data preservation, and comprehensive indicator coverage without premature interpretation.

Current analysis date: {date.today()}
"""

    @staticmethod
    def curate_sources(query, sources, max_results=10):
        return f"""# Role: Source Curation Specialist for Disruption Analysis
You are an expert research curator specializing in identifying high-quality sources for AI disruption analysis with focus on automation potential and workflow penetration evidence.

### Task: Curate Sources for Y/X Disruption Framework
Evaluate and curate sources for: "{query}"
Prioritize sources containing Y-axis (automation potential) and X-axis (workflow penetration) indicators with quantitative data and recent citations.

### Plan:
Step 1: Assess source relevance to Y/X disruption framework
Step 2: Evaluate source authority and recency (prioritize 2023-2025)
Step 3: Prioritize quantitative data and citation-rich sources
Step 4: Select up to {max_results} highest-quality sources

### Curation Guidelines:
**Primary Criteria** (Required for inclusion):
- **Y/X Relevance**: Contains automation potential OR workflow penetration indicators
- **Quantitative Value**: Includes statistics, numbers, workflow data, or concrete metrics
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
- Industry reports with adoption rates, automation metrics, workflow penetration data
- Case studies with measurable automation/workflow penetration outcomes

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
        """Generates YC-style summary prompt optimized for raw indicator evidence extraction.
        
        CoT: Step 1 - Map content to specific Y1-Y6 and X1-X6 indicators
        CoT: Step 2 - Extract raw facts and metrics without interpretation  
        CoT: Step 3 - Preserve all quantitative data and citations for reliability
        
        Args: query (str): The disruption analysis query to extract evidence for
                data (str): The source text to analyze for specific indicators
        Returns: str: The indicator-focused summary prompt
        """

        return f"""# Role: Raw Evidence Extraction Specialist for Indicator Mapping
You are a senior analyst specializing in extracting raw evidence for specific automation and penetration indicators from research sources, focusing on unprocessed data preservation.

### Task: Extract Raw Indicator Evidence Summary
Analyze the provided text for query: "{query}"
Map content to specific Y-axis (automation) and X-axis (penetration) indicators with preserved citations.

### Y-Axis Indicators (AI automation potential WITHIN workflow - Map Content To):
- Y1_TASK_STRUCTURE: Task patterns, repetition rates within application workflow
- Y2_ERROR_RISK: Error rates, quality control requirements for automation within workflow (crucial factor)
- Y3_CONTEXTUAL_KNOWLEDGE: Domain expertise needs, knowledge complexity within workflow tasks
- Y4_DATA_AVAILABILITY: Data structure, accessibility for internal workflow automation
- Y5_PROCESS_VARIABILITY: Exception handling, process variations within application workflow
- Y6_HUMAN_WORKFLOW: UI dependencies, human interaction requirements within application

### X-Axis Indicators (AI workflow penetration by mimicking core logic - Map Content To):
- X1_EXTERNAL_OBSERVABILITY: Workflow logic visibility, observable core processes to external AI
- X2_INDUSTRY_STANDARDIZATION: Protocol adoption, workflow standardization enabling AI mimicry  
- X3_PROPRIETARY_DATA: Data moats, competitive advantages protecting against workflow mimicry
- X4_SWITCHING_FRICTION: Network effects, barriers preventing workflow replication by AI
- X5_REGULATORY_BARRIERS: Certification requirements, legal protections against workflow mimicry
- X6_AGENT_PROTOCOL: API maturity, AI integration capabilities with workflow systems

### Plan:
Step 1: Map source content to specific Y1-Y6, X1-X6 indicators
Step 2: Extract raw facts, metrics, and data points (NO interpretation or scoring)
Step 3: Preserve quantitative data, statistics, and concrete evidence
Step 4: Maintain exact source attribution and publication details

### Guidelines:
- **Indicator Targeting**: Map content to specific indicators, not general Y/X categories
- **Raw Data Focus**: Extract facts, numbers, quotes - avoid analysis or interpretation
- **Quantitative Priority**: Include ALL numbers, stats, percentages, and measurable data
- **Citation Preservation**: Maintain exact source attribution and publication details
- **Temporal Relevance**: Flag publication dates, especially 2023-2025 sources
- **Token Efficiency**: Summarize <500 tokens while preserving critical indicator evidence

### Source Text:
{data}

### Output Requirements:
If the query cannot be answered using the text, YOU MUST summarize the text focusing on indicator-relevant elements.
Include ALL factual information mapped to indicators such as:
- Y1: Task complexity percentages, repetition rates within application workflow
- Y2: Error rates, quality control costs for workflow automation (crucial factor)
- Y3: Expertise requirements, knowledge complexity within workflow tasks
- Y4: Data availability metrics for internal workflow automation
- Y5: Process variation rates, exception handling within application workflow
- Y6: Human dependency ratios, UI interaction requirements within application
- X1: Workflow logic visibility percentages, observable process data to external AI
- X2: Protocol adoption rates, workflow standardization enabling AI mimicry
- X3: Proprietary data advantages protecting against workflow mimicry
- X4: Switching cost data, barriers preventing workflow replication by AI
- X5: Regulatory requirements, legal protections against workflow mimicry
- X6: API maturity levels, AI integration capabilities with workflow systems

Preserve exact citations and quantitative data for reliable downstream indicator analysis.
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
    def validate_evidence_coverage(evidence_list: List[Dict], min_indicators: int = 7) -> Dict[str, Any]:
        """
        Validate evidence coverage for disruption analysis with underspecification detection.
        
        CoT: Step 1 - Count indicators covered instead of scoring evidence quality
        CoT: Step 2 - Validate citation quality and recency for reliability  
        CoT: Step 3 - Conservative approach - escape if insufficient coverage
        
        Args:
            evidence_list: List of evidence items with indicator mapping and citations
            min_indicators: Minimum number of indicators required for analysis
            
        Returns:
            Dict with validation results and error handling
        """
        
        # CoT: Extract unique indicators from evidence rather than scoring
        covered_indicators = set()
        for evidence in evidence_list:
            if indicator := evidence.get("indicator"):
                covered_indicators.add(indicator)
        
        total_coverage = len(covered_indicators)
        
        if total_coverage < min_indicators:
            return {
                "valid": False,
                "error": f"Underspecification: Found {total_coverage} indicators covered, need minimum {min_indicators} for reliable analysis."
            }
        
        # CoT: Check for recent citations (2023-2025) for reliability
        recent_sources = [e for e in evidence_list if any("2023" in str(cite) or "2024" in str(cite) or "2025" in str(cite) for cite in e.get("cites", []))]
        
        if len(recent_sources) < min_indicators * 0.5:  # At least 50% recent sources
            return {
                "valid": False,
                "error": f"Underspecification: Insufficient recent sources (2023-2025). Found {len(recent_sources)} recent, need minimum {min_indicators//2}."
            }
        
        # CoT: Count Y vs X indicator coverage for balance validation
        y_indicators = [i for i in covered_indicators if i.startswith("Y")]
        x_indicators = [i for i in covered_indicators if i.startswith("X")]
        
        if len(y_indicators) < 3 or len(x_indicators) < 3:
            return {
                "valid": False,
                "error": f"Underspecification: Unbalanced indicator coverage. Y indicators: {len(y_indicators)}, X indicators: {len(x_indicators)}, need minimum 3 each."
            }
        
        return {
            "valid": True,
            "total_indicators_covered": total_coverage,
            "y_indicators_covered": len(y_indicators),
            "x_indicators_covered": len(x_indicators),
            "recent_sources": len(recent_sources),
            "total_evidence_items": len(evidence_list)
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
    def calculate_coverage_consensus(variants: List[Dict]) -> Dict[str, Any]:
        """
        Calculate consensus coverage from analysis variants using median voting.
        
        CoT: Step 1 - Extract coverage counts from variants instead of scores
        CoT: Step 2 - Calculate median coverage for reliability assessment
        CoT: Step 3 - Identify consistently covered indicators across variants
        
        Args:
            variants: List of variant analyses with indicator coverage data
            
        Returns:
            Dict with consensus coverage and confidence metrics
        """
        
        # CoT: Extract coverage counts instead of scores
        coverage_counts = [v.get("total_indicators", 0) for v in variants if v.get("total_indicators")]
        
        if not coverage_counts:
            return {"error": "Insufficient variant coverage data for consensus calculation"}
        
        # CoT: Calculate median coverage
        median_coverage = sorted(coverage_counts)[len(coverage_counts)//2]
        
        # CoT: Calculate coverage variance for confidence assessment
        coverage_variance = max(coverage_counts) - min(coverage_counts)
        
        # CoT: Identify consistently covered indicators across variants
        all_indicators = set()
        consistent_indicators = set()
        
        for variant in variants:
            variant_indicators = set(variant.get("covered_indicators", []))
            if not all_indicators:  # First variant
                all_indicators = variant_indicators
                consistent_indicators = variant_indicators
            else:
                all_indicators.update(variant_indicators)
                consistent_indicators.intersection_update(variant_indicators)
        
        # CoT: Determine confidence level based on coverage consistency
        if coverage_variance <= 1:
            confidence = "High"
        elif coverage_variance <= 2:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            "median_coverage": median_coverage,
            "coverage_variance": coverage_variance,
            "confidence_level": confidence,
            "consistently_covered_indicators": list(consistent_indicators),
            "total_unique_indicators": len(all_indicators),
            "requires_additional_research": median_coverage < 7,
            "coverage_adequate": median_coverage >= 7
        }

    @staticmethod
    def get_disruption_workflow_example() -> Dict[str, str]:
        """
        Provide example workflow for complete raw evidence disruption analysis.
        
        CoT: Step 1 - Show indicator-targeted planning instead of generic sub-queries
        CoT: Step 2 - Demonstrate raw evidence extraction without scoring
        CoT: Step 3 - Illustrate JSON output with evidence organization
        CoT: Step 4 - Show coverage validation instead of score consensus
        
        Returns:
            Dict with example usage of optimized prompts for raw evidence analysis
        """
        
        return {
            "step_1_planning": """
# Example: Raw Evidence Planning Phase
query = "Procore construction project management automation"
domain = "construction"

prompt = PromptFamily.generate_disruption_planning_prompt(
    query=query, 
    domain=domain, 
    max_subqueries=5
)

# Expected indicator-targeted sub-queries:
# 1. "Construction task structure repetition within workflow Y1 automation analysis 2023-2025"
# 2. "Procore workflow logic observability X1 external AI visibility core processes 2024"  
# 3. "Construction workflow error risk Y2 automation quality control within application 2024"
# 4. "Construction workflow replication barriers X4 AI mimicry switching friction 2025"
# 5. "Construction workflow protection X5 regulatory barriers AI mimicry prevention 2024"
""",
            
            "step_2_execution": """
# Example: Raw Evidence Extraction Phase  
query = "Procore construction automation indicators"
sources_data = "[scraped content from research]"

prompt = PromptFamily.generate_disruption_execution_prompt(
    query=query,
    sources_data=sources_data,
    max_evidence_items=12
)

# Expected raw evidence format (NO scoring):
# <evidence>
# <indicator>Y1_TASK_STRUCTURE</indicator>
# <raw>Construction project scheduling within Procore involves 73% repetitive tasks according to industry analysis, with standardized workflows in planning, resource allocation, and progress tracking phases within the application</raw>
# <cites>[McKinsey 2024: Construction Automation Report, Bain 2025: Project Management AI Adoption]</cites>
# </evidence>
# 
# <evidence>
# <indicator>X1_EXTERNAL_OBSERVABILITY</indicator>
# <raw>Procore's workflow logic shows 42% external observability to AI systems, with 8 core processes visible to external analysis, including project scheduling, resource management, and progress tracking workflows</raw>
# <cites>[Construction Tech Report 2024, Industry Week 2025: Workflow Analysis]</cites>
# </evidence>
""",
            
            "step_3_publishing": """
# Example: Raw Evidence Publishing Phase
query = "Procore disruption analysis"
evidence_data = "[extracted raw evidence per indicator]"

prompt = PromptFamily.generate_disruption_publishing_prompt(
    query=query,
    evidence_data=evidence_data,
    company_name="Procore"
)

# Expected JSON output format (NO scores/quadrants):
# {
#   "query": "Procore workflow automation and penetration analysis",
#   "company": "Procore",
#   "indicators_evidence": {
#     "Y_automation_potential": {
#       "Y1_task_structure": {
#         "raw_evidence": ["73% repetitive tasks within application", "Standardized workflows within Procore", "..."],
#         "citations": ["McKinsey 2024", "Bain 2025", "..."],
#         "key_metrics": ["73% repetition rate within app", "Average 4.2 complexity score"]
#       }
#     },
#     "X_workflow_penetration": {
#       "X1_external_observability": {
#         "raw_evidence": ["42% workflow logic visibility to external AI", "8 observable core processes", "..."], 
#         "citations": ["Construction Tech Report 2024", "Industry Week 2025"],
#         "key_metrics": ["42% workflow observability", "8 mimicable processes"]
#       }
#     }
#   },
#   "evidence_quality": {
#     "total_indicators_covered": 8,
#     "y_indicators_covered": 4,
#     "x_indicators_covered": 4
#   }
# }
""",
            
            "step_4_validation": """
# Example: Coverage Consensus Validation
base_analysis = "[initial raw evidence analysis]"

variants_prompt = PromptFamily.generate_disruption_variants_prompt(
    base_analysis=base_analysis,
    variant_count=3
)

# Calculate coverage consensus (NOT score consensus)
consensus = PromptFamily.calculate_coverage_consensus([
    {"total_indicators": 7, "covered_indicators": ["Y1", "Y3", "Y4", "X1", "X2", "X5", "X6"]},
    {"total_indicators": 8, "covered_indicators": ["Y1", "Y2", "Y4", "X1", "X2", "X3", "X5", "X6"]}, 
    {"total_indicators": 7, "covered_indicators": ["Y1", "Y4", "Y5", "X1", "X2", "X4", "X5"]}
])

# Result: {"median_coverage": 7, "coverage_adequate": True, "confidence_level": "High"}
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

# CoT: Updated mapping to reflect raw evidence approach instead of scoring/quadrant analysis
report_type_mapping = {
    ReportType.ResearchReport.value: "generate_report_prompt",
    ReportType.ResourceReport.value: "generate_resource_report_prompt",
    ReportType.OutlineReport.value: "generate_outline_report_prompt",
    ReportType.CustomReport.value: "generate_custom_report_prompt",
    ReportType.SubtopicReport.value: "generate_subtopic_report_prompt",
    ReportType.DeepResearch.value: "generate_deep_research_prompt",
    # NOTE: Removed DisruptionAnalysis mapping - use generate_disruption_publishing_prompt directly
    # for raw evidence synthesis without scoring/quadrants
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

################################################################################################
# REFACTORING COMPLETE: Raw Evidence Framework Implementation :-)
#
# CoT Summary: Successfully refactored prompts.py to eliminate scoring/bucketing and focus on 
# raw evidence gathering for 6 Y (automation within workflow) and 6 X (workflow penetration) indicators:
#
# 1. REMOVED scoring logic: No more <y_scores>, <x_scores>, <median>, <quadrant> outputs
# 2. ADDED indicator constants: Y_AUTOMATION_INDICATORS and X_WORKFLOW_PENETRATION_INDICATORS  
# 3. UPDATED planning prompts: Target specific indicators (Y1-Y6, X1-X6) in sub-queries
# 4. REFACTORED execution prompts: Extract raw evidence per indicator with citations  
# 5. REPLACED publishing prompts: JSON output with indicators_evidence structure
# 6. MODIFIED validation: Coverage consensus (>=7 indicators) instead of score consensus
# 7. UPDATED utilities: validate_evidence_coverage() and calculate_coverage_consensus()
#
# Key Changes:
# - Conservative approach: Raw data only, no scoring for accurate downstream analysis
# - Escape conditions: <7 indicators covered or <3 per Y/X axis  
# - Self-consistency: Vote on coverage completeness, not score agreement
# - Evidence format: <evidence><indicator>Y1_TASK_STRUCTURE</indicator><raw>...</raw><cites>...</cites></evidence>
# - JSON outputs: {"indicators_evidence": {"Y_automation_potential": {...}, "X_penetration_potential": {...}}}
# 
# Vision Alignment: Gather raw indicator evidence only (no scoring/bucketing) with trusted sources 
# (min 4 2023-2025), conservative escapes for low coverage, self-consistency on extraction reliability,
# raw JSON outputs for downstream scoring/bucketing systems. Focus on Y-axis (AI automation within workflow) 
# and X-axis (workflow penetration by AI mimicking core logic) :-)
################################################################################################
