from typing import List, Dict, Optional
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from .models import ChartSuggestion, Suggestions, PDFAnalysisReport
from .prompts import ANALYSIS_TEMPLATE, CODE_GEN_TEMPLATE, EXTRACTION_TEMPLATE, QUERY_PROCESSING_TEMPLATE, CSV_FORMATTING_TEMPLATE

class DataAnalyst:
    """Analyzes datasets and generates chart suggestions"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )
        
    def analyze_dataset(self, df_context: Dict[str, str], num_suggestions: int = 5) -> str:
        """Generate textual chart suggestions from dataset context"""
        analysis_prompt = PromptTemplate.from_template(ANALYSIS_TEMPLATE)
        analysis_chain = analysis_prompt | self.llm | StrOutputParser()
        
        # Add num_suggestions to the context
        context_with_suggestions = {**df_context, "num_suggestions": num_suggestions}
        return analysis_chain.invoke(context_with_suggestions)

class SuggestionExtractor:
    """Extracts structured data from textual analysis"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )
        
    def extract_suggestions(self, text_report: str, df_columns: List[str], num_suggestions: int = 5) -> List[ChartSuggestion]:
        """Extract structured chart suggestions from text analysis"""
        json_parser = JsonOutputParser(pydantic_object=Suggestions)
        extraction_prompt = PromptTemplate.from_template(EXTRACTION_TEMPLATE)
        extraction_chain = extraction_prompt | self.llm | json_parser
        
        result = extraction_chain.invoke({
            "text_report": text_report,
            "df_columns": str(df_columns),
            "num_suggestions": num_suggestions,
            "format_instructions": json_parser.get_format_instructions()
        })
        
        # Convert dictionaries to ChartSuggestion objects if needed
        suggestions = []
        if isinstance(result, dict) and 'charts' in result:
            charts_data = result['charts']
        else:
            charts_data = result
            
        for chart_data in charts_data:
            if isinstance(chart_data, dict):
                # Convert dict to ChartSuggestion object
                suggestions.append(ChartSuggestion(**chart_data))
            elif isinstance(chart_data, ChartSuggestion):
                suggestions.append(chart_data)
            else:
                # Handle unexpected format
                print(f"Warning: Unexpected chart data format: {type(chart_data)}")
                
        return suggestions

class ChartQueryProcessor:
    """Processes natural language chart queries and converts them to ChartSuggestion format"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )
    
    def process_query(self, user_query: str, df_context: Dict[str, str]) -> ChartSuggestion:
        """Convert natural language query to structured chart suggestion"""
        query_prompt = PromptTemplate.from_template(QUERY_PROCESSING_TEMPLATE)
        json_parser = JsonOutputParser(pydantic_object=ChartSuggestion)
        query_chain = query_prompt | self.llm | json_parser
        
        result = query_chain.invoke({
            "user_query": user_query,
            "df_columns": df_context["df_columns"],
            "df_info": df_context["df_info"],
            "df_description": df_context["df_description"],
            "format_instructions": json_parser.get_format_instructions()
        })
        
        # Convert to ChartSuggestion object if it's a dict
        if isinstance(result, dict):
            return ChartSuggestion(**result)
        return result

class ChartCodeGenerator:
    """Generates executable Python code for charts"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )
        
    def generate_chart_code(self, suggestion: ChartSuggestion, df_context: Dict[str, str]) -> str:
        """Generate Python code for a specific chart suggestion"""
        code_gen_prompt = PromptTemplate.from_template(CODE_GEN_TEMPLATE)
        code_gen_chain = code_gen_prompt | self.llm | StrOutputParser()
        
        return code_gen_chain.invoke({
            "df_info": df_context["df_info"],
            "df_description": df_context["df_description"],
            "df_columns": df_context["df_columns"],
            "chart_type": suggestion.chart_type,
            "title": suggestion.title,
            "question": suggestion.question,
            "column_mapping": suggestion.column_mapping,
            "pre_processing_steps": suggestion.pre_processing_steps
        })

    def clean_generated_code(self, code_string: str) -> str:
        """Strips markdown and import statements from the LLM's generated code."""
        if "```python" in code_string:
            code_string = code_string.split("```python")[1].strip()
        if "```" in code_string:
            code_string = code_string.split("```")[0].strip()
        
        lines = [line for line in code_string.split('\n') 
                if not (line.strip().startswith('import ') or line.strip().startswith('from '))]
        return '\n'.join(lines).strip()

class CSVFormatter:
    """Formats and cleans unstructured CSV data from government sources and similar complex datasets"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0.0, 
            api_key=api_key
        )
        
    def format_csv_data(self, raw_csv_data: str) -> str:
        """
        Process raw CSV data and return formatted, analysis-ready CSV
        
        Args:
            raw_csv_data: Raw CSV content as string
            
        Returns:
            Formatted CSV data as string
        """
        formatting_prompt = PromptTemplate.from_template(CSV_FORMATTING_TEMPLATE)
        formatting_chain = formatting_prompt | self.llm | StrOutputParser()
        
        formatted_csv = formatting_chain.invoke({
            "raw_csv_data": raw_csv_data
        })
        
        return self._clean_csv_output(formatted_csv)
    
    def _clean_csv_output(self, csv_output: str) -> str:
        """Clean the LLM output to ensure it's valid CSV"""
        # Remove any markdown formatting
        if "```csv" in csv_output:
            csv_output = csv_output.split("```csv")[1].strip()
        if "```" in csv_output:
            csv_output = csv_output.split("```")[0].strip()
        
        # Remove any explanatory text that might be before or after CSV
        lines = csv_output.strip().split('\n')
        
        # Find the start of actual CSV data (first line with commas)
        start_idx = 0
        for i, line in enumerate(lines):
            if ',' in line and not line.strip().startswith('#'):
                start_idx = i
                break
        
        # Find the end of CSV data (last line with commas)
        end_idx = len(lines) - 1
        for i in range(len(lines) - 1, -1, -1):
            if ',' in lines[i] and not lines[i].strip().startswith('#'):
                end_idx = i
                break
        
        # Extract only the CSV portion
        csv_lines = lines[start_idx:end_idx + 1]
        
        return '\n'.join(csv_lines).strip()
