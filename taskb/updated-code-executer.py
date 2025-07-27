import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# Use a non-interactive backend for Matplotlib in a server environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import chardet

# --- Environment and Model Setup ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model_name = "gemini-2.0-flash"

app = FastAPI(title="Robust AI Data Visualization Agent v2")

# --- Helper Functions and Classes ---

def setup_unicode_fonts():
    font_paths = ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:\\Windows\\Fonts\\Arial.ttf"]
    for font_path in font_paths:
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
            return 'DejaVuSans'
    return 'Helvetica'

def read_file_with_encoding_detection(file_content: bytes, filename: str) -> pd.DataFrame:
    try:
        if filename.endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(file_content))
        else:
            encoding = chardet.detect(file_content)['encoding'] or 'utf-8'
            return pd.read_csv(io.StringIO(file_content.decode(encoding, errors='ignore')))
    except Exception as e:
        raise ValueError(f"Could not parse the file. Error: {e}")

def clean_generated_code(code_string: str) -> str:
    # Remove markdown code blocks
    if "```python" in code_string:
        code_string = code_string.split("```python")[1].strip()
    if "```" in code_string:
        code_string = code_string.split("```")[0].strip()
    
    # Remove import statements since they're already in scope
    lines = code_string.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not (stripped_line.startswith('import ') or 
                stripped_line.startswith('from ') or
                stripped_line == ''):
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

class ChartSuggestion(BaseModel):
    chart_type: str
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    title: str
    description: str

class Suggestions(BaseModel):
    charts: List[ChartSuggestion]

# --- Core Application Logic ---

@app.post("/generate-dashboard/", response_class=FileResponse)
async def generate_dashboard(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        df = read_file_with_encoding_detection(file_content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, api_key=GOOGLE_API_KEY)

    df_head_str = df.head(20).to_string()
    buf = io.StringIO()
    df.info(buf=buf)
    df_info_str = buf.getvalue()

    parser = JsonOutputParser(pydantic_object=Suggestions)

    analysis_prompt = PromptTemplate(
        template="""You are a meticulous data analyst. Analyze the schema and data sample below and suggest up to 5 insightful charts.
        CRITICAL: Use ONLY the exact column names from the 'Columns' list in the schema.
        Schema: {df_info}
        Data Sample: {df_head}
        {format_instructions}""",
        input_variables=["df_info", "df_head"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    analysis_chain = analysis_prompt | llm | parser

    try:
        suggestions_result = analysis_chain.invoke({"df_info": df_info_str, "df_head": df_head_str})
        chart_suggestions = suggestions_result['charts']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed to generate valid chart suggestions. Error: {e}")

    chart_images = []
    annotations = []

    # --- FINAL, CORRECTED PROMPT for the Coder Agent ---
    code_gen_prompt = PromptTemplate.from_template(
        """You are an expert Python data scientist. Write ONLY direct executable code (NO FUNCTIONS).
        
        **CONTEXT**:
        - DataFrame `df` is available
        - Variables available: df, pd, plt, img_buffer, io, np
        - Data Schema: {df_info}
        
        **TASK**:
        Create a '{chart_type}' titled '{title}' using columns:
        - X-Column: '{x_column}'
        - Y-Column: '{y_column}'
        
        **MANDATORY STRUCTURE** (execute these steps directly, NO function definitions):
        
        1. Copy dataframe: `df_copy = df.copy()`
        2. Clean data if needed (for numeric operations):
           - `df_copy['column'] = pd.to_numeric(df_copy['column'], errors='coerce')`
           - `df_copy.dropna(subset=['column'], inplace=True)`
        3. Validate data exists after cleaning (must have at least 1 row)
        4. Create figure: `fig, ax = plt.subplots(figsize=(10, 6))`
        5. Plot based on chart type:
           - Bar: `df_copy['column'].value_counts().plot(kind='bar', ax=ax)`
           - Histogram: `ax.hist(df_copy['column'], bins=20)`
           - Scatter: `ax.scatter(df_copy['x_col'], df_copy['y_col'])`
           - Pie: `df_copy['column'].value_counts().plot(kind='pie', ax=ax)`
        6. Set title: `ax.set_title('{title}')`
        7. Set axis labels if applicable
        8. Save to buffer: `fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')`
        9. Close: `plt.close(fig)`
        
        **CRITICAL**: 
        - NO function definitions (def)
        - NO imports
        - NO comments
        - DIRECT executable statements only
        - MUST save to img_buffer
        - Handle empty data gracefully
        
        **EXAMPLE WITH DATA VALIDATION**:
        df_copy = df.copy()
        if len(df_copy) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            df_copy['Company'].value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Company Distribution')
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        """
    )
    code_gen_chain = code_gen_prompt | llm | StrOutputParser()

    for suggestion in chart_suggestions:
        # --- FIX: Initialize generated_code to prevent UnboundLocalError ---
        generated_code = ""
        try:
            raw_code = code_gen_chain.invoke({
                "df_info": df_info_str,
                "chart_type": suggestion['chart_type'],
                "title": suggestion['title'],
                "x_column": suggestion.get('x_column'),
                "y_column": suggestion.get('y_column')
            })
            generated_code = clean_generated_code(raw_code)
            print(f"Generated code for '{suggestion.get('title')}':\n---\n{generated_code}\n---")

            # Validate that the code doesn't contain function definitions
            if 'def ' in generated_code:
                print(f"❌ Generated code contains function definitions, skipping chart '{suggestion.get('title')}'")
                continue

            # Validate column names exist
            missing_columns = []
            if suggestion.get('x_column') and suggestion.get('x_column') not in df.columns:
                missing_columns.append(suggestion.get('x_column'))
            if suggestion.get('y_column') and suggestion.get('y_column') not in df.columns:
                missing_columns.append(suggestion.get('y_column'))
            
            if missing_columns:
                print(f"❌ Missing columns {missing_columns} for chart '{suggestion.get('title')}', skipping")
                print(f"Available columns: {list(df.columns)}")
                continue

            img_buffer = io.BytesIO()
            df_copy = df.copy() 
            exec_scope = {'df': df_copy, 'pd': pd, 'plt': plt, 'img_buffer': img_buffer, 'io': io, 'np': np}
            
            # Add debugging information
            print(f"DataFrame shape before execution: {df_copy.shape}")
            if suggestion.get('x_column'):
                print(f"X-column '{suggestion.get('x_column')}' exists: {suggestion.get('x_column') in df_copy.columns}")
            if suggestion.get('y_column'):
                print(f"Y-column '{suggestion.get('y_column')}' exists: {suggestion.get('y_column') in df_copy.columns}")
            
            # Execute the generated code
            exec(generated_code, exec_scope)
            
            # Add post-execution debugging
            df_after = exec_scope.get('df_copy', df_copy)
            print(f"DataFrame shape after execution: {df_after.shape}")
            
            # Reset buffer position to beginning after execution
            img_buffer.seek(0)
            buffer_content = img_buffer.read()
            buffer_size = len(buffer_content)
            
            # Reset buffer again for later use
            img_buffer.seek(0)
            
            print(f"Buffer size after execution: {buffer_size} bytes")
            
            if buffer_size > 100:
                chart_images.append(img_buffer)
                annotations.append(suggestion['description'])
                print(f"Successfully generated chart '{suggestion.get('title')}' ({buffer_size} bytes)")
            else:
                print(f"Code for chart '{suggestion.get('title')}' executed but produced an empty image ({buffer_size} bytes). Skipping.")
                print(f"Generated Code:\n---\n{generated_code}\n---")

        except Exception as e:
            print("--- FAILED CODE EXECUTION ---")
            print(f"Chart Title: {suggestion.get('title', 'Untitled')}")
            print(f"Error Type: {type(e).__name__}, Error: {e}")
            print(f"Generated Code That Failed:\n---\n{generated_code}\n---")
            continue
    
    if not chart_images:
        raise HTTPException(status_code=500, detail="No charts could be generated successfully. Check server logs for details.")

    output_filename = "dynamic_dashboard.pdf"
    unicode_font = setup_unicode_fonts()
    
    doc = SimpleDocTemplate(output_filename, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['h1'], fontName=unicode_font, alignment=1)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontName=unicode_font, spaceAfter=12, leading=14)
    
    story = [Paragraph("AI-Generated Data Dashboard", title_style), Spacer(1, 0.25 * inch)]
    
    for img_buffer, text in zip(chart_images, annotations):
        img = Image(img_buffer, width=6*inch, height=3.75*inch, kind='proportional')
        story.append(img)
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(text, normal_style))
        story.append(Spacer(1, 0.25 * inch))

    doc.build(story)
    
    return FileResponse(output_filename, media_type="application/pdf", filename=output_filename)