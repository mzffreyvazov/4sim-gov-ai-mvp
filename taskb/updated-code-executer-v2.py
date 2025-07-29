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
from typing import Dict, List, Optional
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

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model_name = "gemini-2.0-flash"

app = FastAPI(title="Robust AI Data Visualization Agent v2")


# Helper functions
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
    if "```python" in code_string:
        code_string = code_string.split("```python")[1].strip()
    if "```" in code_string:
        code_string = code_string.split("```")[0].strip()
    
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
    title: str = Field(
        description="The concise, descriptive title from 'Chart Suggestion Title'."
    )
    question: str = Field(
        description="The analytical question from 'The Question it Answers'."
    )
    chart_type: str = Field(
        description="The single chart type from 'Chart Type'."
    )
    column_mapping: Dict[str, Optional[str]] = Field(
        description="A dictionary extracted from 'Column Mapping', e.g., {'X-Axis': 'col_a', 'Y-Axis': 'col_b', 'Grouping/Color': 'col_c'}."
    )
    description: str = Field(
        description="The rationale and insight from 'Rationale and Insight'."
    )

class Suggestions(BaseModel):
    charts: List[ChartSuggestion]

# Core logics & endpoints
@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        df = read_file_with_encoding_detection(file_content, file.filename)

        df_head_str = df.head(20).to_string()
        df_shape_str = str(df.shape)
        df_columns_str = str(df.columns.tolist())
        df_description_str = df.describe(include='all').to_string()
        buf = io.StringIO()
        df.info(buf=buf)
        df_info_str = buf.getvalue()

        summary = {
            "filename": file.filename,
            "rows": len(df),
            "cols": len(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "head": df.head(3).to_dict(orient="records"),
            "head_str": df_head_str,
            "shape": df_shape_str,
            "columns": df_columns_str,
            "description": df_description_str,
            "info": df_info_str
        }

        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.1, api_key=GOOGLE_API_KEY)

        prompt = PromptTemplate(
            input_variables=["schema", "stats", "df_info_str", "df_description_str"],
            template="""
            You are an expert data analyst. Given the following dataset schema and statistics, provide a detailed summary (up to 7 sentences) covering:
            - The structure and main characteristics of the dataset (columns, types, shape, sample rows)
            - Notable patterns, distributions, or anomalies you observe from the head, description, and info
            - Potential focus areas or questions this dataset could help answer
            - 2-3 specific suggestions for further analysis or insights that could be derived

            Schema: {schema}
            Stats: {stats}
            Info: {df_info_str}
            Description: {df_description_str}
            """
        )
        chain = prompt | llm | StrOutputParser()
        narrative = chain.invoke({"schema": list(df.columns), "stats": summary, "df_info_str": df_info_str, "df_description_str": df_description_str})

        return {"summary": summary, "narrative": narrative}
    
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}


@app.post("/generate-dashboard/", response_class=FileResponse)
async def generate_dashboard(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        df = read_file_with_encoding_detection(file_content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, api_key=GOOGLE_API_KEY)

    df_head_str = df.head(20).to_string()
    df_shape_str = str(df.shape)
    df_columns_str = str(df.columns.tolist())
    df_description_str = df.describe(include='all').to_string()
    buf = io.StringIO()
    df.info(buf=buf)
    df_info_str = buf.getvalue()

    
    # Detailed Analysis Prompt
    analysis_template_prompt = """
    You are an expert Data Scientist and a master of data storytelling. Your primary skill is to look at any dataset and instantly identify the most compelling stories that can be told through visualizations. You think critically about the data, considering potential relationships, distributions, comparisons, compositions, and trends over time.

    Your mission is to analyze the provided dataset context and propose 5-7 insightful and diverse visualizations. Your suggestions must be a complete blueprint that another analyst or an automated tool could use to generate the charts directly.

    **Full Dataset Context:**
    - **Shape (Rows, Columns):** {df_shape}
    - **Column Names:** {df_columns}
    - **Schema (dtypes and non-null counts):**
    {df_info}
    - **Statistical Summary (for numerical and categorical columns):**
    {df_description}
    - **Data Sample (first 20 rows):**
    {df_head}

    **Your Analysis Process:**
    1.  **Understand the Data:** First, mentally classify each column as numerical, categorical, temporal, boolean, or geographical based on the provided context.
    2.  **Identify Potential Stories:** Brainstorm analytical questions you could answer by visualizing relationships between these columns.
    3.  **Structure Your Suggestions:** For each suggested visualization, provide a complete and structured description as detailed below.

    **CRITICAL INSTRUCTIONS for Output:**

    For each of the 5-7 chart suggestions, you MUST provide the following details in a clear, structured format. Follow this template precisely for each suggestion:

    ---

    **1. Chart Suggestion Title:** (A concise, descriptive title, e.g., "Distribution of Athlete Age by Medal Status")

    *   **The Question it Answers:** A clear, one-sentence analytical question. (e.g., "What is the age distribution for athletes who won a medal versus those who did not?")
    *   **Chart Type:** The single most appropriate chart type. (e.g., "Box Plot").
    *   **Column Mapping:**
        *   **X-Axis:** [Exact column name for the x-axis, if applicable. e.g., 'Medal_Binary']
        *   **Y-Axis:** [Exact column name for the y-axis, or "Count/Frequency" for histograms. e.g., 'Age']
        *   **Grouping/Color (Optional):** [Exact column name to segment the data, if applicable]
    *   **Rationale and Insight:** A brief explanation (1-2 sentences) of why this chart is valuable and what patterns to look for. (e.g., "This chart will compare the age spread of medal winners and non-winners, potentially revealing a peak performance age range for Olympic success.")

    ---

    **(Repeat the above structure for all your suggestions)**

    **IMPORTANT GUIDELINES:**
    *   **Use Exact Column Names:** You MUST use ONLY the exact column names from the 'Column Names' list provided. Do not alter or invent them.
    *   **Suggest Diverse Charts:** Provide a mix of visualizations that cover different analytical goals.
    """

    analysis_prompt = PromptTemplate.from_template(analysis_template_prompt)
    analysis_chain = analysis_prompt | llm | StrOutputParser()

    print("Step 1: Invoking Analyst Agent to generate text report...")
    try:
        text_report = analysis_chain.invoke({
            "df_shape": df_shape_str, "df_columns": df_columns_str, "df_info": df_info_str,
            "df_description": df_description_str, "df_head": df_head_str
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyst Agent failed. Error: {e}")

    json_parser = JsonOutputParser(pydantic_object=Suggestions)
    
    extraction_prompt = PromptTemplate(
        template="""You are a data extraction expert. Parse the user's text report and convert it into a structured JSON object that conforms to the provided format instructions.
        For each chart suggestion, extract the 'Chart Suggestion Title' into the 'title' field, 'The Question it Answers' into the 'question' field, etc.
        The 'Column Mapping' section should be converted into a dictionary.
        
        **Text Report to Parse**:
        {text_report}

        {format_instructions}""",
        input_variables=["text_report"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()}
    )
    extraction_chain = extraction_prompt | llm | json_parser

    print("Step 2: Invoking Extractor Agent to parse the report into JSON...")
    try:
        suggestions_result = extraction_chain.invoke({"text_report": text_report})
        chart_suggestions = suggestions_result['charts']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extractor Agent failed to parse report into JSON. Error: {e}")


    chart_images = []
    annotations = []
    
    print(f"=== CHART GENERATION SUMMARY ===")
    print(f"Total chart suggestions received: {len(chart_suggestions)}")
    for i, suggestion in enumerate(chart_suggestions, 1):
        print(f"{i}. {suggestion.get('title', 'Untitled')} - Type: {suggestion.get('chart_type', 'Unknown')}")
    print("=====================================")

    code_gen_prompt_template = """You are an elite Python data scientist focused on writing clean, direct, and executable code for data visualization. Your code must run without any modifications.

    **CONTEXT**:
    - A pandas DataFrame named `df` is already loaded and available.
    - The following libraries and variables are pre-imported and available: `pd`, `plt`, `np`, `io`, `img_buffer`.
    - **DataFrame Schema Context**:
    - Columns Available: {df_columns}
    - Data Types & Nulls:
        {df_info}
    - Statistical Summary:
        {df_description}

    **TASK**:
    - **Goal**: Write the Python code to generate a '{chart_type}'.
    - **Title**: The chart should be titled '{title}'.
    - **Insight**: This chart is intended to answer: "{question}"
    - **Column Mapping**: {column_mapping}

    **CODE GENERATION LOGIC & BEST PRACTICES**:
    Follow these logic patterns based on the requested 'Chart Type'.

    - **Bar Chart**:
    - **If Y is a numeric column and X is categorical**: This is an aggregation plot. You MUST first `groupby()` the X-column and calculate an aggregate (like mean, sum, or count) on the Y-column before plotting. Example: `df.groupby('X_COLUMN')['Y_COLUMN'].mean().sort_values().plot(kind='bar', ax=ax)`
    - **If Y is 'Count'/'Frequency' and X is categorical**: This is a count plot. Use `df['X_COLUMN'].value_counts().plot(kind='bar', ax=ax)`.

    - **Histogram**:
    - This visualizes the distribution of a SINGLE numeric column. Use the column specified under 'Data-Column' or 'X-Axis' in the mapping.
    - Before plotting, drop nulls from that column: `data = df_copy['COLUMN_NAME'].dropna()`.
    - Use `ax.hist(data, bins=30)` for plotting.

    - **Scatter Plot**:
    - This shows the relationship between TWO NUMERIC columns (X and Y).
    - Use `ax.scatter(df_copy['X_COLUMN'], df_copy['Y_COLUMN'])`.

    - **Box Plot**:
    - This compares the distribution of a NUMERIC column (Y) across different CATEGORIES (X).
    - Use pandas: `df_copy.boxplot(column='Y_COLUMN', by='X_COLUMN', ax=ax, grid=False)`.

    - **Line Chart**:
    - This is for showing a trend. The X-axis column MUST be sorted before plotting.
    - First, prepare the data, often with `groupby()`. Example: `data = df_copy.groupby('X_COLUMN')['Y_COLUMN'].mean()`.
    - Then plot: `ax.plot(data.index, data.values)`.

    - **Pie Chart**:
    - Shows the composition of a CATEGORICAL column. Use `df['COLUMN_NAME'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')`.

    **MANDATORY CODE STRUCTURE**:
    You must write a single, direct block of executable Python code.

    1.  `df_copy = df.copy()`
    2.  Perform any necessary data cleaning or aggregation on `df_copy` as per the logic above (e.g., `groupby`, `value_counts`, `dropna`).
    3.  `fig, ax = plt.subplots(figsize=(12, 7))`
    4.  The single plotting command.
    5.  `ax.set_title('{title}', fontsize=16)`
    6.  Set appropriate xlabel and ylabel using `ax.set_xlabel` and `ax.set_ylabel`.
    7.  If the x-axis has categorical labels, use `plt.xticks(rotation=45, ha='right')`.
    8.  `plt.grid(axis='y', linestyle='--', alpha=0.7)` for better readability.
    9.  `fig.tight_layout()`
    10. `fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')`
    11. `plt.close(fig)`

    **CRITICAL RULES**:
    - **NO FUNCTIONS, IMPORTS, COMMENTS, or MARKDOWN**.
    - **DIRECT CODE ONLY**: Your entire output must be executable Python.

    Produce the Python code now.
    """


    code_gen_prompt = PromptTemplate.from_template(code_gen_prompt_template)
    code_gen_chain = code_gen_prompt | llm | StrOutputParser()

    successful_charts = 0
    failed_charts = 0
    skipped_charts = 0

    for i, suggestion in enumerate(chart_suggestions, 1):
        print(f"\n=== Processing Chart {i}/{len(chart_suggestions)}: '{suggestion.get('title')}' ===")
        generated_code = ""
        try:
            raw_code = code_gen_chain.invoke({
                "df_info": df_info_str,
                "chart_type": suggestion['chart_type'],
                "title": suggestion['title'],
                "question": suggestion['description'],
                "column_mapping": {
                    "X-Axis": suggestion.get('x_column'),
                    "Y-Axis": suggestion.get('y_column'),
                    "Grouping/Color": suggestion.get('grouping_column', None)
                },
                "df_columns": df.columns.tolist(),
                "df_description": df.describe(include='all').to_string()
            })
            generated_code = clean_generated_code(raw_code)
            print(f"Generated code for '{suggestion.get('title')}':\n---\n{generated_code}\n---")

            if 'def ' in generated_code:
                print(f"❌ Generated code contains function definitions, skipping chart '{suggestion.get('title')}'")
                skipped_charts += 1
                continue

            # Validate column names 
            missing_columns = []
            if suggestion.get('x_column') and suggestion.get('x_column') not in df.columns:
                missing_columns.append(suggestion.get('x_column'))
            if suggestion.get('y_column') and suggestion.get('y_column') not in df.columns:
                missing_columns.append(suggestion.get('y_column'))
            
            if missing_columns:
                print(f"❌ Missing columns {missing_columns} for chart '{suggestion.get('title')}', skipping")
                print(f"Available columns: {list(df.columns)}")
                skipped_charts += 1
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
                successful_charts += 1
                print(f"✅ Successfully generated chart '{suggestion.get('title')}' ({buffer_size} bytes)")
            else:
                failed_charts += 1
                print(f"❌ Code for chart '{suggestion.get('title')}' executed but produced an empty image ({buffer_size} bytes). Skipping.")
                print(f"Generated Code:\n---\n{generated_code}\n---")

        except Exception as e:
            failed_charts += 1
            print("--- FAILED CODE EXECUTION ---")
            print(f"Chart Title: {suggestion.get('title', 'Untitled')}")
            print(f"Error Type: {type(e).__name__}, Error: {e}")
            print(f"Generated Code That Failed:\n---\n{generated_code}\n---")
            continue
    
    # Print final summary
    print(f"\n=== FINAL CHART GENERATION SUMMARY ===")
    print(f"Total charts requested: 5")
    print(f"✅ Successful charts: {successful_charts}")
    print(f"❌ Failed charts: {failed_charts}")
    print(f"⏭️  Skipped charts: {skipped_charts}")
    print(f"📊 Charts in final PDF: {len(chart_images)}")
    print("=====================================")
    
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