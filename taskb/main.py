import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server environments
import matplotlib.pyplot as plt
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import chardet

# Load env vars
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = "gemini-2.0-flash"

app = FastAPI(title="Task B: Data Viz & Analytics")

def setup_unicode_fonts():
    """
    Registers a Unicode-capable font for ReportLab, falling back to Helvetica.
    """
    try:
        font_paths = [
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\Arial.ttf",
            "/System/Library/Fonts/Arial.ttf",
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
                return 'DejaVuSans'
        return 'Helvetica'
    except Exception:
        return 'Helvetica'

def read_file_with_encoding_detection(file_content: bytes, filename: str) -> pd.DataFrame:
    """
    Reads a file (CSV or Excel) into a pandas DataFrame with auto-detected encoding.
    """
    if filename.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_content))
    else:
        result = chardet.detect(file_content)
        encoding = result['encoding'] or 'utf-8'
        try:
            return pd.read_csv(io.StringIO(file_content.decode(encoding)))
        except (UnicodeDecodeError, pd.errors.ParserError):
            # Fallback to latin-1 if primary encoding fails
            return pd.read_csv(io.StringIO(file_content.decode('latin-1', errors='replace')))

# 1) Ingest & Summarize
@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        df = read_file_with_encoding_detection(file_content, file.filename)
        
        summary = {
            "filename": file.filename,
            "rows": len(df),
            "cols": len(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "head": df.head(3).to_dict(orient="records")
        }
        
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.1, api_key=GOOGLE_API_KEY)
        
        prompt = PromptTemplate(
            input_variables=["schema", "stats"],
            template="""You are a data analyst. Given this schema: {schema}
            and these basic stats: {stats}
            Write a concise, 3-sentence summary of the dataset's structure and potential focus.
            Also suggest 2-3 potential areas of analysis or insights that could be derived from this data.
            """
        )
        chain = prompt | llm | StrOutputParser()
        narrative = chain.invoke({"schema": list(df.columns), "stats": summary})
        
        return {"summary": summary, "narrative": narrative}
    
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

# 2) Chart suggestion & generation
@app.post("/charts/")
async def generate_charts(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        df = read_file_with_encoding_detection(file_content, file.filename)
        
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.2, api_key=GOOGLE_API_KEY)

        # --- KEY CHANGE 1: Define a new, context-aware prompt for annotations ---
        ann_prompt = PromptTemplate(
            input_variables=["chart_type", "chart_data"],
            template="""You are a helpful data analyst. A {chart_type} chart has been generated.
            Here is the data used for the chart:
            ---
            {chart_data}
            ---
            Based *only* on this data, write a 2-3 sentence summary describing the key insight a user should notice from the chart. Focus on the main patterns or distributions revealed by the numbers."""
        )
        ann_chain = ann_prompt | llm | StrOutputParser()

        chart_images = []
        annotations = []
        charts_to_generate = ["bar", "line", "pie"]

        for chart_type in charts_to_generate:
            fig, ax = plt.subplots(figsize=(8, 5))
            chart_data_summary = "No data available." # Default summary

            try:
                # --- KEY CHANGE 2: Generate chart and extract its data for context ---
                if chart_type == "bar" and not df.empty:
                    col_name = df.columns[0]
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        # For numeric data, create a histogram
                        counts, bins, _ = ax.hist(df[col_name].dropna(), bins=15)
                        ax.set_title(f"Histogram of {col_name}")
                        chart_data_summary = f"The chart shows a histogram for the numeric column '{col_name}'. Bin edges are roughly { [round(b, 2) for b in bins] } with corresponding counts { [int(c) for c in counts] }."
                    else:
                        # For categorical data, create a bar chart
                        value_counts = df[col_name].value_counts().head(10)
                        value_counts.plot.bar(ax=ax, rot=45)
                        ax.set_title(f"Top 10 Distribution for {col_name}")
                        chart_data_summary = f"The chart shows top 10 value counts for the categorical column '{col_name}'. Data: {value_counts.to_dict()}"

                elif chart_type == "line":
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) >= 2:
                        x_col, y_col = numeric_cols[0], numeric_cols[1]
                        plot_df = df.head(50) # Plot a sample
                        plot_df.plot.line(x=x_col, y=y_col, ax=ax, grid=True)
                        ax.set_title(f"{y_col} over {x_col}")
                        chart_data_summary = f"The chart shows the trend of '{y_col}' against '{x_col}'. The first few data points are: {plot_df[[x_col, y_col]].to_dict(orient='records')}"
                    else:
                        raise ValueError("Not enough numeric columns for a line chart.")

                elif chart_type == "pie":
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    if not categorical_cols.empty:
                        col_name = categorical_cols[0]
                        value_counts = df[col_name].value_counts().head(6)
                        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
                        ax.set_title(f"Top 6 Distribution in {col_name}")
                        ax.set_ylabel('') # Remove y-axis label for pie charts
                        chart_data_summary = f"The chart shows a percentage distribution for the top 6 categories of '{col_name}'. Data: {value_counts.to_dict()}"
                    else:
                        raise ValueError("No suitable categorical column found for a pie chart.")

                else:
                    plt.close(fig)
                    continue
                
                # --- If chart creation was successful, save it and generate annotation ---
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150)
                buf.seek(0)
                chart_images.append(buf) # Keep buffer in memory
                plt.close(fig)

                # --- KEY CHANGE 3: Invoke the LLM with the specific data summary ---
                annotation = ann_chain.invoke({"chart_type": chart_type, "chart_data": chart_data_summary})
                annotations.append(annotation)

            except Exception as e:
                plt.close(fig) # Ensure plot is closed on error
                print(f"Skipping chart '{chart_type}': {e}") # Log error for debugging

        # C) Assemble PDF using ReportLab with Unicode support
        output_filename = "dashboard.pdf"
        unicode_font = setup_unicode_fonts()
        
        doc = SimpleDocTemplate(output_filename, pagesize=A4)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['h1'], fontName=unicode_font, alignment=1)
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontName=unicode_font, spaceAfter=12)
        
        story = [Paragraph("Data Analysis Dashboard", title_style), Spacer(1, 0.25 * inch)]
        
        # --- IMPROVEMENT: Zip the successful charts and annotations to build the story ---
        for img_buffer, text in zip(chart_images, annotations):
            # Pass the in-memory buffer directly to the Image object
            img = Image(img_buffer, width=6*inch, height=3.75*inch, kind='proportional')
            story.append(img)
            story.append(Spacer(1, 0.1 * inch))
            story.append(Paragraph(text.replace("\n", "<br/>"), normal_style))
            story.append(Spacer(1, 0.25 * inch))

        if not annotations:
            story.append(Paragraph("Could not generate any charts from the provided data.", normal_style))
        
        doc.build(story)

        return FileResponse(output_filename, media_type="application/pdf", filename="dashboard.pdf")
    
    except Exception as e:
        return {"error": f"Failed to generate dashboard: {str(e)}"}