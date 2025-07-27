import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import matplotlib.pyplot as plt
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
import chardet

# Load env vars
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = "gemini-2.0-flash"  # Updated model version

app = FastAPI(title="Task B: Data Viz & Analytics")

def setup_unicode_fonts():
    """
    Setup Unicode-capable fonts for ReportLab
    """
    try:
        # Try to register DejaVu Sans font for better Unicode support
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase import pdfmetrics
        
        # Try different font paths that might exist on the system
        font_paths = [
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        
        for font_path in font_paths:
            try:
                pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
                return 'DejaVuSans'
            except:
                continue
        
        # If no custom font found, return default font
        return 'Helvetica'
    except:
        # If anything fails, return default font
        return 'Helvetica'
        
def read_file_with_encoding_detection(file_content, filename):
    """
    Read file with automatic encoding detection
    """
    if filename.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_content))
    else:
        # Detect encoding for CSV files
        detected = chardet.detect(file_content)
        encoding = detected['encoding'] if detected['encoding'] else 'utf-8'
        
        # Try multiple encodings in order of preference
        encodings_to_try = [encoding, 'utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for enc in encodings_to_try:
            try:
                # Decode bytes to string first
                text_content = file_content.decode(enc)
                # Then read with pandas
                return pd.read_csv(io.StringIO(text_content))
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # If all encodings fail, try with error handling
        try:
            text_content = file_content.decode('utf-8', errors='replace')
            return pd.read_csv(io.StringIO(text_content))
        except Exception as e:
            raise ValueError(f"Could not read file with any encoding. Error: {str(e)}")

# 1) Ingest & Summarize
@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    try:
        # Read file content as bytes
        file_content = await file.read()
        
        # Use the encoding detection function
        df = read_file_with_encoding_detection(file_content, file.filename)
        
        summary = {
            "rows": len(df), "cols": len(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "head": df.head(3).to_dict(orient="records")
        }
        
        # Initialize Gemini with API key
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            api_key=GOOGLE_API_KEY
        )
        
        # Create modern chain for narrative summary
        prompt = PromptTemplate(
            input_variables=["schema", "stats"],
            template="""You are a data analyst. Given this schema: {schema}
            and these basic stats: {stats}
            Write a 3‑sentence summary."""
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
        # Read file content as bytes
        file_content = await file.read()
        
        # Use the encoding detection function
        df = read_file_with_encoding_detection(file_content, file.filename)
        
        # Initialize Gemini with API key
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0,
            api_key=GOOGLE_API_KEY
        )

        # A) Suggest chart types
        suggest_prompt = PromptTemplate(
          input_variables=["columns","summary"],
          template="""Schema: {columns}
          Summary: {summary}
          Suggest the top 3 chart types (bar, line, pie, KPI, etc.) to visualize this data. Explain each choice."""
        )
        suggest_chain = suggest_prompt | llm | StrOutputParser()
        suggestions = suggest_chain.invoke({"columns": list(df.columns), "summary": f"{len(df)} rows"})

        # B) Generate each chart + annotation
        chart_images = []
        annotations = []
        for i, chart_type in enumerate(["bar","line","pie"]):  # stub: parse from suggestions
            fig, ax = plt.subplots(figsize=(8, 6))
            
            try:
                if chart_type == "bar" and len(df.columns) > 0:
                    # Create bar chart from first column
                    if df.iloc[:, 0].dtype == 'object':
                        df.iloc[:, 0].value_counts().head(10).plot.bar(ax=ax)
                        ax.set_title(f"Distribution of {df.columns[0]}")
                    else:
                        df.iloc[:, 0].hist(bins=10, ax=ax)
                        ax.set_title(f"Histogram of {df.columns[0]}")
                elif chart_type == "line" and len(df.columns) > 1:
                    # Create line chart if we have numeric data
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) >= 2:
                        df[numeric_cols[:2]].plot.line(ax=ax)
                        ax.set_title(f"Line Chart: {numeric_cols[0]} vs {numeric_cols[1]}")
                    else:
                        df.iloc[:, 0].plot.line(ax=ax)
                        ax.set_title(f"Line Chart: {df.columns[0]}")
                elif chart_type == "pie" and len(df.columns) > 0:
                    # Create pie chart from first column
                    if df.iloc[:, 0].dtype == 'object':
                        value_counts = df.iloc[:, 0].value_counts().head(5)
                        ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
                        ax.set_title(f"Pie Chart: {df.columns[0]}")
                else:
                    # Fallback: simple bar chart
                    if len(df.columns) > 0:
                        df.iloc[:, 0].value_counts().head(5).plot.bar(ax=ax)
                        ax.set_title(f"Data Overview: {df.columns[0]}")
            except Exception as e:
                # Fallback for any plotting errors
                ax.text(0.5, 0.5, f"Chart generation failed\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Chart Error: {chart_type}")
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
            buf.seek(0)
            chart_images.append(buf.getvalue())
            plt.close(fig)

            ann_prompt = PromptTemplate(
                input_variables=["chart_type","insight"],
                template="""Write 2‑3 sentences in English describing this {chart_type} showing {insight}."""
            )
            ann_chain = ann_prompt | llm | StrOutputParser()
            eng = ann_chain.invoke({"chart_type": chart_type, "insight": "key patterns"})
            
            # translate to Azerbaijani
            trans = llm.invoke(f"Translate into:\n\n{eng}")
            # Extract content from response
            trans_text = trans.content if hasattr(trans, 'content') else str(trans)
            annotations.append(eng)

        # C) Assemble PDF using ReportLab with Unicode support
        output_filename = "dashboard.pdf"
        
        # Setup Unicode fonts
        unicode_font = setup_unicode_fonts()
        
        # Create document with UTF-8 support
        doc = SimpleDocTemplate(
            output_filename, 
            pagesize=A4,
            encoding='utf-8'
        )
        
        # Get styles and create Unicode-compatible styles
        styles = getSampleStyleSheet()
        
        # Create custom styles with Unicode font
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontName=unicode_font,
            encoding='utf-8'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName=unicode_font,
            encoding='utf-8',
            wordWrap='CJK'  # Better wrapping for international text
        )
        
        story = []
        temp_files = []  # Keep track of temp files
        
        # Add title
        title = Paragraph("Data Analysis Dashboard", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Add charts and annotations
        for i, (img_data, text) in enumerate(zip(chart_images, annotations)):
            # Create absolute path for temporary file
            temp_img_path = os.path.abspath(f"temp_chart_{i}.png")
            temp_files.append(temp_img_path)
            
            # Save chart image to temporary file
            with open(temp_img_path, "wb") as f:
                f.write(img_data)
            
            # Create image from BytesIO instead of file path
            img_buffer = io.BytesIO(img_data)
            img = Image(img_buffer, width=5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 10))
            
            # Add annotation text with proper Unicode handling
            try:
                # Ensure text is properly encoded
                if isinstance(text, bytes):
                    text = text.decode('utf-8', errors='replace')
                
                # Clean text for ReportLab
                cleaned_text = text.replace('\x00', '').strip()
                
                annotation = Paragraph(cleaned_text, normal_style)
                story.append(annotation)
            except Exception as e:
                # Fallback with basic text
                fallback_text = f"Chart {i+1} annotation (encoding issue: {str(e)})"
                annotation = Paragraph(fallback_text, normal_style)
                story.append(annotation)
            
            story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        
        # Clean up temporary files after PDF is built
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass  # Ignore cleanup errors

        # Return the PDF file
        return FileResponse(output_filename, media_type="application/pdf", filename="dashboard.pdf")
    
    except Exception as e:
        return {"error": f"Failed to generate charts: {str(e)}"}        