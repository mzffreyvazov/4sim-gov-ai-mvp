# task_d/main.py

import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pptx import Presentation

# Load env vars
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

app = FastAPI(title="Task D: Slide Generation")

def extract_json_from_response(text: str) -> str:
    """Extract JSON content from markdown code blocks or plain text."""
    text = text.strip()
    
    # Check if it's wrapped in markdown code blocks
    if text.startswith("```json") and text.endswith("```"):
        # Extract content between ```json and ```
        lines = text.split('\n')
        json_lines = []
        in_json = False
        for line in lines:
            if line.strip() == "```json":
                in_json = True
                continue
            elif line.strip() == "```" and in_json:
                break
            elif in_json:
                json_lines.append(line)
        return '\n'.join(json_lines)
    elif text.startswith("```") and text.endswith("```"):
        # Extract content between ``` and ```
        return text[3:-3].strip()
    else:
        return text

@app.post("/parse_doc/")
async def parse_doc(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    data = await file.read()
    doc = fitz.open(stream=data, filetype=ext)
    text = ""
    images = []
    for page in doc:
        text += page.get_text()
        for img_meta in page.get_images(full=True):
            xref = img_meta[0]
            pix = fitz.Pixmap(doc, xref)
            buf = pix.tobytes("png")
            images.append(buf)
    return {"text": text, "image_count": len(images)}


@app.post("/make_slides/")
async def make_slides(
    doc: UploadFile = File(...),
    prompt: str = Form(...),
    slide_count: int = Form(...)
):
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # A) Parse the document
    parsed = await parse_doc(doc)
    doc_text = parsed["text"]

    # B) Initialize Gemini with API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,
        api_key=GOOGLE_API_KEY
    )

    # C) Analyze user prompt + doc
    analysis_prompt = PromptTemplate(
        input_variables=["prompt", "text"],
        template=(
            "User instruction: {prompt}\n\n"
            "Document content:\n{text}\n\n"
            "Return ONLY a JSON object with:\n"
            "  num_slides: integer,\n"
            "  sections: array of key topics.\n"
        )
    )
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    analysis_resp = analysis_chain.invoke({
        "prompt": prompt,
        "text": doc_text
    })
    # Extract the content from the response
    logger.info(f"Analysis response type: {type(analysis_resp)}")
    logger.info(f"Analysis response: {analysis_resp}")
    
    # With the modern chain, analysis_resp is already a string
    analysis_text = analysis_resp
    
    logger.info(f"Extracted analysis text: {analysis_text}")
    # Clean and parse the JSON content
    clean_json = extract_json_from_response(analysis_text)
    logger.info(f"Cleaned JSON: {clean_json}")
    details = json.loads(clean_json)

    # D) Generate outline
    outline_prompt = PromptTemplate(
        input_variables=["sections", "num_slides"],
        template=(
            "Given sections {sections} and {num_slides} slides,\n"
            "output a JSON array where each entry has:\n"
            "  title: string,\n"
            "  bullets: array of short bullet points.\n"
        )
    )
    outline_chain = outline_prompt | llm | StrOutputParser()
    outline_resp = outline_chain.invoke({
        "sections": details["sections"],
        "num_slides": str(details["num_slides"])
    })
    # Extract the content from the response
    outline_text = outline_resp
    # Clean and parse the JSON content
    clean_outline_json = extract_json_from_response(outline_text)
    slides_outline = json.loads(clean_outline_json)

    # E) Flesh out each slide + translate
    generated_slides = []
    for slide_def in slides_outline:
        content_prompt = PromptTemplate(
            input_variables=["title", "bullets"],
            template=(
                "Write a concise 2–3 sentence slide text for a slide titled "
                "'{title}' with bullet points {bullets}."
            )
        )
        content_chain = content_prompt | llm | StrOutputParser()
        eng_resp = content_chain.invoke({
            "title": slide_def["title"],
            "bullets": slide_def["bullets"]
        })
        # Extract the content from the response
        eng_text = eng_resp

        trans_resp = llm.invoke(f"Translate the following into Azerbaijani:\n\n{eng_text}")
        # Extract the content from the response
        az_text = trans_resp
        if hasattr(trans_resp, "content"):
            az_text = trans_resp.content
        elif isinstance(trans_resp, dict) and "text" in trans_resp:
            az_text = trans_resp["text"]
        
        # Ensure we have a string
        if not isinstance(az_text, str):
            az_text = str(az_text)
        
        az_text = az_text.strip()

        generated_slides.append({
            "title": slide_def["title"],
            "paragraphs": [p.strip() for p in az_text.split("\n") if p.strip()]
        })

    # F) Assemble PPTX
    # Create a new presentation with default template
    prs = Presentation()
    
    # Add a title slide
    title_slide_layout = prs.slide_layouts[0]  # Title slide layout
    title_slide = prs.slides.add_slide(title_slide_layout)
    title_slide.shapes.title.text = "Generated Presentation"
    title_slide.placeholders[1].text = "Created from document analysis"
    
    # Add content slides
    for slide in generated_slides:
        # Use the title and content layout
        slide_layout = prs.slide_layouts[1]  # Title and Content layout
        new_slide = prs.slides.add_slide(slide_layout)
        new_slide.shapes.title.text = slide["title"]
        
        # Add content to the slide
        content_shape = new_slide.placeholders[1]
        text_frame = content_shape.text_frame
        text_frame.clear()
        
        for para in slide["paragraphs"]:
            paragraph = text_frame.add_paragraph()
            paragraph.text = para
            paragraph.level = 0  # Bullet point level

    output_path = "presentation.pptx"
    prs.save(output_path)

    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="presentation.pptx"
    )
