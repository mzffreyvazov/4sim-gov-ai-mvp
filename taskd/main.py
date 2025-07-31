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
from pptx.util import Inches, Pt
import logging
from datetime import datetime


# Load env vars
load_dotenv()
GOOGLE_API_KEY = ""

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

    # C) Analyze user prompt + doc and generate outline
    combined_prompt = PromptTemplate(
        input_variables=["prompt", "text", "slide_count"],
        template=(
            "User instruction: {prompt}\n\n"
            "Document content:\n{text}\n\n"
            "Based on the document and the user instruction, generate a presentation outline with exactly {slide_count} slides.\n"
            "Return ONLY a JSON object with a single key 'slides' which is an array of objects, where each object has:\n"
            "  title: string (in Azerbaijani),\n"
            "  bullets: array of short bullet points (in Azerbaijani).\n"
        )
    )
    combined_chain = combined_prompt | llm | StrOutputParser()
    combined_resp = combined_chain.invoke({
        "prompt": prompt,
        "text": doc_text,
        "slide_count": slide_count
    })
    logger.info(f"Combined response: {combined_resp}")
    clean_json = extract_json_from_response(combined_resp)
    logger.info(f"Cleaned JSON for outline: {clean_json}")
    slides_outline = json.loads(clean_json)["slides"]



    # E) Flesh out each slide + translate
    generated_slides = []
    # Prepare project_content_text and project_goals_text using LLM for better quality
    # 1. Project Content (summary)
    content_summary_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Summarize the following document in 1-2 sentences for a presentation slide (in Azerbaijani):\n{text}"
        )
    )
    content_summary_chain = content_summary_prompt | llm | StrOutputParser()
    project_content_text = content_summary_chain.invoke({"text": doc_text[:2000]})
    if hasattr(project_content_text, "content"):
        project_content_text = project_content_text.content
    elif isinstance(project_content_text, dict) and "text" in project_content_text:
        project_content_text = project_content_text["text"]
    if not isinstance(project_content_text, str):
        project_content_text = str(project_content_text)
    project_content_text = project_content_text.strip()

    # 2. Project Goals & Next Steps
    goals_steps_prompt = PromptTemplate(
        input_variables=["prompt"],
        template=(
            "Given the user instruction '{prompt}', generate a JSON object with two keys:\n"
            "  goals: A concise summary of the project goals in Azerbaijani.\n"
            "  next_steps: A short 'Next Steps' section in Azerbaijani."
        )
    )
    goals_steps_chain = goals_steps_prompt | llm | StrOutputParser()
    goals_steps_resp = goals_steps_chain.invoke({"prompt": prompt})
    clean_json_gs = extract_json_from_response(goals_steps_resp)
    goals_steps_data = json.loads(clean_json_gs)
    project_goals_text = goals_steps_data["goals"]
    next_steps_text = goals_steps_data["next_steps"]

    # 4. Flesh out each slide in Azerbaijani
    for slide_def in slides_outline:
        content_prompt = PromptTemplate(
            input_variables=["title", "bullets"],
            template=(
                "Write a concise 2–3 sentence slide text in Azerbaijani for a slide titled "
                "'{title}' with bullet points {bullets}."
            )
        )
        content_chain = content_prompt | llm | StrOutputParser()
        az_text = content_chain.invoke({
            "title": slide_def["title"],
            "bullets": slide_def["bullets"]
        })
        if hasattr(az_text, "content"):
            az_text = az_text.content
        elif isinstance(az_text, dict) and "text" in az_text:
            az_text = az_text["text"]
        if not isinstance(az_text, str):
            az_text = str(az_text)
        az_text = az_text.strip()
        generated_slides.append({
            "title": slide_def["title"],
            "paragraphs": [p.strip() for p in az_text.split("\n") if p.strip()]
        })

    # F) Create the PowerPoint presentation
    # Layout indices from your template master
    INTRO_LAYOUT_INDEX = 0
    CONTENT_LAYOUT_INDEX = 1  # Project Content & Goals
    INFO_LAYOUT_INDEX = 2     # Main punkts slides
    NEXT_LAYOUT_INDEX = 4     # Next Steps slide

    # Background and logo image paths
    BG_INTRO = "taskd/assets/slide-basliq.jpg"
    BG_CONTENT = "taskd/assets/slide-giris_and_novbetiAddim.jpg"
    BG_INFO = "taskd/assets/slide-punkts.jpg"

    # F) Assemble PPTX from your designed template
    prs = Presentation("taskd/assets/template.pptx")

    # 1. Introduction slide
    slide = prs.slides.add_slide(prs.slide_layouts[INTRO_LAYOUT_INDEX])
    slide.shapes.add_picture(BG_INTRO, 0, 0, width=prs.slide_width, height=prs.slide_height)
    # Title textbox
    title_box = slide.shapes.add_textbox(Inches(3.12), Inches(2.94), Inches(8.45), Inches(0.74))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Your Presentation Title"
    p.font.bold = True
    p.font.size = Pt(44)
    p.font.name = "Arial"
    # Subtitle textbox
    sub_box = slide.shapes.add_textbox(Inches(3.12), Inches(3.81), Inches(8.45), Inches(0.34))
    tf2 = sub_box.text_frame
    p2 = tf2.paragraphs[0]
    p2.text = datetime.now().strftime("%Y-%m-%d")
    p2.font.size = Pt(20)
    p2.font.name = "Arial"

    # 2. Project Content & Goals slide
    slide = prs.slides.add_slide(prs.slide_layouts[CONTENT_LAYOUT_INDEX])
    slide.shapes.add_picture(BG_CONTENT, 0, 0, width=prs.slide_width, height=prs.slide_height)
    # Project Content textbox
    pc_box = slide.shapes.add_textbox(Inches(0.6), Inches(1.29), Inches(4.44), Inches(0.4))
    tf_pc = pc_box.text_frame
    p_pc = tf_pc.paragraphs[0]
    p_pc.text = project_content_text
    p_pc.font.bold = True
    p_pc.font.size = Pt(24)
    p_pc.font.name = "Arial"
    # Goals textbox
    goal_box = slide.shapes.add_textbox(Inches(7.11), Inches(1.29), Inches(4.44), Inches(0.4))
    tf_goal = goal_box.text_frame
    p_goal = tf_goal.paragraphs[0]
    p_goal.text = project_goals_text
    p_goal.font.bold = True
    p_goal.font.size = Pt(24)
    p_goal.font.name = "Arial"

    # 3. Info/Punkts slides
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN

    for gen in generated_slides:
        slide = prs.slides.add_slide(prs.slide_layouts[INFO_LAYOUT_INDEX])
        slide.shapes.add_picture(BG_INFO, 0, 0, width=prs.slide_width, height=prs.slide_height)
        # Title textbox
        top_title = slide.shapes.add_textbox(Inches(0.61), Inches(1.03), Inches(10), Inches(0.42))
        tf_t = top_title.text_frame
        p_t = tf_t.paragraphs[0]
        p_t.text = gen["title"]
        p_t.font.bold = True
        p_t.font.size = Pt(25)
        p_t.font.name = "Arial"
        # Four punkts columns
        x_start = 0.61
        for idx, content in enumerate(gen["punkts"]):
            # Circle label
            cx = Inches(x_start + idx * 2.86 + 1.27)
            cy = Inches(2.51)
            circle = slide.shapes.add_shape(
                MSO_SHAPE.OVAL,
                cx, cy,
                Inches(0.31), Inches(0.31)
            )
            circle.fill.solid()
            circle.fill.fore_color.rgb = RGBColor(0x8B, 0x77, 0x35)
            lbl = circle.text_frame.paragraphs[0]
            lbl.text = chr(65 + idx)
            lbl.font.size = Pt(16)
            lbl.font.bold = True
            lbl.font.name = "Arial"
            # Header box
            header = slide.shapes.add_textbox(
                Inches(x_start + idx * 2.86), Inches(2.76),
                Inches(2.86), Inches(1.23)
            )
            th = header.text_frame
            ph = th.paragraphs[0]
            ph.text = f"Punk {idx+1}"
            ph.font.bold = True
            ph.font.size = Pt(16)
            ph.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            ph.font.name = "Arial"
            header.fill.solid()
            header.fill.fore_color.rgb = RGBColor(0x8B, 0x77, 0x35)
            # Content box
            cb = slide.shapes.add_textbox(
                Inches(x_start + idx * 2.86), Inches(4.16),
                Inches(2.86), Inches(1.08)
            )
            tc = cb.text_frame
            pc = tc.paragraphs[0]
            pc.text = content
            pc.font.size = Pt(14)
            pc.font.name = "Arial"
            pc.alignment = PP_ALIGN.LEFT

    # 4. Next Steps slide
    slide = prs.slides.add_slide(prs.slide_layouts[NEXT_LAYOUT_INDEX])
    slide.shapes.add_picture(BG_CONTENT, 0, 0, width=prs.slide_width, height=prs.slide_height)
    ns_box = slide.shapes.add_textbox(Inches(0.6), Inches(1.29), Inches(10), Inches(3))
    tf_ns = ns_box.text_frame
    p_ns = tf_ns.paragraphs[0]
    p_ns.text = next_steps_text
    p_ns.font.size = Pt(18)
    p_ns.font.name = "Arial"

    output_path = "generated_presentation.pptx"
    prs.save(output_path)

    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename="presentation.pptx"
    )
