import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import fitz
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

# --- Setup remains the same ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
app = FastAPI(title="Task D: Slide Generation")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper functions remain the same ---
def extract_json_from_response(text: str):
    # ... (code is unchanged)
    try:
        json_start = text.index('{')
        json_end = text.rindex('}') + 1
        return text[json_start:json_end]
    except ValueError:
        raise ValueError("No valid JSON object found in the LLM response.")

async def parse_doc(file: UploadFile = File(...)):
    # ... (code is unchanged)
    try:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        data = await file.read()
        doc = fitz.open(stream=data, filetype=ext)
        text = "".join(page.get_text() for page in doc)
        return {"text": text}
    except Exception as e:
        logger.error(f"Error parsing document: {e}")
        raise HTTPException(status_code=400, detail="Failed to parse the document.")

# --- UPDATED HTML GENERATION FUNCTION ---
def generate_html_slides(data: dict) -> list[str]:
    """
    Uses Jinja2 to populate HTML templates with JSON data and save them as files.
    """
    # 1. Setup Jinja2 Environment
    templates_dir = os.path.join(os.path.dirname(__file__), "html-templates")
    env = Environment(loader=FileSystemLoader(templates_dir))
    logger.info(f"Looking for templates in: {templates_dir}")
    logger.info(f"Files in template directory: {os.listdir(templates_dir)}")
    logger.info(f"Templates seen by Jinja2: {env.list_templates()}")
    output_dir = "output_slides_html"
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    slide_counter = 1

    presentation_data = data.get("presentation", {})

    # 2. Generate Intro Slide
    intro_data = presentation_data.get("intro_slide", {})
    if intro_data:
        template = env.get_template("slide-basliq.html")
        rendered_html = template.render(
            presentation_title=intro_data.get("presentation_title", "Başlıq yoxdur"),
            presentation_date=intro_data.get("presentation_date", "")
        )
        file_path = os.path.join(output_dir, f"{slide_counter:02d}_intro.html")
        with open(file_path, "w", encoding="utf-8") as f: f.write(rendered_html)
        generated_files.append(file_path)
        slide_counter += 1

    # 3. Generate Project Content & Goal Slide
    content_goal_data = presentation_data.get("project_content_slide", {})
    if content_goal_data:
        template = env.get_template("slide-giris_and_novbeti.html")
        rendered_html = template.render(
            left_panel_title="Layihənin Məzmunu",
            right_panel_title="Məqsəd",
            left_panel_content=content_goal_data.get("presentation_overview", ""),
            right_panel_content=content_goal_data.get("presentation_goal", ""),
            page_number=slide_counter
        )
        file_path = os.path.join(output_dir, f"{slide_counter:02d}_content_goal.html")
        with open(file_path, "w", encoding="utf-8") as f: f.write(rendered_html)
        generated_files.append(file_path)
        slide_counter += 1

    # 4. Generate Main Content Slides (Punkts)
    content_slides = presentation_data.get("content_slides", [])
    if content_slides:
        template = env.get_template("slide-punkts.html")
        for i, slide in enumerate(content_slides):
            # The looping logic is now inside the HTML template
            rendered_html = template.render(
                general_content_title=slide.get("general_content_title", "Məzmun"),
                contents=slide.get("contents", []),
                page_number=slide_counter
            )
            file_path = os.path.join(output_dir, f"{slide_counter:02d}_content_{i+1}.html")
            with open(file_path, "w", encoding="utf-8") as f: f.write(rendered_html)
            generated_files.append(file_path)
            slide_counter += 1
            
    # 5. Generate Chart Slides
    # (Leaving this logic here for when you're ready to implement it)
    chart_slides = presentation_data.get("chart_slides", [])
    if chart_slides:
        template = env.get_template("slide-with_charts.html")
        for i, slide in enumerate(chart_slides):
             rendered_html = template.render(
                content_title=slide.get("content_title", "Diaqram"),
                chart_summary=slide.get("chart_summary", ""), # You would pass chart data here
                page_number=slide_counter
            )
             file_path = os.path.join(output_dir, f"{slide_counter:02d}_chart_{i+1}.html")
             with open(file_path, "w", encoding="utf-8") as f: f.write(rendered_html)
             generated_files.append(file_path)
             slide_counter += 1

    # 6. Generate Final Slide (Next Steps)
    final_data = presentation_data.get("final_slide", {})
    if final_data:
        next_steps_list = final_data.get("next_steps", [])
        # Format as an HTML list
        next_steps_html = "<ul>" + "".join(f"<li>{step}</li>" for step in next_steps_list) + "</ul>"
        
        template = env.get_template("slide-giris_and_novbeti.html")
        rendered_html = template.render(
            left_panel_title="Növbəti Addımlar",
            right_panel_title="",
            left_panel_content=next_steps_html,
            right_panel_content="",
            page_number=slide_counter
        )
        file_path = os.path.join(output_dir, f"{slide_counter:02d}_final.html")
        with open(file_path, "w", encoding="utf-8") as f: f.write(rendered_html)
        generated_files.append(file_path)

    return generated_files

# --- /make_slides/ endpoint remains the same, it will now call the corrected function ---
@app.post("/make_slides/")
async def make_slides(
    doc: UploadFile = File(...),
    prompt: str = Form(...),
    slide_count: int = Form(...)
):
    # ... (Part 1: LLM call is unchanged)
    parsed_doc = await parse_doc(doc)
    doc_text = parsed_doc["text"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1, api_key=GOOGLE_API_KEY)
    json_format_string = """
{
  "presentation": {
    "intro_slide": { "presentation_title": "...", "presentation_date": "..." },
    "project_content_slide": { "presentation_overview": "...", "presentation_goal": "..." },
    "content_slides": [ { "general_content_title": "...", "contents": [ { "title": "...", "content": "..." } ] } ],
    "chart_slides": [],
    "final_slide": { "next_steps": ["..."] }
  }
}
"""
    generation_prompt = PromptTemplate(
        input_variables=["prompt", "text", "slide_count", "json_format", "current_date"],
        template=(
            "User instruction: {prompt}\n\nDocument content:\n---\n{text}\n---\n\n"
            "**CRITICAL RULES:**\n"
            "1. `content_slides` array MUST contain EXACTLY {slide_count} objects.\n"
            "2. Inside EACH `content_slides` object, the `contents` array MUST contain EXACTLY 4 objects.\n"
            "3. The `presentation_title` in `intro_slide` MUST be a maximum of 4 words.\n\n"
            "Return ONLY a single valid JSON object.\n\nJSON Format Example:\n{json_format}"
        )
    )
    generation_chain = generation_prompt | llm | StrOutputParser()
    current_date_str = datetime.now().strftime("%d.%m.%Y")
    llm_response = generation_chain.invoke({
        "prompt": prompt, "text": doc_text, "slide_count": slide_count,
        "json_format": json_format_string, "current_date": current_date_str
    })
    
    try:
        clean_json_str = extract_json_from_response(llm_response)
        output_json = json.loads(clean_json_str)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate valid JSON. Error: {e}")

    # --- Part 2: HTML generation call (now works correctly) ---
    try:
        generated_files = generate_html_slides(output_json)
        logger.info(f"Successfully generated {len(generated_files)} HTML slide files.")
        return JSONResponse(content={
            "message": "HTML slides generated successfully.",
            "files": generated_files
        })
    except Exception as e:
        logger.error(f"Failed during HTML generation stage: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate HTML slides. Error: {e}")