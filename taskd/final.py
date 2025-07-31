import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from datetime import datetime

# Load environment variables from a .env file
load_dotenv()
# It's recommended to handle the API key securely, e.g., through environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Initialize FastAPI app
app = FastAPI(title="Task D: Slide Generation")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_json_from_response(text: str) -> str:
    """
    Extracts a JSON object from a string, reliably finding the start and end braces
    and ignoring any surrounding text, newlines, or markdown code blocks.
    """
    try:
        # Find the first opening brace of the JSON object
        json_start = text.index('{')
        # Find the last closing brace of the JSON object
        json_end = text.rindex('}') + 1
        # Slice the string to get just the JSON part
        return text[json_start:json_end]
    except ValueError as e:
        # This error occurs if '{' or '}' are not found
        logger.error(f"Could not find a valid JSON object in the response string. Error: {e}")
        raise ValueError("No valid JSON object found in the LLM response.")

@app.post("/parse_doc/")
async def parse_doc(file: UploadFile = File(...)):
    """
    Parses an uploaded document (PDF, etc.) to extract raw text.
    """
    try:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        data = await file.read()
        doc = fitz.open(stream=data, filetype=ext)
        text = "".join(page.get_text() for page in doc)
        return {"text": text}
    except Exception as e:
        logger.error(f"Error parsing document: {e}")
        raise HTTPException(status_code=400, detail="Failed to parse the document.")


@app.post("/make_slides/")
async def make_slides(
    doc: UploadFile = File(...),
    prompt: str = Form(...),
    slide_count: int = Form(...)
):
    """
    Generates slide content in a structured JSON format based on a document and a prompt.
    """
    # A) Parse the document to get its text content
    parsed_doc = await parse_doc(doc)
    doc_text = parsed_doc["text"]

    # B) Initialize the Google Generative AI model
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        api_key=GOOGLE_API_KEY
    )

    # C) Define the desired JSON output structure as a string
    json_format_string = """
{
  "presentation": {
    "intro_slide": {
      "presentation_title": "The main title of the presentation in Azerbaijani",
      "presentation_date": "DD.MM.YYYY"
    },
    "project_content_slide": {
      "presentation_overview": "A 3-sentence summary of the document's content in Azerbaijani.",
      "presentation_goal": "A concise statement about the presentation's primary goal in Azerbaijani."
    },
    "content_slides": [
      {
        "general_content_title": "A title for this section of the presentation in Azerbaijani",
        "contents": [
          { "title": "Punkt A", "content": "A 2-3 sentence explanation for the first point in Azerbaijani." },
          { "title": "Punkt B", "content": "A 2-3 sentence explanation for the second point in Azerbaijani." },
          { "title": "Punkt C", "content": "A 2-3 sentence explanation for the third point in Azerbaijani." },
          { "title": "Punkt D", "content": "A 2-3 sentence explanation for the fourth point in Azerbaijani." }
        ]
      }
    ],
    "chart_slides": [],
    "final_slide": { "next_steps": ["First next step or action item in Azerbaijani."] }
  }
}
"""

    # D) Create a single, powerful prompt with strict rules to generate the entire JSON structure
    generation_prompt = PromptTemplate(
        input_variables=["prompt", "text", "slide_count", "json_format", "current_date"],
        template=(
            "User instruction: {prompt}\n\n"
            "Document content:\n---\n{text}\n---\n\n"
            "Based on the document and the user instruction, generate a complete presentation structure in Azerbaijani. "
            "The current date is {current_date}.\n\n"
            "**CRITICAL RULES - YOU MUST FOLLOW THESE:**\n"
            "1. The `content_slides` array MUST contain EXACTLY {slide_count} slide objects. Do not generate more or less than {slide_count}.\n"
            "2. Inside EACH object in the `content_slides` array, the `contents` array MUST contain EXACTLY 4 content objects. Each of these 4 objects must have a 'title' and a 'content' field. If the source text for a slide seems to have fewer than 4 points, you MUST break down the existing points or elaborate to create 4 distinct points to meet this requirement.\n\n"
            "You MUST return ONLY a single valid JSON object that strictly follows the structure, conventions, and the CRITICAL RULES above. "
            "Do not add any text, explanations, or markdown formatting before or after the JSON object.\n\n"
            "JSON Format Example:\n{json_format}"
        )
    )

    # E) Create and run the generation chain
    generation_chain = generation_prompt | llm | StrOutputParser()
    current_date_str = datetime.now().strftime("%d.%m.%Y")
    
    logger.info("Generating slide content with strict rules...")
    llm_response = generation_chain.invoke({
        "prompt": prompt,
        "text": doc_text,
        "slide_count": slide_count,
        "json_format": json_format_string,
        "current_date": current_date_str
    })
    logger.info("LLM raw response received.")

    # F) Extract, parse, and return the JSON from the response
    try:
        clean_json_str = extract_json_from_response(llm_response)
        output_json = json.loads(clean_json_str)
        
        # Final validation (optional but good practice)
        if len(output_json.get("presentation", {}).get("content_slides", [])) != slide_count:
            logger.warning(f"LLM did not return the correct number of content slides. Expected {slide_count}, got {len(output_json.get('presentation', {}).get('content_slides', []))}")
            
        return JSONResponse(content=output_json)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to decode or find JSON in LLM response. Error: {e}")
        logger.error(f"Problematic string from LLM: {llm_response}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate valid JSON from the language model. Error: {e}"
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during slide generation."
        )