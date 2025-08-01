# Slide Generator - Full Stack Application

This document describes the complete slide generation system including PowerPoint presentation (PPTX) generation and interactive web frontend for the Task D slide generation system.

## Overview

The system provides a complete workflow for document-to-presentation generation:
1. **Document Upload**: Upload PDF/DOC/DOCX files through a web interface
2. **AI Content Generation**: Generate structured slide content using Google's Gemini AI
3. **Interactive Editing**: Edit slide content through a visual web interface
4. **PPTX Generation**: Generate PowerPoint presentations from edited content

The system includes both backend APIs and a frontend web interface, using `python-pptx` library and custom PowerPoint templates with predefined slide layouts.

## Features

### Backend Features
- **Template-based generation**: Uses `template_final.pptx` as the base template
- **Multiple slide layouts**: Supports 5 different slide layout types
- **Automatic placeholder population**: Intelligently maps JSON data to PowerPoint placeholders
- **Flexible placeholder matching**: Uses multiple strategies to find and populate placeholders
- **Error handling**: Robust error handling and logging for troubleshooting
- **Debug functionality**: Built-in template structure debugging
- **RESTful API**: Complete API for document processing and PPTX generation

### Frontend Features
- **🎨 Modern Web Interface**: Clean, responsive design with smooth animations
- **📄 Document Upload**: Drag-and-drop file upload with support for PDF, DOC, DOCX
- **⚡ Real-time Editing**: Click-to-edit functionality with auto-save
- **📊 Visual Slide Preview**: Each slide displayed as an editable card
- **💾 Auto-save**: Changes automatically saved to backend
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices
- **🔄 Live Updates**: Real-time synchronization between frontend and backend
- **🎯 User-friendly Workflow**: Step-by-step guided process

## Slide Layout Types

The template includes 5 slide layouts that correspond to the JSON structure:

1. **intro_slide**: Title slide with presentation title and date
2. **project_content_slide**: Overview and goal slide with two content areas
3. **content_slides**: Main content slides with bullet points
4. **chart_slides**: Chart slides (reserved for future use)
5. **final_slide**: Final slide with next steps

## JSON Data Structure

The system expects JSON data in the following format:

```json
{
  "presentation": {
    "intro_slide": {
      "presentation_title": "Main title (max 4 words)",
      "presentation_date": "DD.MM.YYYY"
    },
    "project_content_slide": {
      "presentation_overview": "3-sentence summary",
      "presentation_goal": "Primary goal statement"
    },
    "content_slides": [
      {
        "general_content_title": "Section title",
        "contents": [
          {
            "title": "Punkt A",
            "content-1": "2-3 sentence explanation"
          },
          {
            "title": "Punkt B", 
            "content-2": "2-3 sentence explanation"
          },
          {
            "title": "Punkt C",
            "content-3": "2-3 sentence explanation"
          },
          {
            "title": "Punkt D",
            "content-4": "2-3 sentence explanation"
          }
        ]
      }
    ],
    "final_slide": {
      "next_steps": [
        "First next step",
        "Second next step"
      ]
    }
  }
}
```

## Placeholder Naming Convention

The PowerPoint template should include placeholders with names that match the JSON structure:

### Intro Slide Placeholders:
- `presentation_title`
- `presentation_date`

### Project Content Slide Placeholders:
- `presentation_overview`
- `presentation_goal`

### Content Slides Placeholders:
- `general_content_title`
- `content-1`, `content-2`, `content-3`, `content-4`

### Final Slide Placeholders:
- `next_steps`

## API Endpoints

### Frontend Interface
- **`GET /`**: Serves the main web interface

### Document Processing
- **`POST /generate_json/`**: Generate slide content from uploaded document
  - **Request**: `doc` (file), `prompt` (string), `slide_count` (int)
  - **Response**: JSON structure with slide content
  - **Usage**: Initial content generation from document

### Content Management
- **`POST /update_json/`**: Update stored JSON with user edits
  - **Request**: Updated JSON structure
  - **Response**: Confirmation of successful update
  - **Usage**: Save user edits from frontend

- **`GET /get_current_json/`**: Retrieve current stored JSON data
  - **Response**: Current JSON structure
  - **Usage**: Frontend synchronization

### Presentation Generation
- **`POST /generate_pptx_from_json/`**: Generate PPTX from stored JSON
  - **Response**: File path to generated PPTX
  - **Usage**: Final presentation generation

### Legacy Endpoints
- **`POST /make_slides/`**: Original endpoint - generates both HTML and PPTX
  - **Request**: `doc` (file), `prompt` (string), `slide_count` (int)
  - **Response**: Both HTML files and PPTX file paths

- **`POST /test_pptx/`**: Test endpoint with sample data
  - **Response**: Test PPTX file path

## Getting Started

### Prerequisites
- Python 3.8+
- Google API key for Gemini AI
- Required Python packages (see Dependencies section)

### Installation
1. Clone the repository and navigate to the taskd directory
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   # Create .env file with:
   GOOGLE_API_KEY=your_google_api_key_here
   ```

### Running the Application

#### Option 1: Web Interface (Recommended)
1. Start the server:
   ```bash
   python run_server.py
   ```
2. Open your browser to `http://localhost:8000`
3. Follow the step-by-step workflow:
   - **Step 1**: Upload your document (PDF/DOC/DOCX)
   - **Step 2**: Edit the generated slide content
   - **Step 3**: Generate the final PowerPoint presentation

#### Option 2: API Only
1. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
2. Access API documentation at `http://localhost:8000/docs`
3. Use the API endpoints directly

### Usage Workflow

#### Web Interface Workflow
1. **📄 Upload Document**: 
   - Select PDF, DOC, or DOCX file
   - Enter generation instructions/prompt
   - Specify number of content slides
   - Click "Generate Slides Content"

2. **✏️ Edit Content**:
   - View generated slides as visual cards
   - Click on any text to edit it directly
   - Changes are automatically saved
   - Preview shows actual slide structure

3. **📥 Generate Presentation**:
   - Click "Generate PowerPoint Presentation"
   - Download the generated PPTX file
   - File saved with timestamp in `output_folder_name/`

#### API Workflow
1. **POST** `/generate_json/` - Upload document and generate content
2. **POST** `/update_json/` - Update content with edits (optional)
3. **POST** `/generate_pptx_from_json/` - Generate final PPTX

## File Output

Generated PPTX files are saved with the naming convention:
```
output_presentation_YYYYMMDD_HHMMSS.pptx
```

Files are saved in the `output_folder_name/` directory within the taskd folder.

## Error Handling

The system includes comprehensive error handling:

- **Template not found**: Clear error message if template file is missing
- **Layout not found**: Falls back to index-based layout selection
- **Placeholder not found**: Uses multiple matching strategies and fallbacks
- **Content issues**: Graceful handling of missing or malformed content

## Troubleshooting

### Debug Template Structure
The system includes a debug function that logs template structure:
```python
debug_template_structure(template_path)
```

This will log:
- Number of slide layouts
- Layout names
- Placeholder names for each layout

### Common Issues

1. **Template file not found**: Ensure `template_final.pptx` exists in `html-templates/` directory
2. **Placeholders not populated**: Check placeholder names in PowerPoint template match expected names
3. **Layout not found**: Verify slide master layouts are properly named in template

### Logging

The system uses Python logging to provide detailed information:
- INFO level: Normal operation progress
- WARNING level: Non-critical issues (fallbacks used)
- ERROR level: Critical errors with full stack traces

## Testing

### Web Interface Testing
1. Start the server:
   ```bash
   python run_server.py
   ```
2. Navigate to `http://localhost:8000`
3. Use the test workflow with a sample document

### API Testing
#### Standalone Test
```bash
python test_pptx.py
```

#### API Test with curl
```bash
# Test PPTX generation
curl -X POST http://localhost:8000/test_pptx/

# Test document upload (replace with actual file)
curl -X POST \
  -F "doc=@sample_document.pdf" \
  -F "prompt=Create a professional presentation" \
  -F "slide_count=3" \
  http://localhost:8000/generate_json/
```

#### Interactive API Documentation
Visit `http://localhost:8000/docs` for Swagger UI documentation with interactive testing.

## Dependencies

### Backend Dependencies
- `fastapi`: Web framework for API development
- `uvicorn`: ASGI server for running FastAPI
- `python-pptx`: PowerPoint file generation
- `python-multipart`: File upload support
- `PyMuPDF` (fitz): PDF document processing
- `langchain-google-genai`: Google Gemini AI integration
- `python-dotenv`: Environment variable management
- `logging`: Error tracking and debugging

### Frontend Dependencies
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with animations and responsive design
- **Vanilla JavaScript**: No external frameworks, pure JS for maximum compatibility

### Installation
```bash
# Install all backend dependencies
pip install fastapi uvicorn python-pptx python-multipart PyMuPDF langchain-google-genai python-dotenv

# Or use requirements file if available
pip install -r requirements.txt
```

## Project Structure

```
taskd/
├── main.py                 # Main FastAPI application
├── run_server.py           # Server startup script
├── test_pptx.py           # Testing utilities
├── README.md              # This documentation
├── static/                # Frontend files
│   ├── index.html         # Main web interface
│   ├── styles.css         # Styling and responsive design
│   └── script.js          # Frontend JavaScript logic
├── html-templates/        # PowerPoint templates and assets
│   ├── template_final.pptx # Main PowerPoint template
│   └── assets/            # Images and other assets
└── output_folder_name/    # Generated PPTX files
    └── output_presentation_*.pptx
```

## Future Enhancements

### Backend Improvements
- Chart generation support with data visualization
- Image insertion capability for richer presentations
- Advanced formatting options (fonts, colors, themes)
- Theme customization and branding options
- Batch processing support for multiple documents
- Export to additional formats (PDF, HTML)

### Frontend Improvements
- Drag-and-drop slide reordering
- Live preview of PPTX slides
- Collaborative editing support
- Version history and undo/redo functionality
- Advanced text formatting tools
- Image upload and insertion
- Template selection interface
- Real-time collaboration features

### Integration Enhancements
- User authentication and authorization
- Cloud storage integration (Google Drive, OneDrive)
- API rate limiting and caching
- Multi-language support
- Mobile app development
- Integration with presentation platforms (Teams, Zoom)

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For support and questions:
- Review the troubleshooting section above
- Check the API documentation at `/docs`
- [Add contact information or issue tracking]
