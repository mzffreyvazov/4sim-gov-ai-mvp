# PPTX Generation Feature

This document describes the new PowerPoint presentation (PPTX) generation functionality added to the Task D slide generation system.

## Overview

The system can now generate both HTML slides and PowerPoint presentations from the same JSON data structure. The PPTX generation uses the `python-pptx` library and a custom PowerPoint template with predefined slide layouts.

## Features

- **Template-based generation**: Uses `template_final.pptx` as the base template
- **Multiple slide layouts**: Supports 5 different slide layout types
- **Automatic placeholder population**: Intelligently maps JSON data to PowerPoint placeholders
- **Flexible placeholder matching**: Uses multiple strategies to find and populate placeholders
- **Error handling**: Robust error handling and logging for troubleshooting
- **Debug functionality**: Built-in template structure debugging

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

### `/make_slides/` (Updated)
The main endpoint now generates both HTML and PPTX files:

**Request:**
- `doc`: PDF document file
- `prompt`: Generation instructions
- `slide_count`: Number of content slides

**Response:**
```json
{
  "message": "HTML slides and PPTX presentation generated successfully.",
  "html_files": ["path/to/slide1.html", "path/to/slide2.html", ...],
  "pptx_file": "path/to/output_presentation_YYYYMMDD_HHMMSS.pptx",
  "json_data": { ... }
}
```

### `/test_pptx/` (New)
Test endpoint for PPTX generation with sample data:

**Response:**
```json
{
  "message": "Test PPTX presentation generated successfully.",
  "pptx_file": "path/to/test_presentation.pptx", 
  "sample_data": { ... }
}
```

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

### Standalone Test
Run the standalone test script:
```bash
python test_pptx.py
```

### API Test
1. Start the server:
```bash
python run_server.py
```

2. Test the endpoint:
```bash
curl -X POST http://localhost:8000/test_pptx/
```

## Dependencies

The PPTX generation requires:
- `python-pptx`: PowerPoint file generation
- `fastapi`: Web framework
- `logging`: Error tracking
- `datetime`: Timestamp generation

## Future Enhancements

Potential improvements:
- Chart generation support
- Image insertion capability
- Advanced formatting options
- Theme customization
- Batch processing support
