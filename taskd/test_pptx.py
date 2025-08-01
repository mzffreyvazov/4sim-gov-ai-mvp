#!/usr/bin/env python3
"""
Test script for PPTX generation functionality.
This script tests the PPTX generation without requiring the full FastAPI server.
"""

import sys
import os
import json
import logging
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

# Import the function from final2.py
from final2 import generate_pptx_presentation, debug_template_structure

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pptx_generation():
    """Test the PPTX generation with sample data."""
    
    # Sample JSON data for testing (same as in the test endpoint)
    sample_data = {
        "presentation": {
            "intro_slide": {
                "presentation_title": "Test Təqdimat",
                "presentation_date": "01.08.2025"
            },
            "project_content_slide": {
                "presentation_overview": "Bu təqdimat test məqsədilə yaradılmışdır. Bu, python-pptx kitabxanasının işlədiyini yoxlamaq üçündür.",
                "presentation_goal": "PPTX generasiya funksionallığının düzgün işlədiyini təsdiqləmək."
            },
            "content_slides": [
                {
                    "general_content_title": "Əsas Məzmun",
                    "contents": [
                        {
                            "title": "Punkt A",
                            "content-1": "Bu birinci punkt haqqında məlumatdır."
                        },
                        {
                            "title": "Punkt B", 
                            "content-2": "Bu ikinci punkt haqqında məlumatdır."
                        },
                        {
                            "title": "Punkt C",
                            "content-3": "Bu üçüncü punkt haqqında məlumatdır."
                        },
                        {
                            "title": "Punkt D",
                            "content-4": "Bu dördüncü punkt haqqında məlumatdır."
                        }
                    ]
                }
            ],
            "final_slide": {
                "next_steps": [
                    "Növbəti addım: Test nəticələrini yoxlamaq",
                    "İkinci addım: Real məlumatlarla test etmək"
                ]
            }
        }
    }
    
    print("Starting PPTX generation test...")
    print(f"Sample data: {json.dumps(sample_data, indent=2, ensure_ascii=False)}")
    
    try:
        # Test the template debugging first
        template_path = os.path.join(os.path.dirname(__file__), "html-templates", "template_final.pptx")
        print(f"\nTemplate path: {template_path}")
        print(f"Template exists: {os.path.exists(template_path)}")
        
        if os.path.exists(template_path):
            print("\nDebugging template structure...")
            debug_template_structure(template_path)
        
        # Generate PPTX
        print("\nGenerating PPTX presentation...")
        pptx_file_path = generate_pptx_presentation(sample_data)
        
        print(f"✅ Success! PPTX file generated: {pptx_file_path}")
        print(f"File exists: {os.path.exists(pptx_file_path)}")
        
        if os.path.exists(pptx_file_path):
            file_size = os.path.getsize(pptx_file_path)
            print(f"File size: {file_size} bytes")
        
        return pptx_file_path
        
    except Exception as e:
        print(f"❌ Error during PPTX generation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_pptx_generation()
