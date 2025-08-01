#!/usr/bin/env python3
"""
Simple test runner for the FastAPI endpoints.
Run this to test the /test_pptx/ endpoint.
"""

import uvicorn

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run("final2:app", host="0.0.0.0", port=8000, reload=True)
