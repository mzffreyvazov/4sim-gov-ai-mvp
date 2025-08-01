#!/usr/bin/env python3
"""
Server runner for the Slide Generator FastAPI application.
Run this to start the server with the frontend interface.
"""

import uvicorn

if __name__ == "__main__":
    # Run the FastAPI app
    print("🚀 Starting Slide Generator Server...")
    print("📱 Frontend available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
