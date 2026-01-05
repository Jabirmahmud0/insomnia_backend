"""
Entry point for the Sleep Disorder Prediction API.
This file imports and runs the FastAPI application from the api module.
"""

from api.main import app

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)