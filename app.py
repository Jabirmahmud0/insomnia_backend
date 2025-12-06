"""
Entry point for the Sleep Disorder Prediction API.
This file imports and runs the FastAPI application from the api module.
"""

from api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)