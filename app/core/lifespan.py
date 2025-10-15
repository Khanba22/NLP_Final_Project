# app/core/lifespan.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
import nltk
from app.services.summarizer import SummarizerService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load resources
    print("Starting up and loading models...")
    
    # Download NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    # Initialize and load the summarizer model
    summarizer_service = SummarizerService()
    app.state.summarizer = summarizer_service
    print("Models loaded successfully!")
    
    yield
    
    # Shutdown: Clean up resources
    print("Shutting down and clearing resources...")
    app.state.summarizer = None
    # Add any additional cleanup logic here