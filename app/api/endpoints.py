# app/api/endpoints.py
from fastapi import APIRouter, Request, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Optional, List

from app.services.summarizer import SummarizerService
from app.utils.file_handler import extract_text_from_pdf
from app.core.config import settings
from .schemas import SummarizationResponse, ComparisonResponse

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def get_summarizer_service(request: Request) -> SummarizerService:
    """Dependency to get the summarizer service from app state."""
    return request.app.state.summarizer

@router.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    """Serve the HTML UI"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/summarize", response_model=SummarizationResponse)
async def summarize(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    method: str = Form(...),
    summarizer: SummarizerService = Depends(get_summarizer_service)
):
    """
    Summarize text using either abstractive or extractive method
    """
    input_text = ""
    try:
        if file and file.filename.endswith('.pdf'):
            content = await file.read()
            input_text = extract_text_from_pdf(content)
        elif text:
            input_text = text
        else:
            raise HTTPException(status_code=400, detail="Please provide either a PDF file or text input")

        if not input_text or len(input_text.strip()) < settings.MIN_TEXT_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Text is too short to summarize (minimum {settings.MIN_TEXT_LENGTH} characters)"
            )

        if method == "abstractive":
            summary = summarizer.abstractive_summarize(input_text)
        elif method == "extractive":
            summary = summarizer.extractive_summarize(input_text)
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Choose 'abstractive' or 'extractive'")

        return SummarizationResponse(
            summary=summary,
            method=method,
            original_length=len(input_text),
            summary_length=len(summary)
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=ComparisonResponse)
async def compare_documents(
    files: Optional[List[UploadFile]] = File(None),
    texts: Optional[List[str]] = Form(None),
    method: str = Form("abstractive"),
    summarizer: SummarizerService = Depends(get_summarizer_service)
):
    """
    Compare multiple documents by generating summaries and finding similarities/differences
    """
    try:
        documents = []
        
        # Process uploaded files
        if files:
            for file in files:
                if file.filename and file.filename.endswith('.pdf'):
                    content = await file.read()
                    text = extract_text_from_pdf(content)
                    documents.append({"name": file.filename, "text": text})
        
        # Process text inputs
        if texts:
            for i, text in enumerate(texts):
                if text and text.strip():
                    documents.append({"name": f"Document {i+1}", "text": text.strip()})
        
        if len(documents) < 2:
            raise HTTPException(status_code=400, detail="Please provide at least 2 documents to compare")
        
        if len(documents) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 documents allowed for comparison")
        
        # Generate summaries for each document
        summaries = []
        for doc in documents:
            if len(doc["text"].strip()) < settings.MIN_TEXT_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"{doc['name']} is too short to summarize"
                )
            
            if method == "abstractive":
                summary = summarizer.abstractive_summarize(doc["text"])
            else:
                summary = summarizer.extractive_summarize(doc["text"])
            
            summaries.append({
                "name": doc["name"],
                "summary": summary,
                "length": len(doc["text"])
            })
        
        # Find common themes and differences
        comparison_analysis = summarizer.compare_summaries([s["summary"] for s in summaries])
        
        return ComparisonResponse(
            summaries=summaries,
            common_themes=comparison_analysis["common_themes"],
            unique_points=comparison_analysis["unique_points"],
            method=method
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))