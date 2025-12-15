"""
FastAPI application for Physician Notetaker.

Provides REST API endpoints for medical transcript processing.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import os

from physician_notetaker import (
    MedicalPreprocessor,
    MedicalNER,
    MedicalSummarizer,
    SentimentIntentClassifier,
    SOAPGenerator,
    KeywordExtractor
)
from physician_notetaker.utils import get_logger, redact_phi

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Physician Notetaker API",
    description="End-to-end NLP pipeline for medical transcription processing",
    version="1.0.0"
)

# Initialize components
preprocessor = MedicalPreprocessor()
ner = None  # Will be lazy-loaded
summarizer = MedicalSummarizer()
sentiment_classifier = None  # Will be lazy-loaded
soap_generator = SOAPGenerator()
keyword_extractor = KeywordExtractor()


def get_ner():
    """Lazy load NER model."""
    global ner
    if ner is None:
        try:
            ner = MedicalNER(model_name="en_core_sci_sm")
        except:
            # Fallback to basic model
            ner = MedicalNER(model_name="en_core_web_sm")
    return ner


def get_sentiment_classifier():
    """Lazy load sentiment classifier."""
    global sentiment_classifier
    if sentiment_classifier is None:
        sentiment_classifier = SentimentIntentClassifier()
    return sentiment_classifier


# Request/Response models
class TranscriptRequest(BaseModel):
    """Request model for transcript input."""
    text: str = Field(..., description="Medical transcript text")
    redact_phi: bool = Field(False, description="Whether to redact PHI")


class NERResponse(BaseModel):
    """Response model for NER extraction."""
    entities: List[Dict[str, Any]]
    entities_by_type: Dict[str, List[Dict[str, Any]]]
    confidence: float
    num_entities: int


class SummaryResponse(BaseModel):
    """Response model for medical summary."""
    Patient_Name: Optional[str]
    Incident: Dict[str, Optional[str]]
    Symptoms: List[str]
    Diagnosis: List[str]
    Treatment: List[str]
    Duration: List[str]
    Current_Status: Optional[str]
    Prognosis: Optional[str]
    Follow_Up_Advice: Optional[str]
    keywords: List[Dict[str, Any]]
    confidence: float
    requires_clarification: bool
    notes: str


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    overall_sentiment: str
    overall_sentiment_score: float
    overall_intent: str
    overall_confidence: float
    turn_classifications: List[Dict[str, Any]]


class SOAPResponse(BaseModel):
    """Response model for SOAP note."""
    Subjective: Dict[str, Any]
    Objective: Dict[str, Any]
    Assessment: Dict[str, Any]
    Plan: Dict[str, Any]
    confidence: float


class FullPipelineResponse(BaseModel):
    """Response model for full pipeline."""
    ner: NERResponse
    summary: SummaryResponse
    sentiment: SentimentResponse
    soap: SOAPResponse


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Physician Notetaker API",
        "version": "1.0.0",
        "endpoints": {
            "/extract": "Extract named entities (NER)",
            "/summarize": "Generate structured medical summary",
            "/sentiment": "Analyze sentiment and intent",
            "/soap": "Generate SOAP note",
            "/process": "Run full pipeline"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/extract", response_model=NERResponse)
async def extract_entities(request: TranscriptRequest):
    """
    Extract named entities from medical transcript.
    
    Extracts:
    - SYMPTOM (e.g., "neck pain", "headache")
    - DIAGNOSIS (e.g., "whiplash injury")
    - TREATMENT (e.g., "physiotherapy", "painkillers")
    - PROGNOSIS (e.g., "full recovery")
    - EVENT_DATE, INCIDENT, DURATION, SEVERITY
    """
    try:
        text = request.text
        if request.redact_phi:
            text = redact_phi(text)
        
        # Preprocess
        preprocessed = preprocessor.preprocess(text)
        
        # Extract entities
        ner_model = get_ner()
        ner_result = ner_model.extract_entities(text, preprocessed)
        
        return NERResponse(**ner_result)
    
    except Exception as e:
        logger.error(f"Error in extract_entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize", response_model=SummaryResponse)
async def generate_summary(request: TranscriptRequest):
    """
    Generate structured medical report from transcript.
    
    Returns JSON with:
    - Patient information
    - Incident details
    - Symptoms, Diagnosis, Treatment
    - Prognosis and follow-up advice
    - Keywords with relevance scores
    - Confidence scores
    """
    try:
        text = request.text
        if request.redact_phi:
            text = redact_phi(text)
        
        # Preprocess
        preprocessed = preprocessor.preprocess(text)
        
        # Extract entities
        ner_model = get_ner()
        ner_result = ner_model.extract_entities(text, preprocessed)
        
        # Extract keywords
        keywords = keyword_extractor.extract_keywords(
            text,
            entities=ner_result['entities'],
            top_k=10
        )
        
        # Generate summary
        summary = summarizer.summarize(
            text,
            entities=ner_result['entities'],
            entities_by_type=ner_result['entities_by_type'],
            preprocessed_data=preprocessed,
            keywords=keywords
        )
        
        return SummaryResponse(**summary)
    
    except Exception as e:
        logger.error(f"Error in generate_summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TranscriptRequest):
    """
    Analyze sentiment and intent of patient utterances.
    
    Returns:
    - Overall sentiment: Anxious, Neutral, Reassured
    - Overall intent: Seeking reassurance, Reporting symptoms, etc.
    - Per-turn classifications
    - Confidence scores
    """
    try:
        text = request.text
        if request.redact_phi:
            text = redact_phi(text)
        
        # Preprocess
        preprocessed = preprocessor.preprocess(text)
        
        # Analyze sentiment
        classifier = get_sentiment_classifier()
        sentiment_result = classifier.classify_transcript(
            text,
            speaker_turns=preprocessed.get('speaker_turns')
        )
        
        return SentimentResponse(**sentiment_result)
    
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/soap", response_model=SOAPResponse)
async def generate_soap_note(request: TranscriptRequest):
    """
    Generate SOAP (Subjective, Objective, Assessment, Plan) note.
    
    Returns structured SOAP note with:
    - Subjective: Patient complaints and history
    - Objective: Physical examination findings
    - Assessment: Diagnosis and prognosis
    - Plan: Treatment and follow-up
    """
    try:
        text = request.text
        if request.redact_phi:
            text = redact_phi(text)
        
        # Preprocess
        preprocessed = preprocessor.preprocess(text)
        
        # Extract entities (for context)
        ner_model = get_ner()
        ner_result = ner_model.extract_entities(text, preprocessed)
        
        # Extract keywords
        keywords = keyword_extractor.extract_keywords(
            text,
            entities=ner_result['entities'],
            top_k=10
        )
        
        # Generate summary (for context)
        summary = summarizer.summarize(
            text,
            entities=ner_result['entities'],
            entities_by_type=ner_result['entities_by_type'],
            preprocessed_data=preprocessed,
            keywords=keywords
        )
        
        # Generate SOAP note
        soap_note = soap_generator.generate_soap(
            text,
            speaker_turns=preprocessed.get('speaker_turns'),
            entities_by_type=ner_result['entities_by_type'],
            medical_summary=summary
        )
        
        return SOAPResponse(**soap_note)
    
    except Exception as e:
        logger.error(f"Error in generate_soap_note: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process", response_model=FullPipelineResponse)
async def process_full_pipeline(request: TranscriptRequest):
    """
    Run the complete NLP pipeline on a transcript.
    
    Returns all outputs:
    - NER entities
    - Structured medical summary
    - Sentiment and intent analysis
    - SOAP note
    """
    try:
        text = request.text
        if request.redact_phi:
            text = redact_phi(text)
        
        # Preprocess
        preprocessed = preprocessor.preprocess(text)
        
        # Extract entities
        ner_model = get_ner()
        ner_result = ner_model.extract_entities(text, preprocessed)
        
        # Extract keywords
        keywords = keyword_extractor.extract_keywords(
            text,
            entities=ner_result['entities'],
            top_k=10
        )
        
        # Generate summary
        summary = summarizer.summarize(
            text,
            entities=ner_result['entities'],
            entities_by_type=ner_result['entities_by_type'],
            preprocessed_data=preprocessed,
            keywords=keywords
        )
        
        # Analyze sentiment
        classifier = get_sentiment_classifier()
        sentiment_result = classifier.classify_transcript(
            text,
            speaker_turns=preprocessed.get('speaker_turns')
        )
        
        # Generate SOAP note
        soap_note = soap_generator.generate_soap(
            text,
            speaker_turns=preprocessed.get('speaker_turns'),
            entities_by_type=ner_result['entities_by_type'],
            medical_summary=summary
        )
        
        return FullPipelineResponse(
            ner=NERResponse(**ner_result),
            summary=SummaryResponse(**summary),
            sentiment=SentimentResponse(**sentiment_result),
            soap=SOAPResponse(**soap_note)
        )
    
    except Exception as e:
        logger.error(f"Error in process_full_pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_transcript(file: UploadFile = File(...)):
    """
    Upload a transcript file and process it.
    
    Accepts .txt files.
    """
    try:
        # Read file
        content = await file.read()
        text = content.decode('utf-8')
        
        # Process using full pipeline
        request = TranscriptRequest(text=text)
        result = await process_full_pipeline(request)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in upload_transcript: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
