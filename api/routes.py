"""
FastAPI routes for ML-powered web scraper
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime
import json

from scraper.pipeline import pipeline, ContentProcessor
from ml.ollama_client import OllamaClient
from ml.nlp_processor import NLPProcessor
from ml.neural_networks import ContentClassifier
from database.connection import get_db_session, health_monitor
from storage.s3_client import S3Client
from utils.validators import validate_url, validate_content
from utils.exporters import DataExporter
from config.settings import settings

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ML-Powered Web Scraper API",
    description="Advanced web scraping with ML analysis and local AI processing",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
ollama_client = OllamaClient()
nlp_processor = NLPProcessor()
content_classifier = ContentClassifier()
s3_client = S3Client()
data_exporter = DataExporter()

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

manager = ConnectionManager()

# Pydantic models
class ScrapeRequest(BaseModel):
    urls: List[HttpUrl]
    engine_type: str = "auto"
    ml_analysis: bool = True
    ollama_model: Optional[str] = None

class ContentAnalysisRequest(BaseModel):
    content: str
    title: Optional[str] = ""
    enable_nlp: bool = True
    enable_classification: bool = True
    enable_ollama: bool = True
    ollama_model: Optional[str] = None

class OllamaRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7

class ExportRequest(BaseModel):
    job_id: str
    format: str = "json"  # json, csv, pdf
    filters: Optional[Dict[str, Any]] = None

# Health check endpoints
@app.get("/health")
async def health_check():
    """System health check"""
    try:
        # Check database
        db_health = await health_monitor.check_health()
        
        # Check Ollama
        ollama_health = await ollama_client.health_check()
        
        # Check S3
        s3_health = await s3_client.health_check()
        
        overall_status = "healthy" if all([
            db_health['status'] == 'healthy',
            ollama_health,
            s3_health
        ]) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                "database": db_health,
                "ollama": {"status": "healthy" if ollama_health else "unhealthy"},
                "s3": {"status": "healthy" if s3_health else "unhealthy"}
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "ML-Powered Web Scraper API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Scraping endpoints
@app.post("/scrape")
async def scrape_urls(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Start scraping job"""
    try:
        # Validate URLs
        valid_urls = []
        for url in request.urls:
            if validate_url(str(url)):
                valid_urls.append(str(url))
        
        if not valid_urls:
            raise HTTPException(status_code=400, detail="No valid URLs provided")
        
        # Start scraping job
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(valid_urls)}"
        
        # Create progress callback for WebSocket updates
        async def progress_callback(job_data):
            await manager.broadcast({
                "type": "job_progress",
                "job_id": job_id,
                "data": job_data
            })
        
        # Start background scraping
        background_tasks.add_task(
            pipeline.process_urls,
            valid_urls,
            job_id,
            request.engine_type,
            request.ml_analysis,
            request.ollama_model,
            progress_callback
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "total_urls": len(valid_urls),
            "message": "Scraping job started successfully"
        }
        
    except Exception as e:
        logger.error(f"Scraping request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results"""
    try:
        job_data = pipeline.get_job_status(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job_data
        
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs")
async def list_jobs():
    """List all active jobs"""
    try:
        active_jobs = pipeline.get_active_jobs()
        return {
            "active_jobs": active_jobs,
            "total_count": len(active_jobs)
        }
        
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    try:
        success = await pipeline.cancel_job(job_id)
        
        if success:
            await manager.broadcast({
                "type": "job_cancelled",
                "job_id": job_id
            })
            return {"message": "Job cancelled successfully"}
        else:
            raise HTTPException(status_code=404, detail="Job not found or cannot be cancelled")
            
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ML Analysis endpoints
@app.post("/analyze/content")
async def analyze_content(request: ContentAnalysisRequest):
    """Analyze content using ML pipeline"""
    try:
        if not validate_content(request.content):
            raise HTTPException(status_code=400, detail="Invalid content provided")
        
        # Prepare options
        options = {
            'enable_nlp': request.enable_nlp,
            'enable_classification': request.enable_classification,
            'enable_ollama': request.enable_ollama,
            'ollama_model': request.ollama_model or settings.DEFAULT_MODEL
        }
        
        # Process content
        processor = ContentProcessor()
        results = await processor.process_content(
            request.content,
            request.title,
            options=options
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/nlp")
async def analyze_nlp(content: str, title: str = ""):
    """NLP analysis only"""
    try:
        if not validate_content(content):
            raise HTTPException(status_code=400, detail="Invalid content provided")
        
        results = await nlp_processor.analyze_content(content, title)
        return results
        
    except Exception as e:
        logger.error(f"NLP analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/classify")
async def classify_content(content: str):
    """Content classification only"""
    try:
        if not validate_content(content):
            raise HTTPException(status_code=400, detail="Invalid content provided")
        
        results = await content_classifier.classify_content(content)
        return results
        
    except Exception as e:
        logger.error(f"Content classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/quality")
async def score_quality(content: str):
    """Content quality scoring"""
    try:
        if not validate_content(content):
            raise HTTPException(status_code=400, detail="Invalid content provided")
        
        results = await content_classifier.score_quality(content)
        return results
        
    except Exception as e:
        logger.error(f"Quality scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/duplicates")
async def detect_duplicates(content1: str, content2: str):
    """Duplicate content detection"""
    try:
        if not validate_content(content1) or not validate_content(content2):
            raise HTTPException(status_code=400, detail="Invalid content provided")
        
        results = await content_classifier.detect_duplicates(content1, content2)
        return results
        
    except Exception as e:
        logger.error(f"Duplicate detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Ollama endpoints
@app.get("/ollama/models")
async def list_ollama_models():
    """List available Ollama models"""
    try:
        models = await ollama_client.list_models()
        return {
            "models": models,
            "default_model": settings.DEFAULT_MODEL,
            "backup_model": settings.BACKUP_MODEL
        }
        
    except Exception as e:
        logger.error(f"Failed to list Ollama models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ollama/generate")
async def generate_text(request: OllamaRequest):
    """Generate text using Ollama"""
    try:
        result = await ollama_client.generate(
            request.prompt,
            request.model,
            request.system_prompt,
            request.max_tokens,
            request.temperature
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ollama/analyze")
async def ollama_analyze(content: str, model: str = None, analysis_type: str = "comprehensive"):
    """Analyze content using Ollama"""
    try:
        if not validate_content(content):
            raise HTTPException(status_code=400, detail="Invalid content provided")
        
        result = await ollama_client.analyze_content(content, model, analysis_type)
        return result
        
    except Exception as e:
        logger.error(f"Ollama analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export endpoints
@app.post("/export")
async def export_data(request: ExportRequest, background_tasks: BackgroundTasks):
    """Export job data"""
    try:
        # Check if job exists
        job_data = pipeline.get_job_status(request.job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Start export task
        export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        background_tasks.add_task(
            data_exporter.export_job_data,
            request.job_id,
            request.format,
            export_id,
            request.filters
        )
        
        return {
            "export_id": export_id,
            "status": "started",
            "format": request.format,
            "message": "Export started successfully"
        }
        
    except Exception as e:
        logger.error(f"Export request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/exports/{export_id}")
async def get_export_status(export_id: str):
    """Get export status"""
    try:
        # This would typically check export status from database
        # For now, return a simple response
        return {
            "export_id": export_id,
            "status": "completed",  # Simplified
            "message": "Export completed"
        }
        
    except Exception as e:
        logger.error(f"Failed to get export status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics endpoints
@app.get("/stats/system")
async def get_system_stats():
    """Get system statistics"""
    try:
        from database.connection import get_table_stats
        
        # Get database stats
        db_stats = await get_table_stats()
        
        # Get S3 stats
        s3_stats = await s3_client.get_bucket_stats()
        
        # Get model info
        model_info = content_classifier.get_model_info()
        
        return {
            "database": db_stats,
            "storage": s3_stats,
            "models": model_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/jobs")
async def get_job_stats():
    """Get job statistics"""
    try:
        # This would query the database for job statistics
        # Simplified implementation
        active_jobs = pipeline.get_active_jobs()
        
        stats = {
            "active_jobs": len(active_jobs),
            "total_jobs": 0,  # Would come from database
            "completed_jobs": 0,  # Would come from database
            "failed_jobs": 0,  # Would come from database
            "timestamp": datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get job stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            
            # Echo back for testing
            await websocket.send_text(f"Echo: {data}")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Utility endpoints
@app.post("/validate/url")
async def validate_url_endpoint(url: str):
    """Validate URL"""
    try:
        is_valid = validate_url(url)
        return {
            "url": url,
            "is_valid": is_valid,
            "message": "URL is valid" if is_valid else "URL is invalid"
        }
        
    except Exception as e:
        logger.error(f"URL validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate/content")
async def validate_content_endpoint(content: str):
    """Validate content"""
    try:
        is_valid = validate_content(content)
        return {
            "content_length": len(content),
            "is_valid": is_valid,
            "message": "Content is valid" if is_valid else "Content is invalid"
        }
        
    except Exception as e:
        logger.error(f"Content validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("Starting ML-Powered Web Scraper API")
    
    # Initialize database
    try:
        from database.connection import init_database
        await init_database()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Check Ollama connection
    try:
        models = await ollama_client.list_models()
        logger.info(f"Ollama connected with models: {models}")
    except Exception as e:
        logger.warning(f"Ollama connection failed: {e}")
    
    # Check S3 connection
    try:
        s3_health = await s3_client.health_check()
        if s3_health:
            logger.info("S3 connection verified")
        else:
            logger.warning("S3 connection failed")
    except Exception as e:
        logger.warning(f"S3 check failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("Shutting down ML-Powered Web Scraper API")
    
    # Close database connections
    try:
        from database.connection import close_database
        await close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Database shutdown error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.routes:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT + 1,
        reload=settings.DEBUG
    )