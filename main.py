"""
Document Q&A AI Agent - Main Entry Point
=========================================
An enterprise-ready AI agent for intelligent document question-answering.

This application provides:
- Multi-document PDF processing
- Intelligent text extraction
- LLM-powered Q&A interface
- ArXiv integration (Bonus feature)
- Enterprise-grade features (caching, rate limiting)

Author: AI Developer
Version: 1.0.0
"""

import os
import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import settings
from api.routes import router as api_router
from utils import setup_logging, cleanup_old_files

# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Document Q&A AI Agent...")
    
    # Create necessary directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.DISK_CACHE_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Cleanup old uploaded files (older than 24 hours)
    cleanup_old_files(settings.UPLOAD_DIR, max_age_hours=24)
    
    logger.info("✅ Application startup complete!")
    logger.info(f"📡 Server running at http://{settings.HOST}:{settings.PORT}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Document Q&A AI Agent...")


# Create FastAPI application
app = FastAPI(
    title="Document Q&A AI Agent",
    description="""
    An enterprise-ready AI agent for intelligent document question-answering.
    
    ## Features
    - 📄 Multi-document PDF processing
    - 🔍 Intelligent text extraction
    - 💬 LLM-powered Q&A interface
    - 📚 ArXiv integration (Bonus)
    - ⚡ Enterprise-grade optimizations
    
    ## Getting Started
    1. Upload PDF documents using `/api/documents/upload`
    2. Ask questions using `/api/qa/query`
    3. Search ArXiv using `/api/arxiv/search`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else settings.CORS_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Include API routes
app.include_router(api_router, prefix="/api")


# Rate limiting middleware
from collections import defaultdict
from datetime import datetime, timedelta

rate_limit_storage = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    if not settings.RATE_LIMIT_ENABLED:
        return await call_next(request)
    
    client_ip = request.client.host
    current_time = datetime.now()
    window_start = current_time - timedelta(seconds=settings.RATE_LIMIT_WINDOW)
    
    # Clean old requests
    rate_limit_storage[client_ip] = [
        req_time for req_time in rate_limit_storage[client_ip]
        if req_time > window_start
    ]
    
    # Check rate limit
    if len(rate_limit_storage[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too many requests",
                "message": f"Rate limit exceeded. Please wait before making more requests.",
                "retry_after": settings.RATE_LIMIT_WINDOW
            }
        )
    
    # Add current request
    rate_limit_storage[client_ip].append(current_time)
    
    response = await call_next(request)
    return response


# Request logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request logging middleware."""
    start_time = datetime.now()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds() * 1000
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.2f}ms"
    )
    
    return response


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Document Q&A AI Agent",
        "version": "1.0.0",
        "llm_provider": settings.LLM_PROVIDER,
        "model": settings.MODEL_NAME if settings.LLM_PROVIDER == "openai" else settings.GEMINI_MODEL_NAME
    }


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic info."""
    return {
        "service": "Document Q&A AI Agent",
        "version": "1.0.0",
        "description": "An enterprise-ready AI agent for intelligent document Q&A",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "api": "/api"
        }
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "type": str(type(exc).__name__)
        }
    )


if __name__ == "__main__":
    """Run the application."""
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )

