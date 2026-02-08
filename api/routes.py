"""
Document Q&A AI Agent - API Routes
==================================
FastAPI routes for document upload, Q&A, and ArXiv integration.
"""

import os
import logging
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import settings
from src.qa_engine import QAEngine, QueryType
from src.arxiv_service import ArxivService
from src.utils import sanitize_filename, format_file_size, generate_unique_id

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services
qa_engine = QAEngine()
arxiv_service = ArxivService(qa_engine)


# ============================================
# Pydantic Models
# ============================================

class QueryRequest(BaseModel):
    """Q&A query request model."""
    question: str = Field(..., min_length=1, max_length=2000)
    document_ids: Optional[List[str]] = Field(default=None)
    llm_provider: Optional[str] = None
    use_cache: bool = Field(default=True)


class QueryResponse(BaseModel):
    """Q&A query response model."""
    answer: str
    query_type: str
    sources: List[str]
    confidence: float
    tokens_used: int
    response_time: float
    cached: bool = False
    sources_info: Optional[List[dict]] = None


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    document_id: str
    title: str
    page_count: int
    file_size: str
    message: str


class DocumentListItem(BaseModel):
    """Document list item model."""
    document_id: str
    title: str
    page_count: int
    processed_at: str


class ArxivSearchRequest(BaseModel):
    """ArXiv search request model."""
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=10, ge=1, le=50)
    sort_by: str = Field(default="relevance")
    sort_order: str = Field(default="descending")


class ArxivIngestRequest(BaseModel):
    """ArXiv ingest request model."""
    arxiv_id: str = Field(..., min_length=3, max_length=20)


class ArxivPaperInfo(BaseModel):
    """ArXiv paper info model."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    categories: List[str]
    pdf_url: str
    relevance_score: float = 0.0


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    service: str
    version: str
    documents_count: int
    cache_enabled: bool


# ============================================
# Document Routes
# ============================================

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document for processing.
    
    Args:
        file: PDF file to upload
        
    Returns:
        Document metadata and processing info
    """
    logger.info(f"Uploading document: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )
    
    # Check file size
    content = await file.read()
    file_size = len(content)
    
    if file_size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {format_file_size(settings.MAX_FILE_SIZE)}"
        )
    
    # Save file
    safe_filename = sanitize_filename(file.filename)
    unique_id = generate_unique_id("doc")
    file_path = Path(settings.UPLOAD_DIR) / f"{unique_id}_{safe_filename}"
    
    try:
        with open(file_path, 'wb') as f:
            f.write(content)
        
        # Process document
        metadata = qa_engine.add_document(str(file_path))
        
        logger.info(f"Document uploaded successfully: {metadata.get('document_id')}")
        
        return DocumentUploadResponse(
            document_id=metadata.get("document_id", unique_id),
            title=metadata.get("title", file.filename),
            page_count=metadata.get("page_count", 0),
            file_size=format_file_size(file_size),
            message="Document uploaded and processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    finally:
        # Cleanup uploaded file
        if file_path.exists():
            file_path.unlink()


@router.get("/documents", response_model=List[DocumentListItem])
async def list_documents():
    """
    List all uploaded documents.
    
    Returns:
        List of document metadata
    """
    documents = qa_engine.list_documents()
    return [
        DocumentListItem(
            document_id=doc["document_id"],
            title=doc["title"],
            page_count=doc["page_count"],
            processed_at=doc["processed_at"]
        )
        for doc in documents
    ]


@router.get("/documents/{document_id}")
async def get_document(document_id: str):
    """
    Get detailed information about a document.
    
    Args:
        document_id: Document ID
        
    Returns:
        Document details
    """
    doc_info = qa_engine.get_document_info(document_id)
    
    if not doc_info:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    return {
        "document_id": doc_info["document_id"],
        "metadata": doc_info["metadata"],
        "structure": doc_info.get("structure", {}),
        "tables_count": len(doc_info.get("tables", [])),
        "text_length": len(doc_info.get("text_content", "")),
        "processed_at": doc_info.get("processed_at", "")
    }


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document from the knowledge base.
    
    Args:
        document_id: Document ID to delete
        
    Returns:
        Deletion status
    """
    success = qa_engine.remove_document(document_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    return {"message": f"Document {document_id} deleted successfully"}


# ============================================
# Q&A Routes
# ============================================

@router.post("/qa/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Ask a question about uploaded documents.
    
    Args:
        request: Query request with question and options
        
    Returns:
        Answer with metadata
    """
    logger.info(f"Processing query: {request.question[:100]}...")
    
    try:
        result = qa_engine.query(
            question=request.question,
            document_ids=request.document_ids,
            llm_provider=request.llm_provider,
            use_cache=request.use_cache
        )
        
        # Get source info
        sources_info = []
        for doc_id in result.sources:
            doc_info = qa_engine.get_document_info(doc_id)
            if doc_info:
                sources_info.append({
                    "document_id": doc_id,
                    "title": doc_info.get("metadata", {}).get("title", "Unknown"),
                    "page_count": doc_info.get("metadata", {}).get("page_count", 0)
                })
        
        return QueryResponse(
            answer=result.answer,
            query_type=result.query_type,
            sources=result.sources,
            confidence=result.confidence,
            tokens_used=result.tokens_used,
            response_time=result.response_time,
            cached=result.cached,
            sources_info=sources_info if sources_info else None
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/qa/history")
async def get_query_history():
    """
    Get query history (placeholder for future implementation).
    
    Returns:
        Query history
    """
    # This would require session management
    return {"message": "Query history feature coming soon"}


# ============================================
# ArXiv Routes (Bonus Feature)
# ============================================

@router.post("/arxiv/search")
async def search_arxiv(request: ArxivSearchRequest):
    """
    Search ArXiv for papers.
    
    Args:
        request: Search request with query
        
    Returns:
        List of matching papers
    """
    logger.info(f"Searching ArXiv: {request.query}")
    
    try:
        papers = arxiv_service.search(
            query=request.query,
            max_results=request.max_results,
            sort_by=request.sort_by,
            sort_order=request.sort_order
        )
        
        return {
            "query": request.query,
            "results_count": len(papers),
            "papers": [paper.to_dict() for paper in papers]
        }
        
    except Exception as e:
        logger.error(f"ArXiv search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching ArXiv: {str(e)}"
        )


@router.post("/arxiv/search-by-description")
async def search_arxiv_by_description(
    query: str = Query(..., min_length=10),
    max_results: int = Query(default=5, ge=1, le=10)
):
    """
    Search ArXiv based on a natural language description.
    
    This is useful when users describe what they're looking for
    rather than having a specific query.
    
    Args:
        query: Natural language description
        max_results: Maximum results
        
    Returns:
        List of relevant papers
    """
    logger.info(f"Searching ArXiv by description: {query[:100]}...")
    
    try:
        papers = arxiv_service.search_by_description(
            description=query,
            max_results=max_results
        )
        
        return {
            "description": query,
            "results_count": len(papers),
            "papers": [paper.to_dict() for paper in papers]
        }
        
    except Exception as e:
        logger.error(f"ArXiv search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching ArXiv: {str(e)}"
        )


@router.post("/arxiv/ingest")
async def ingest_arxiv_paper(request: ArxivIngestRequest):
    """
    Download and ingest an ArXiv paper.
    
    Args:
        request: Ingest request with ArXiv ID
        
    Returns:
        Ingestion result with document ID
    """
    logger.info(f"Ingesting ArXiv paper: {request.arxiv_id}")
    
    try:
        result = arxiv_service.ingest_paper(
            arxiv_id=request.arxiv_id,
            qa_engine=qa_engine
        )
        
        return {
            "arxiv_id": request.arxiv_id,
            "document_id": result["document_id"],
            "title": result["title"],
            "authors": result["authors"],
            "pdf_path": result["pdf_path"],
            "metadata": result["metadata"],
            "message": "Paper ingested successfully"
        }
        
    except Exception as e:
        logger.error(f"ArXiv ingest error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error ingesting paper: {str(e)}"
        )


@router.get("/arxiv/paper/{arxiv_id}")
async def get_arxiv_paper(arxiv_id: str):
    """
    Get information about an ArXiv paper.
    
    Args:
        arxiv_id: ArXiv paper ID
        
    Returns:
        Paper information
    """
    paper = arxiv_service.get_paper_info(arxiv_id)
    
    if not paper:
        raise HTTPException(
            status_code=404,
            detail="Paper not found"
        )
    
    return paper.to_dict()


@router.get("/arxiv/trending")
async def get_trending_papers(
    category: Optional[str] = Query(default=None),
    max_results: int = Query(default=10, ge=1, le=20)
):
    """
    Get trending/recent papers from ArXiv.
    
    Args:
        category: Optional category filter
        max_results: Maximum results
        
    Returns:
        List of recent papers
    """
    try:
        papers = arxiv_service.get_trending_papers(
            category=category,
            max_results=max_results
        )
        
        return {
            "category": category,
            "results_count": len(papers),
            "papers": [paper.to_dict() for paper in papers]
        }
        
    except Exception as e:
        logger.error(f"Error getting trending papers: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting trending papers: {str(e)}"
        )


# ============================================
# Utility Routes
# ============================================

@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics.
    
    Returns:
        Cache statistics
    """
    from src.cache_manager import CacheManager
    cache = CacheManager()
    return cache.stats()


@router.post("/cache/clear")
async def clear_cache():
    """
    Clear all cached responses.
    
    Returns:
        Clear status
    """
    from src.cache_manager import CacheManager
    cache = CacheManager()
    success = cache.clear()
    
    return {
        "message": "Cache cleared successfully" if success else "Failed to clear cache"
    }

