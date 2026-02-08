"""
Document Q&A AI Agent - ArXiv Service
=====================================
Integration with ArXiv API for paper lookup and ingestion.

This module provides:
- Search ArXiv for papers
- Download and ingest ArXiv papers
- Metadata extraction from ArXiv
"""

import logging
import re
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import urllib.parse

import arxiv
from bs4 import BeautifulSoup
import requests

from src.config import settings
from src.document_processor import DocumentProcessor
from src.qa_engine import QAEngine

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """ArXiv paper metadata and content."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    updated_date: str
    categories: List[str]
    primary_category: str
    journal_ref: str = ""
    doi: str = ""
    pdf_url: str = ""
    comment: str = ""
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published_date": self.published_date,
            "updated_date": self.updated_date,
            "categories": self.categories,
            "primary_category": self.primary_category,
            "journal_ref": self.journal_ref,
            "doi": self.doi,
            "pdf_url": self.pdf_url,
            "comment": self.comment,
            "relevance_score": self.relevance_score
        }


class ArxivService:
    """
    Service for interacting with ArXiv API.
    
    Features:
    - Search papers by query
    - Download paper PDFs
    - Ingest papers into Q&A system
    """
    
    def __init__(self, qa_engine: Optional[QAEngine] = None):
        """
        Initialize ArXiv service.
        
        Args:
            qa_engine: Optional QA engine for direct ingestion
        """
        self.qa_engine = qa_engine
        self.document_processor = DocumentProcessor()
        self.base_url = "http://arxiv.org"
        self.search_url = "http://export.arxiv.org/api/query"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 3  # seconds
    
    def _rate_limit(self):
        """Enforce rate limiting for ArXiv API."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _parse_arxiv_id(self, id_string: str) -> str:
        """
        Parse and normalize ArXiv ID.
        
        Args:
            id_string: Raw ID string
            
        Returns:
            Normalized ArXiv ID
        """
        # Remove 'arxiv:' prefix if present
        id_string = re.sub(r'^arxiv:', '', id_string.strip(), flags=re.IGNORECASE)
        
        # Handle old-style IDs (hep-th/0307001)
        if '/' in id_string:
            return id_string
        
        # Handle new-style IDs (2301.00001)
        return id_string
    
    def search(self, query: str, max_results: int = 10, 
               sort_by: str = "relevance",
               sort_order: str = "descending") -> List[ArxivPaper]:
        """
        Search ArXiv for papers.
        
        Args:
            query: Search query (supports full ArXiv query syntax)
            max_results: Maximum number of results
            sort_by: Sort by 'relevance', 'lastUpdatedDate', 'submittedDate'
            sort_order: Sort order 'ascending' or 'descending'
            
        Returns:
            List of ArxivPaper objects
        """
        logger.info(f"Searching ArXiv for: {query}")
        
        try:
            self._rate_limit()
            
            # Map sort parameters
            sort_mapping = {
                "relevance": arxiv.SortOrder.Relevance,
                "lastUpdatedDate": arxiv.SortOrder.LastUpdatedDate,
                "submittedDate": arxiv.SortOrder.SubmittedDate
            }
            
            order = arxiv.SortOrder.Descending if sort_order == "descending" else arxiv.SortOrder.Ascending
            
            # Search ArXiv
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=sort_mapping.get(sort_by, arxiv.SortOrder.Relevance),
                sort_order=order
            )
            
            papers = []
            
            for result in search.results():
                # Extract authors
                authors = [author.name for author in result.authors]
                
                # Extract categories
                categories = [cat.split('.')[0] for cat in result.categories]
                primary_category = result.primary_category.split('.')[0] if result.primary_category else ""
                
                # Create paper object
                paper = ArxivPaper(
                    arxiv_id=self._parse_arxiv_id(result.entry_id),
                    title=result.title,
                    authors=authors,
                    abstract=result.summary,
                    published_date=str(result.published),
                    updated_date=str(result.updated),
                    categories=categories,
                    primary_category=primary_category,
                    journal_ref=result.journal_ref or "",
                    doi=result.doi or "",
                    pdf_url=result.pdf_url,
                    comment=result.comment or ""
                )
                
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            raise
    
    def search_by_description(self, description: str, 
                            max_results: int = 5) -> List[ArxivPaper]:
        """
        Search ArXiv based on a natural language description.
        
        This is useful when user describes what they're looking for
        rather than having a specific query.
        
        Args:
            description: Natural language description
            max_results: Maximum number of results
            
        Returns:
            List of relevant ArxivPaper objects
        """
        # Convert description to search query
        keywords = self._extract_keywords(description)
        
        # Build query from keywords
        query = " OR ".join(keywords[:5])  # Limit keywords
        
        # Add common ML/AI categories if relevant
        if any(kw in description.lower() for kw in ['neural', 'network', 'deep learning', 'transformer']):
            query = f"({query}) AND cat:cs.CV"
        
        papers = self.search(query, max_results=max_results)
        
        # Re-rank by relevance to original description
        papers = self._rerank_by_description(papers, description)
        
        return papers
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for search."""
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stopwords
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can',
            'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been',
            'what', 'when', 'where', 'who', 'how', 'which', 'this', 'that'
        }
        
        keywords = [w for w in words if w not in stopwords]
        
        # Get most common words
        from collections import Counter
        word_counts = Counter(keywords)
        top_keywords = [word for word, count in word_counts.most_common(10)]
        
        return top_keywords
    
    def _rerank_by_description(self, papers: List[ArxivPaper], 
                              description: str) -> List[ArxivPaper]:
        """Re-rank papers by relevance to description."""
        desc_words = set(self._extract_keywords(description))
        
        for paper in papers:
            # Score based on title and abstract
            title_words = set(paper.title.lower().split())
            abstract_words = set(paper.abstract.lower().split())
            
            overlap = desc_words & (title_words | abstract_words)
            paper.relevance_score = len(overlap) / max(len(desc_words), 1)
        
        # Sort by relevance score
        papers.sort(key=lambda p: p.relevance_score, reverse=True)
        
        return papers
    
    def download_pdf(self, arxiv_id: str, output_dir: str = "uploads") -> str:
        """
        Download PDF for an ArXiv paper.
        
        Args:
            arxiv_id: ArXiv paper ID
            output_dir: Directory to save PDF
            
        Returns:
            Path to downloaded PDF
        """
        logger.info(f"Downloading PDF for arxiv:{arxiv_id}")
        
        try:
            self._rate_limit()
            
            # Construct PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Download PDF
            output_path = Path(output_dir) / f"{arxiv_id}.pdf"
            
            response = requests.get(pdf_url, timeout=60)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"PDF saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            raise
    
    def ingest_paper(self, arxiv_id: str, 
                    qa_engine: Optional[QAEngine] = None) -> Dict[str, Any]:
        """
        Download and ingest an ArXiv paper into the Q&A system.
        
        Args:
            arxiv_id: ArXiv paper ID
            qa_engine: QA engine instance (uses default if not provided)
            
        Returns:
            Dictionary with paper metadata and document ID
        """
        if qa_engine is None:
            qa_engine = self.qa_engine
        
        if qa_engine is None:
            raise ValueError("No QA engine provided or configured")
        
        logger.info(f"Ingesting paper: {arxiv_id}")
        
        # Download PDF
        pdf_path = self.download_pdf(arxiv_id)
        
        # Add to Q&A system
        metadata = qa_engine.add_document(pdf_path)
        
        # Get paper metadata from ArXiv
        paper_info = self.get_paper_info(arxiv_id)
        
        result = {
            "document_id": metadata.get("document_id"),
            "arxiv_id": arxiv_id,
            "title": paper_info.title,
            "authors": paper_info.authors,
            "pdf_path": pdf_path,
            "metadata": metadata
        }
        
        return result
    
    def get_paper_info(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Get detailed information about a specific paper.
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            ArxivPaper object or None if not found
        """
        try:
            papers = self.search(f"id:{arxiv_id}", max_results=1)
            return papers[0] if papers else None
        except Exception as e:
            logger.error(f"Error getting paper info: {e}")
            return None
    
    def get_papers_by_category(self, category: str, 
                               max_results: int = 10) -> List[ArxivPaper]:
        """
        Get recent papers from a specific ArXiv category.
        
        Args:
            category: ArXiv category (e.g., 'cs.AI', 'cs.LG')
            max_results: Maximum number of results
            
        Returns:
            List of ArxivPaper objects
        """
        return self.search(f"cat:{category}", max_results=max_results)
    
    def get_trending_papers(self, category: Optional[str] = None,
                           max_results: int = 10) -> List[ArxivPaper]:
        """
        Get trending/recent papers.
        
        Args:
            category: Optional category filter
            max_results: Maximum number of results
            
        Returns:
            List of recent ArxivPaper objects
        """
        query = f"cat:{category}" if category else "all"
        return self.search(query, max_results=max_results, 
                          sort_by="submittedDate", sort_order="descending")
    
    def summarize_paper(self, arxiv_id: str, 
                       qa_engine: Optional[QAEngine] = None) -> Dict[str, Any]:
        """
        Get a quick summary of a paper without full ingestion.
        
        Args:
            arxiv_id: ArXiv paper ID
            qa_engine: QA engine instance
            
        Returns:
            Dictionary with paper info and summary
        """
        paper_info = self.get_paper_info(arxiv_id)
        
        if not paper_info:
            raise ValueError(f"Paper not found: {arxiv_id}")
        
        # Download PDF for processing
        pdf_path = self.download_pdf(arxiv_id)
        
        # Create a simple document processor
        doc_data = self.document_processor.process_document(pdf_path, arxiv_id)
        
        # Clean up PDF
        try:
            Path(pdf_path).unlink()
        except:
            pass
        
        # Generate summary using the document content
        if qa_engine is None:
            qa_engine = self.qa_engine
        
        if qa_engine:
            # Check if document was already ingested
            doc_list = qa_engine.list_documents()
            doc_ids = [d["document_id"] for d in doc_list]
            
            matching_id = next((d for d in doc_ids if arxiv_id in d), None)
            
            if matching_id:
                # Document already ingested, query it
                result = qa_engine.query(
                    "Provide a comprehensive summary of this paper including: "
                    "main contribution, methodology, key results, and conclusions.",
                    document_ids=[matching_id]
                )
                
                return {
                    "paper": paper_info.to_dict(),
                    "summary": result.answer,
                    "response_time": result.response_time
                }
        
        # Return basic info from metadata
        return {
            "paper": paper_info.to_dict(),
            "summary": paper_info.abstract,
            "message": "Full summary requires document ingestion"
        }

