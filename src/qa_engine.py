"""
Document Q&A AI Agent - Q&A Engine
==================================
Core question-answering logic with different query types.

This module handles:
- Direct content lookup
- Summarization
- Specific metric extraction
- Query classification and routing
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path
import hashlib
import json

from src.config import settings
from src.llm_service import get_llm_service, LLMResponse, SYSTEM_PROMPTS
from src.cache_manager import CacheManager
from src.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types."""
    LOOKUP = "lookup"           # Direct content lookup
    SUMMARIZATION = "summarize"  # Summarization request
    EXTRACTION = "extract"      # Specific data extraction
    COMPARISON = "compare"       # Comparison between documents/sections
    EXPLANATION = "explain"     # Explanation of concepts
    GENERAL = "general"         # General question


@dataclass
class QueryResult:
    """Result of a Q&A query."""
    answer: str
    query_type: str
    sources: List[str]
    confidence: float
    tokens_used: int
    response_time: float
    cached: bool = False
    context_used: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "query_type": self.query_type,
            "sources": self.sources,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "response_time": self.response_time,
            "cached": self.cached,
            "context_used": self.context_used
        }


class QAEngine:
    """
    Question-Answering Engine.
    
    Features:
    - Intelligent query classification
    - Context-aware response generation
    - Response caching
    - Multi-document support
    """
    
    def __init__(self):
        """Initialize the Q&A engine."""
        self.document_processor = DocumentProcessor(
            max_pages=settings.MAX_PAGES_PER_DOCUMENT
        )
        self.cache_manager = CacheManager()
        self._load_processed_documents()
    
    def _load_processed_documents(self):
        """Load already processed documents from storage."""
        self.documents: Dict[str, Dict[str, Any]] = {}
        data_dir = Path("data")
        
        if data_dir.exists():
            for json_file in data_dir.glob("*.json"):
                if json_file.name.startswith("doc_"):
                    try:
                        with open(json_file, 'r') as f:
                            doc_data = json.load(f)
                            self.documents[doc_data.get("document_id", json_file.stem)] = doc_data
                    except Exception as e:
                        logger.warning(f"Error loading document {json_file}: {e}")
        
        logger.info(f"Loaded {len(self.documents)} processed documents")
    
    def _get_cache_key(self, query: str, document_ids: List[str], 
                       query_type: str) -> str:
        """Generate cache key for query."""
        # Create unique key based on query, documents, and type
        key_content = f"{query}:{':'.join(sorted(document_ids))}:{query_type}"
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify the type of query.
        
        Args:
            query: User's question
            
        Returns:
            Tuple of (QueryType, confidence_score)
        """
        query_lower = query.lower()
        
        # Define patterns for each query type
        patterns = {
            QueryType.SUMMARIZATION: [
                r'\bsummarize\b',
                r'\bsummary\b',
                r'\bgive me an overview\b',
                r'\bwhat is the main idea\b',
                r'\bkey points\b',
                r'\bmain contributions\b'
            ],
            QueryType.EXTRACTION: [
                r'\bwhat (is|are) the .* (score|rate|accuracy|f1|precision|recall)\b',
                r'\bextract\b',
                r'\blist all\b',
                r'\bhow many\b',
                r'\bpercentage\b',
                r'\bnumbers?\b',
                r'\btable\b',
                r'\bstatistic'
            ],
            QueryType.COMPARISON: [
                r'\bcompare\b',
                r'\bdifference\b',
                r'\bsimilar\b',
                r'\bversus\b',
                r'\bvs\b',
                r'\bhow (is|are).* different\b'
            ],
            QueryType.EXPLANATION: [
                r'\bexplain\b',
                r'\bwhat is\b',
                r'\bhow does\b',
                r'\bwhy\b',
                r'\bdescribe\b',
                r'\bwhat does\b'
            ],
            QueryType.LOOKUP: [
                r'\bwhat (is|are|does|do)\b',
                r'\bwhere\b',
                r'\bwhen\b',
                r'\bwho\b',
                r'\bwhich\b',
                r'\bfind\b',
                r'\bsearch\b'
            ]
        }
        
        # Check each pattern
        best_match = QueryType.GENERAL
        best_confidence = 0.0
        
        for qtype, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    # Calculate confidence based on match strength
                    confidence = 0.7 + (0.1 * pattern_list.index(pattern))
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = qtype
        
        return best_match, best_confidence if best_confidence > 0 else 0.5
    
    def find_relevant_context(self, query: str, document_ids: List[str],
                             max_context_length: int = 50000) -> Tuple[str, List[str]]:
        """
        Find relevant context from documents for the query.
        
        Args:
            query: User's question
            document_ids: List of document IDs to search
            max_context_length: Maximum characters of context
            
        Returns:
            Tuple of (context_string, list of source document IDs)
        """
        contexts = []
        sources = []
        
        for doc_id in document_ids:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                text_content = doc.get("text_content", "")
                
                if not text_content:
                    continue
                
                # Simple keyword-based relevance scoring
                query_words = set(re.findall(r'\w+', query.lower()))
                doc_words = set(re.findall(r'\w+', text_content.lower()))
                
                # Calculate overlap
                overlap = query_words & doc_words
                if overlap:
                    # Find relevant sections
                    relevant_text = self._extract_relevant_sections(
                        text_content, query, max_context_length // len(document_ids)
                    )
                    
                    if relevant_text:
                        contexts.append(f"\n=== Document: {doc.get('metadata', {}).get('title', doc_id)} ===\n")
                        contexts.append(relevant_text)
                        sources.append(doc_id)
        
        # Combine contexts
        combined_context = "\n".join(contexts)
        
        # Truncate if needed
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length] + "\n... [content truncated]"
        
        return combined_context, sources
    
    def _extract_relevant_sections(self, text: str, query: str,
                                   max_length: int) -> str:
        """Extract relevant sections based on query keywords."""
        query_words = set(re.findall(r'\w+', query.lower()))
        query_words = {w for w in query_words if len(w) > 2}  # Skip short words
        
        if not query_words:
            return text[:max_length]
        
        # Split into sentences/sections
        sections = re.split(r'\n\n|(?:\-\-\- Page \d+ ---)', text)
        
        # Score each section
        scored_sections = []
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            section_lower = section.lower()
            score = sum(1 for word in query_words if word in section_lower)
            
            # Boost if section is near query matches
            if re.search('|'.join(query_words), section_lower):
                score *= 2
            
            if score > 0:
                scored_sections.append((score, i, section))
        
        # Sort by score and take top sections
        scored_sections.sort(key=lambda x: -x[0])
        
        # Combine top sections
        relevant_parts = []
        total_length = 0
        
        for score, idx, section in scored_sections:
            if total_length + len(section) > max_length:
                # Add partial section
                remaining = max_length - total_length
                if remaining > 100:
                    relevant_parts.append(section[:remaining])
                    total_length += remaining
                break
            
            relevant_parts.append(section)
            total_length += len(section)
        
        return "\n\n".join(relevant_parts)
    
    def _select_system_prompt(self, query_type: QueryType) -> str:
        """Select appropriate system prompt based on query type."""
        prompt_mapping = {
            QueryType.LOOKUP: SYSTEM_PROMPTS["general"],
            QueryType.SUMMARIZATION: SYSTEM_PROMPTS["summarization"],
            QueryType.EXTRACTION: SYSTEM_PROMPTS["extraction"],
            QueryType.COMPARISON: SYSTEM_PROMPTS["comparison"],
            QueryType.EXPLANATION: SYSTEM_PROMPTS["explanation"],
            QueryType.GENERAL: SYSTEM_PROMPTS["general"]
        }
        
        return prompt_mapping.get(query_type, SYSTEM_PROMPTS["general"])
    
    def _extract_specific_metrics(self, text: str, query: str) -> str:
        """Extract specific metrics from text based on query."""
        # Common metrics patterns
        metrics_patterns = {
            'accuracy': r'(?:accuracy)[\s:]*([\d.]+(?:\s*%)?)',
            'f1': r'(?:F1|F1[- ]?score)[\s:]*([\d.]+(?:\s*%)?)',
            'precision': r'(?:precision)[\s:]*([\d.]+(?:\s*%)?)',
            'recall': r'(?:recall)[\s:]*([\d.]+(?:\s*%)?)',
            'loss': r'(?:loss)[\s:]*([\d.]+)',
            'time': r'(?:time)[\s:]*([\d.]+(?:\s*[a-zA-Z]+)?)',
            'epochs': r'(?:epoch)[\s:]*(\d+)',
            'learning_rate': r'(?:learning[- ]?rate|lr)[\s:]*([\d.eE+-]+)'
        }
        
        # Check which metric is being queried
        query_lower = query.lower()
        extracted = []
        
        for metric_name, pattern in metrics_patterns.items():
            if metric_name in query_lower:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Get context around each match
                    for match in matches[:3]:  # Limit to 3 matches
                        context = self._get_metric_context(text, match, metric_name)
                        if context:
                            extracted.append(context)
        
        if extracted:
            return "\n\n".join(extracted[:5])
        
        return ""
    
    def _get_metric_context(self, text: str, match: str, 
                           metric_name: str) -> str:
        """Get context around a metric value."""
        # Find the match in text
        pattern = re.escape(match)
        match_pos = re.search(pattern, text, re.IGNORECASE)
        
        if not match_pos:
            return ""
        
        # Get surrounding context
        start = max(0, match_pos.start() - 100)
        end = min(len(text), match_pos.end() + 100)
        
        context = text[start:end]
        
        return f"...{context}..."
    
    def query(self, question: str, document_ids: Optional[List[str]] = None,
              llm_provider: Optional[str] = None,
              use_cache: bool = True) -> QueryResult:
        """
        Process a question and generate an answer.
        
        Args:
            question: User's question
            document_ids: List of document IDs to search (None for all)
            llm_provider: LLM provider to use
            use_cache: Whether to use cached responses
            
        Returns:
            QueryResult with answer and metadata
        """
        import time
        start_time = time.time()
        
        # Use all documents if none specified
        if document_ids is None:
            document_ids = list(self.documents.keys())
        
        if not document_ids:
            return QueryResult(
                answer="No documents have been uploaded yet. Please upload a document first.",
                query_type=QueryType.GENERAL.value,
                sources=[],
                confidence=1.0,
                tokens_used=0,
                response_time=time.time() - start_time
            )
        
        # Check cache
        if use_cache and settings.CACHE_ENABLED:
            query_type, _ = self.classify_query(question)
            cache_key = self._get_cache_key(question, document_ids, query_type.value)
            cached_result = self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for query: {question[:50]}...")
                cached_result.cached = True
                return cached_result
        
        # Classify query
        query_type, confidence = self.classify_query(question)
        logger.info(f"Query classified as: {query_type.value} (confidence: {confidence:.2f})")
        
        # Get relevant context
        context, sources = self.find_relevant_context(question, document_ids)
        
        if not context:
            # No relevant context found
            result = QueryResult(
                answer="I couldn't find relevant information in the uploaded documents to answer your question.",
                query_type=query_type.value,
                sources=sources,
                confidence=0.0,
                tokens_used=0,
                response_time=time.time() - start_time,
                context_used="none"
            )
            
            # Cache even negative results briefly
            if use_cache and settings.CACHE_ENABLED:
                self.cache_manager.set(cache_key, result, ttl=300)  # 5 min cache
            
            return result
        
        # Get LLM service
        llm = get_llm_service(llm_provider)
        
        # Get system prompt
        system_prompt = self._select_system_prompt(query_type)
        
        # Special handling for metric extraction
        if query_type == QueryType.EXTRACTION:
            metrics_context = self._extract_specific_metrics(context, question)
            if metrics_context:
                context = metrics_context
        
        # Generate response
        try:
            llm_response = llm.generate(
                prompt=question,
                system_prompt=system_prompt,
                context=context
            )
            
            response_time = time.time() - start_time
            
            result = QueryResult(
                answer=llm_response.text,
                query_type=query_type.value,
                sources=sources,
                confidence=confidence,
                tokens_used=llm_response.tokens_used,
                response_time=response_time,
                context_used=f"{len(context)} chars"
            )
            
            # Cache successful results
            if use_cache and settings.CACHE_ENABLED:
                self.cache_manager.set(cache_key, result, ttl=settings.CACHE_TTL)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            return QueryResult(
                answer=f"An error occurred while processing your question: {str(e)}",
                query_type=query_type.value,
                sources=sources,
                confidence=0.0,
                tokens_used=0,
                response_time=time.time() - start_time,
                context_used=f"{len(context)} chars"
            )
    
    async def aquery(self, question: str, document_ids: Optional[List[str]] = None,
                    llm_provider: Optional[str] = None,
                    use_cache: bool = True) -> QueryResult:
        """Async version of query."""
        # For now, use sync version (OpenAI async is handled internally)
        return self.query(question, document_ids, llm_provider, use_cache)
    
    def add_document(self, file_path: str, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process and add a document to the knowledge base.
        
        Args:
            file_path: Path to the document file
            document_id: Optional custom document ID
            
        Returns:
            Document metadata
        """
        if document_id is None:
            # Generate document ID from filename
            document_id = Path(file_path).stem
            # Add timestamp to ensure uniqueness
            document_id += f"_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Process document
        doc_data = self.document_processor.process_document(file_path, document_id)
        
        # Store in memory
        self.documents[document_id] = doc_data
        
        # Save to disk
        self._save_document(doc_data)
        
        logger.info(f"Document added: {document_id}")
        
        return doc_data["metadata"]
    
    def _save_document(self, doc_data: Dict[str, Any]):
        """Save processed document to disk."""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        file_path = data_dir / f"doc_{doc_data['document_id']}.json"
        
        with open(file_path, 'w') as f:
            json.dump(doc_data, f, indent=2)
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove a document from the knowledge base.
        
        Args:
            document_id: ID of document to remove
            
        Returns:
            True if removed, False if not found
        """
        if document_id in self.documents:
            del self.documents[document_id]
            
            # Also remove from disk
            file_path = Path(f"data/doc_{document_id}.json")
            if file_path.exists():
                file_path.unlink()
            
            # Clear related cache
            self.cache_manager.clear_pattern(document_id)
            
            logger.info(f"Document removed: {document_id}")
            return True
        
        return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all loaded documents."""
        return [
            {
                "document_id": doc_id,
                "title": doc.get("metadata", {}).get("title", "Unknown"),
                "page_count": doc.get("metadata", {}).get("page_count", 0),
                "processed_at": doc.get("processed_at", "")
            }
            for doc_id, doc in self.documents.items()
        ]
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a document."""
        if document_id in self.documents:
            return self.documents[document_id]
        return None

