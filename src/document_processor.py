"""
Document Q&A AI Agent - Document Processor
===========================================
Multi-modal PDF document ingestion and text extraction.

This module handles:
- PDF text extraction using multiple libraries
- Structure recognition (titles, sections, tables)
- Table detection and extraction
- Figure and equation preservation hints
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import pypdf
import pdfplumber
import fitz  # PyMuPDF
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""
    title: str = ""
    author: str = ""
    creation_date: str = ""
    page_count: int = 0
    file_size: int = 0
    file_path: str = ""
    file_name: str = ""
    document_id: str = ""
    processed_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "author": self.author,
            "creation_date": self.creation_date,
            "page_count": self.page_count,
            "file_size": self.file_size,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "document_id": self.document_id,
            "processed_at": self.processed_at
        }


@dataclass
class ExtractedSection:
    """Represents a section of extracted content."""
    title: str = ""
    level: int = 0
    content: str = ""
    page_number: int = 0
    element_type: str = "text"  # text, table, figure, equation


@dataclass
class TableData:
    """Represents extracted table data."""
    page_number: int
    table_number: int
    caption: str = ""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    text_representation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "page_number": self.page_number,
            "table_number": self.table_number,
            "caption": self.caption,
            "headers": self.headers,
            "rows": self.rows,
            "text_representation": self.text_representation
        }


class DocumentProcessor:
    """
    Multi-modal document processor for PDF files.
    
    Features:
    - Extract text with high accuracy
    - Detect and extract tables
    - Identify document structure
    - Preserve context and relationships
    """
    
    def __init__(self, max_pages: int = 100):
        """
        Initialize the document processor.
        
        Args:
            max_pages: Maximum pages to process per document
        """
        self.max_pages = max_pages
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize regex patterns for structure detection."""
        # Section title patterns
        self.section_patterns = [
            r'^Chapter\s+\d+[\.:]\s*(.+)$',
            r'^\d+[\.\:]\s*(.+)$',
            r'^(Abstract|Introduction|Related\s+Work|Methodology|Methods|Results|Discussion|Conclusion|References|Bibliography)\s*$',
            r'^\s*\d+\.\d+\s+(.+)$',  # 1.1, 2.3, etc.
        ]
        
        # Table caption pattern
        self.table_caption_pattern = r'^(Table|Tab\.)\s*(\d+)[\.\:\s]*(.+)$'
        
        # Figure caption pattern
        self.figure_caption_pattern = r'^(Figure|Fig\.)\s*(\d+)[\.\:\s]*(.+)$'
        
        # Reference pattern
        self.reference_pattern = r'^\[\d+\]\s+(.+)$'
    
    def process_document(self, file_path: str, document_id: str) -> Dict[str, Any]:
        """
        Process a PDF document and extract all content.
        
        Args:
            file_path: Path to the PDF file
            document_id: Unique identifier for the document
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        logger.info(f"Processing document: {file_path}")
        
        try:
            # Verify file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, document_id)
            
            # Extract text content
            text_content = self._extract_text(file_path)
            
            # Extract tables
            tables = self._extract_tables(file_path)
            
            # Analyze structure
            structure = self._analyze_structure(text_content, metadata.page_count)
            
            # Combine into result
            result = {
                "metadata": metadata.to_dict(),
                "text_content": text_content,
                "tables": [t.to_dict() for t in tables],
                "structure": structure,
                "document_id": document_id,
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(f"Document processed successfully: {document_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def _extract_metadata(self, file_path: str, document_id: str) -> DocumentMetadata:
        """Extract document metadata."""
        metadata = DocumentMetadata()
        metadata.document_id = document_id
        metadata.file_path = file_path
        metadata.file_name = Path(file_path).name
        
        # Get file info
        file_stat = os.stat(file_path)
        metadata.file_size = file_stat.st_size
        
        try:
            # Try PyMuPDF first (more reliable)
            doc = fitz.open(file_path)
            metadata.page_count = len(doc)
            
            # Extract metadata from PDF
            pdf_metadata = doc.metadata
            if pdf_metadata:
                metadata.title = pdf_metadata.get("title", "") or Path(file_path).stem
                metadata.author = pdf_metadata.get("author", "")
                metadata.creation_date = pdf_metadata.get("creationDate", "")
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"Error extracting metadata with PyMuPDF: {e}")
            # Fallback to pypdf
            try:
                reader = pypdf.PdfReader(file_path)
                metadata.page_count = len(reader.pages)
                metadata.title = reader.metadata.get("/Title", "") or Path(file_path).stem
                metadata.author = reader.metadata.get("/Author", "")
            except Exception as e2:
                logger.error(f"Error with pypdf fallback: {e2}")
        
        return metadata
    
    def _extract_text(self, file_path: str, start_page: int = 0, 
                      end_page: Optional[int] = None) -> str:
        """
        Extract text from PDF using multiple methods for accuracy.
        
        Args:
            file_path: Path to PDF file
            start_page: Start page (0-indexed)
            end_page: End page (inclusive), None for all pages
            
        Returns:
            Extracted text
        """
        text_content = []
        
        try:
            # Method 1: pdfplumber (best for layout preservation)
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                if end_page is None:
                    end_page = min(total_pages, self.max_pages) - 1
                else:
                    end_page = min(end_page, total_pages - 1, self.max_pages - 1)
                
                for page_num in range(start_page, end_page + 1):
                    page = pdf.pages[page_num]
                    
                    # Extract text with layout
                    page_text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        keep_blank_chars=True,
                        use_line_margin=False
                    )
                    
                    if page_text:
                        # Clean up text
                        cleaned_text = self._clean_text(page_text)
                        text_content.append(f"\n--- Page {page_num + 1} ---\n")
                        text_content.append(cleaned_text)
                    else:
                        # Fallback to PyMuPDF
                        logger.info(f"No text extracted from page {page_num + 1} with pdfplumber, trying PyMuPDF")
                        page_text = self._extract_text_pymupdf(file_path, page_num)
                        if page_text:
                            cleaned_text = self._clean_text(page_text)
                            text_content.append(f"\n--- Page {page_num + 1} ---\n")
                            text_content.append(cleaned_text)
        
        except Exception as e:
            logger.error(f"Error with pdfplumber: {e}")
            # Final fallback to PyMuPDF
            try:
                text_content.append(self._extract_text_pymupdf(file_path))
            except Exception as e2:
                logger.error(f"All text extraction methods failed: {e2}")
                raise
        
        return "\n".join(text_content)
    
    def _extract_text_pymupdf(self, file_path: str, page_num: int = 0) -> str:
        """Extract text using PyMuPDF."""
        doc = fitz.open(file_path)
        
        if page_num < len(doc):
            page = doc[page_num]
            text = page.get_text("text")
            doc.close()
            return self._clean_text(text)
        
        doc.close()
        return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip but preserve meaningful whitespace
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(stripped)
        
        # Join with single spaces
        text = ' '.join(cleaned_lines)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Restore paragraph breaks
        text = re.sub(r'(?<=[^\n])\n(?=[^\n])', ' ', text)
        
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Restore list items
        text = re.sub(r'^[\s]*[-•*][\s]+', r'\n• ', text)
        text = re.sub(r'^[\s]*\d+[\.\)][\s]+', r'\n1. ', text)
        
        return text.strip()
    
    def _extract_tables(self, file_path: str) -> List[TableData]:
        """
        Detect and extract tables from PDF.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of extracted table data
        """
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:self.max_pages]):
                    # Find tables
                    found_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(found_tables):
                        table_data = TableData(
                            page_number=page_num + 1,
                            table_number=table_num + 1
                        )
                        
                        if table:
                            # Extract headers (first row usually)
                            if len(table) > 0:
                                table_data.headers = [str(cell) if cell else "" 
                                                      for cell in table[0]]
                            
                            # Extract rows
                            for row in table[1:]:
                                cleaned_row = [str(cell) if cell else "" 
                                               for cell in row]
                                table_data.rows.append(cleaned_row)
                            
                            # Create text representation
                            table_data.text_representation = self._table_to_text(table_data)
                            tables.append(table_data)
            
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")
        
        return tables
    
    def _table_to_text(self, table: TableData) -> str:
        """Convert table data to readable text format."""
        lines = []
        
        # Add caption if exists
        if table.caption:
            lines.append(f"Table {table.table_number}: {table.caption}\n")
        
        # Add headers
        if table.headers:
            header_line = " | ".join(table.headers)
            lines.append(header_line)
            lines.append("-" * len(header_line))
        
        # Add rows
        for row in table.rows:
            row_line = " | ".join(row)
            lines.append(row_line)
        
        return "\n".join(lines)
    
    def _analyze_structure(self, text: str, page_count: int) -> Dict[str, Any]:
        """
        Analyze document structure (sections, references, etc.).
        
        Args:
            text: Full text content
            page_count: Total number of pages
            
        Returns:
            Dictionary with structure analysis
        """
        structure = {
            "sections": [],
            "has_abstract": False,
            "has_references": False,
            "estimated_sections": 0
        }
        
        lines = text.split('\n')
        current_section = {"title": "Introduction", "level": 1, "page": 1}
        
        for line in lines:
            line = line.strip()
            
            # Check for section headings
            if self._is_section_heading(line):
                section = self._parse_section_heading(line)
                if section:
                    structure["sections"].append(section)
                    current_section = section
        
        structure["estimated_sections"] = len(structure["sections"])
        
        # Check for abstract
        if re.search(r'\babstract\b', text[:2000], re.IGNORECASE):
            structure["has_abstract"] = True
        
        # Check for references
        ref_patterns = [
            r'\breferences\b',
            r'\bbibliography\b',
            r'\[\d+\].*\d{4}'  # [1] Author, 2020
        ]
        for pattern in ref_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                structure["has_references"] = True
                break
        
        return structure
    
    def _is_section_heading(self, line: str) -> bool:
        """Check if a line is likely a section heading."""
        # Must be short and not contain typical sentence endings
        if len(line) > 100:
            return False
        
        # Check against patterns
        for pattern in self.section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        # Check if line is all caps or title case and short
        if line.isupper() and 3 < len(line) < 80:
            return True
        
        if line.istitle() and 3 < len(line) < 80:
            return True
        
        return False
    
    def _parse_section_heading(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a section heading."""
        # Remove numbering if present
        cleaned = re.sub(r'^[\d\.]+\s*', '', line)
        cleaned = cleaned.strip()
        
        if not cleaned:
            return None
        
        # Determine level
        level = 1
        if re.match(r'^chapter\s+\d+', line, re.IGNORECASE):
            level = 0
        elif re.match(r'^\d+\.\d+', line):
            level = 2
        elif re.match(r'^\d+[\.\:]\s*\w+', line):
            level = 1
        
        return {
            "title": cleaned,
            "level": level,
            "raw": line
        }
    
    def extract_specific_pages(self, file_path: str, 
                               page_numbers: List[int]) -> Dict[int, str]:
        """
        Extract text from specific pages.
        
        Args:
            file_path: Path to PDF file
            page_numbers: List of page numbers (0-indexed)
            
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        results = {}
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in page_numbers:
                if page_num < len(doc):
                    page = doc[page_num]
                    text = page.get_text("text")
                    results[page_num] = self._clean_text(text)
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error extracting specific pages: {e}")
        
        return results
    
    def find_context(self, text: str, query: str, 
                     context_size: int = 500) -> str:
        """
        Find relevant context around query matches.
        
        Args:
            text: Full text to search
            query: Search query
            context_size: Characters of context on each side
            
        Returns:
            Text with relevant context highlighted
        """
        # Find all matches
        pattern = re.escape(query)
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        if not matches:
            return ""
        
        # Get context around matches
        contexts = []
        for match in matches[:5]:  # Limit to 5 matches
            start = max(0, match.start() - context_size)
            end = min(len(text), match.end() + context_size)
            
            context = text[start:end]
            
            # Add ellipsis if truncated
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
            
            contexts.append(context)
        
        return "\n\n".join(contexts)

