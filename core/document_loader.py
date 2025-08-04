import os
import fitz  # PyMuPDF
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from core.logger import log

def extract_text_from_pdf(filepath: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text and metadata from PDF file.
    
    Args:
        filepath: Path to PDF file
        
    Returns:
        Tuple of (extracted_text, metadata)
    """
    try:
        doc = fitz.open(filepath)
        
        # Extract text from all pages
        text_parts = []
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        
        full_text = "\n\n".join(text_parts)
        
        # Extract metadata
        metadata = {
            "page_count": len(doc),
            "file_size": os.path.getsize(filepath),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", "")
        }
        
        doc.close()
        
        log("PDF processed", extra={
            "filepath": filepath,
            "pages": metadata["page_count"],
            "text_length": len(full_text)
        })
        
        return full_text, metadata
        
    except Exception as e:
        log("Error processing PDF", extra={"filepath": filepath, "error": str(e)})
        return "", {}

def extract_text_from_txt(filepath: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text and metadata from text file.
    
    Args:
        filepath: Path to text file
        
    Returns:
        Tuple of (text_content, metadata)
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        text_content = ""
        encoding_used = ""
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    text_content = f.read()
                    encoding_used = encoding
                    break
            except UnicodeDecodeError:
                continue
        
        if not text_content:
            raise ValueError("Could not decode file with any supported encoding")
        
        # Extract metadata
        file_stats = os.stat(filepath)
        metadata = {
            "file_size": file_stats.st_size,
            "creation_date": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modification_date": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "encoding": encoding_used,
            "line_count": text_content.count('\n') + 1,
            "word_count": len(text_content.split()),
            "character_count": len(text_content)
        }
        
        log("Text file processed", extra={
            "filepath": filepath,
            "encoding": encoding_used,
            "text_length": len(text_content)
        })
        
        return text_content, metadata
        
    except Exception as e:
        log("Error processing text file", extra={"filepath": filepath, "error": str(e)})
        return "", {}

def split_into_chunks(text: str, 
                     chunk_size: int = 500, 
                     overlap: int = 100,
                     preserve_paragraphs: bool = True,
                     min_chunk_size: int = 50) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with enhanced splitting logic.
    
    Args:
        text: Text to split
        chunk_size: Target size for each chunk (in words)
        overlap: Number of words to overlap between chunks
        preserve_paragraphs: Whether to try to keep paragraphs intact
        min_chunk_size: Minimum chunk size (in words)
        
    Returns:
        List of chunk dictionaries with content and metadata
    """
    if not text.strip():
        return []
    
    chunks = []
    
    try:
        if preserve_paragraphs:
            # Try to split on paragraph boundaries first
            paragraphs = text.split('\n\n')
            current_chunk = ""
            current_word_count = 0
            chunk_id = 0
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                
                para_words = paragraph.split()
                para_word_count = len(para_words)
                
                # If paragraph alone is too big, split it
                if para_word_count > chunk_size:
                    # First, add current chunk if it exists
                    if current_chunk and current_word_count >= min_chunk_size:
                        chunks.append(_create_chunk(current_chunk, chunk_id, current_word_count))
                        chunk_id += 1
                        current_chunk = ""
                        current_word_count = 0
                    
                    # Split the large paragraph
                    para_chunks = _split_large_text(paragraph, chunk_size, overlap, min_chunk_size)
                    for chunk_text in para_chunks:
                        word_count = len(chunk_text.split())
                        chunks.append(_create_chunk(chunk_text, chunk_id, word_count))
                        chunk_id += 1
                
                # If adding this paragraph would exceed chunk size
                elif current_word_count + para_word_count > chunk_size:
                    # Save current chunk if it's big enough
                    if current_chunk and current_word_count >= min_chunk_size:
                        chunks.append(_create_chunk(current_chunk, chunk_id, current_word_count))
                        chunk_id += 1
                    
                    # Start new chunk with this paragraph
                    current_chunk = paragraph
                    current_word_count = para_word_count
                
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                    current_word_count += para_word_count
            
            # Add remaining chunk
            if current_chunk and current_word_count >= min_chunk_size:
                chunks.append(_create_chunk(current_chunk, chunk_id, current_word_count))
        
        else:
            # Simple word-based splitting
            words = text.split()
            chunk_texts = []
            
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) >= min_chunk_size:
                    chunk_text = " ".join(chunk_words)
                    chunk_texts.append(chunk_text)
            
            for i, chunk_text in enumerate(chunk_texts):
                word_count = len(chunk_text.split())
                chunks.append(_create_chunk(chunk_text, i, word_count))
        
        log("Text chunking completed", extra={
            "original_length": len(text),
            "chunks_created": len(chunks),
            "avg_chunk_size": sum(c["word_count"] for c in chunks) / len(chunks) if chunks else 0
        })
        
        return chunks
        
    except Exception as e:
        log("Error in text chunking", extra={"error": str(e)})
        # Fallback to simple splitting
        words = text.split()
        fallback_chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            fallback_chunks.append(_create_chunk(chunk_text, i // chunk_size, len(chunk_text.split())))
        return fallback_chunks

def _create_chunk(text: str, chunk_id: int, word_count: int) -> Dict[str, Any]:
    """Create a chunk dictionary with metadata."""
    return {
        "content": text,
        "chunk_id": chunk_id,
        "word_count": word_count,
        "character_count": len(text),
        "content_hash": hashlib.md5(text.encode()).hexdigest(),
        "preview": text[:100] + "..." if len(text) > 100 else text
    }

def _split_large_text(text: str, chunk_size: int, overlap: int, min_chunk_size: int) -> List[str]:
    """Split large text that doesn't fit in a single chunk."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) >= min_chunk_size:
            chunks.append(" ".join(chunk_words))
    
    return chunks

def load_documents(data_dir: str = "data", 
                  supported_extensions: Optional[List[str]] = None,
                  chunk_size: int = 500,
                  overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Load and process documents from a directory with enhanced error handling and metadata.
    
    Args:
        data_dir: Directory containing documents
        supported_extensions: File extensions to process (default: ['.pdf', '.txt'])
        chunk_size: Size of text chunks in words
        overlap: Overlap between chunks in words
        
    Returns:
        List of document chunks with metadata
    """
    if supported_extensions is None:
        supported_extensions = ['.pdf', '.txt']
    
    documents = []
    processing_stats = {
        "files_found": 0,
        "files_processed": 0,
        "files_failed": 0,
        "total_chunks": 0,
        "total_text_length": 0
    }
    
    try:
        if not os.path.exists(data_dir):
            log("Data directory not found", extra={"data_dir": data_dir})
            return []
        
        # Get all supported files
        data_path = Path(data_dir)
        all_files = []
        
        for ext in supported_extensions:
            pattern = f"*{ext}"
            all_files.extend(data_path.glob(pattern))
        
        processing_stats["files_found"] = len(all_files)
        
        if not all_files:
            log("No supported files found", extra={
                "data_dir": data_dir,
                "supported_extensions": supported_extensions
            })
            return []
        
        log("Starting document loading", extra={
            "data_dir": data_dir,
            "files_found": len(all_files),
            "supported_extensions": supported_extensions
        })
        
        for filepath in all_files:
            try:
                filename = filepath.name
                file_extension = filepath.suffix.lower()
                
                log("Processing file", extra={"filename": filename})
                
                # Extract text and metadata based on file type
                if file_extension == '.pdf':
                    text, file_metadata = extract_text_from_pdf(str(filepath))
                elif file_extension == '.txt':
                    text, file_metadata = extract_text_from_txt(str(filepath))
                else:
                    log("Unsupported file type", extra={"filename": filename, "extension": file_extension})
                    continue
                
                if not text.strip():
                    log("No text extracted from file", extra={"filename": filename})
                    processing_stats["files_failed"] += 1
                    continue
                
                # Split into chunks
                chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
                
                if not chunks:
                    log("No chunks created from file", extra={"filename": filename})
                    processing_stats["files_failed"] += 1
                    continue
                
                # Add file metadata to each chunk
                for i, chunk in enumerate(chunks):
                    # Calculate global document position
                    global_position = len(documents) + i
                    
                    document = {
                        "content": chunk["content"],
                        "source": filename,
                        "file_type": file_extension,
                        "chunk_metadata": {
                            "chunk_id": chunk["chunk_id"],
                            "word_count": chunk["word_count"],
                            "character_count": chunk["character_count"],
                            "content_hash": chunk["content_hash"],
                            "preview": chunk["preview"],
                            "global_position": global_position
                        },
                        "file_metadata": file_metadata,
                        "processing_metadata": {
                            "processed_at": datetime.now().isoformat(),
                            "chunk_size_config": chunk_size,
                            "overlap_config": overlap,
                            "data_directory": data_dir
                        }
                    }
                    documents.append(document)
                
                processing_stats["files_processed"] += 1
                processing_stats["total_chunks"] += len(chunks)
                processing_stats["total_text_length"] += len(text)
                
                log("File processed successfully", extra={
                    "filename": filename,
                    "chunks_created": len(chunks),
                    "text_length": len(text)
                })
                
            except Exception as e:
                processing_stats["files_failed"] += 1
                log("Error processing file", extra={
                    "filename": str(filepath),
                    "error": str(e)
                })
                continue
        
        # Log final statistics
        log("Document loading completed", extra=processing_stats)
        
        # Validate that we have documents
        if not documents:
            log("No documents were successfully processed", extra={
                "data_dir": data_dir,
                "files_attempted": processing_stats["files_found"]
            })
        
        return documents
        
    except Exception as e:
        log("Critical error in document loading", extra={
            "data_dir": data_dir,
            "error": str(e)
        })
        return []

def get_document_summary(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary of loaded documents.
    
    Args:
        documents: List of document chunks
        
    Returns:
        Summary statistics
    """
    if not documents:
        return {"status": "no_documents"}
    
    # Aggregate statistics
    sources = set()
    file_types = set()
    total_words = 0
    total_chars = 0
    chunk_sizes = []
    
    for doc in documents:
        sources.add(doc["source"])
        file_types.add(doc["file_type"])
        
        chunk_meta = doc.get("chunk_metadata", {})
        total_words += chunk_meta.get("word_count", 0)
        total_chars += chunk_meta.get("character_count", 0)
        chunk_sizes.append(chunk_meta.get("word_count", 0))
    
    # Calculate statistics
    avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
    max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
    
    summary = {
        "total_documents": len(documents),
        "unique_sources": len(sources),
        "file_types": list(file_types),
        "sources": list(sources),
        "content_statistics": {
            "total_words": total_words,
            "total_characters": total_chars,
            "average_chunk_size": round(avg_chunk_size, 1),
            "min_chunk_size": min_chunk_size,
            "max_chunk_size": max_chunk_size
        },
        "processing_info": {
            "chunk_count": len(documents),
            "avg_words_per_chunk": round(avg_chunk_size, 1)
        }
    }
    
    return summary

def validate_documents(documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate the structure and content of loaded documents.
    
    Args:
        documents: List of document chunks
        
    Returns:
        Validation results
    """
    validation_results = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "document_count": len(documents)
    }
    
    if not documents:
        validation_results["valid"] = False
        validation_results["issues"].append("No documents provided")
        return validation_results
    
    required_fields = ["content", "source"]
    empty_content_count = 0
    missing_metadata_count = 0
    
    for i, doc in enumerate(documents):
        # Check required fields
        for field in required_fields:
            if field not in doc:
                validation_results["issues"].append(f"Document {i}: Missing required field '{field}'")
                validation_results["valid"] = False
        
        # Check content quality
        if "content" in doc:
            content = doc["content"]
            if not content or not content.strip():
                empty_content_count += 1
            elif len(content.split()) < 10:
                validation_results["warnings"].append(f"Document {i}: Very short content ({len(content.split())} words)")
        
        # Check metadata presence
        if "chunk_metadata" not in doc and "file_metadata" not in doc:
            missing_metadata_count += 1
    
    # Add summary warnings
    if empty_content_count > 0:
        validation_results["warnings"].append(f"{empty_content_count} documents have empty content")
    
    if missing_metadata_count > 0:
        validation_results["warnings"].append(f"{missing_metadata_count} documents missing metadata")
    
    # Check for duplicate content
    content_hashes = []
    for doc in documents:
        if "chunk_metadata" in doc and "content_hash" in doc["chunk_metadata"]:
            content_hashes.append(doc["chunk_metadata"]["content_hash"])
    
    if len(content_hashes) != len(set(content_hashes)):
        duplicate_count = len(content_hashes) - len(set(content_hashes))
        validation_results["warnings"].append(f"{duplicate_count} duplicate content chunks detected")
    
    return validation_results