import hashlib
import json
import time
from typing import List, Dict, Any
from core.cache import CacheManager
from core.document_loader import load_documents
from core.embedding import get_embedding_model
from core.index import get_index_manager  
from core.llm import get_llm
from core.output_formatter import format_output
from core.logger import log
from core.reasoning import explain_reasoning, detect_conflicts

def _determine_optimal_k(query: str, total_docs: int) -> int:
    """Determine optimal number of documents to retrieve."""
    base_k = 5
    if len(query.split()) > 15:
        base_k = 8
    elif "compare" in query.lower() or "difference" in query.lower():
        base_k = 6
    return min(base_k, total_docs)

def _generate_cache_key(text: str, model: str) -> str:
    """
    Generate a unique cache key based on text and model name.
    This is used for caching embeddings or query results.
    """
    content = f"embedding:{model}:{text}"
    return hashlib.md5(content.encode()).hexdigest()

def run_assistant(user_query: str, 
                 enable_reasoning_trace: bool = True,
                 data_dir: str = "data",
                 use_cache: bool = True) -> str:
    """
    Enhanced RAG assistant with multi-document reasoning and conflict detection.
    """
    try:
        # Load and process documents
        documents = load_documents(data_dir)
        log("Documents loaded", extra={"count": len(documents)})

        if not documents:
            return json.dumps({
                "error": "No documents found",
                "answer": "I couldn't find any documents to analyze. Please add PDF or TXT files to the data directory.",
                "sources": [],
                "documents_analyzed": 0
            }, indent=4)

        # FIX: Use IndexManager instead of direct functions
        embedding_model = get_embedding_model()
        index_manager = get_index_manager()
        
        # Build semantic search index using IndexManager
        index = index_manager.build_index(
            documents, 
            embedding_model.model,  # Pass the underlying model
            "main_index",
            force_rebuild=False
        )
        log("Index built", extra={})

        # Retrieve relevant documents using IndexManager
        retrieved_docs = index_manager.search_index(
            index, 
            user_query, 
            k = min(8, len(documents)),
            use_cache=use_cache
        )
        log("Documents retrieved", extra={"count": len(retrieved_docs)})
        
        if not retrieved_docs:
            return json.dumps({
                "answer": "I couldn't find relevant information to answer your question in the available documents.",
                "sources": [],
                "documents_analyzed": 0,
                "reasoning_trace": "No relevant documents found for the query.",
                "conflicts_detected": []
            }, indent=4)
        
        # Group documents by source for conflict detection
        docs_by_source = {}
        for doc in retrieved_docs:
            source = doc.metadata.get('source', 'unknown')
            if source not in docs_by_source:
                docs_by_source[source] = []
            docs_by_source[source].append(doc.page_content)
        
        # Detect potential conflicts
        conflicts = detect_conflicts(retrieved_docs, user_query)
        
        # Enhanced context preparation with source attribution
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get('source', 'unknown')
            context_parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Enhanced prompt for complex reasoning
        enhanced_prompt = f"""
You are an AI assistant analyzing technical documents. Answer the following question using the provided context, following these guidelines:

1. REASONING: Provide step-by-step reasoning that shows how you arrived at your answer
2. SOURCES: Reference specific sources when making claims
3. CONFLICTS: If you find conflicting information, acknowledge it and explain how you resolved it
4. SYNTHESIS: When information comes from multiple sources, explain how you combined them

Context from documents:
{context}

Conflicts detected: {json.dumps(conflicts, indent=2) if conflicts else "None"}

Question: {user_query}

Provide your response in the following structure:
- **Answer**: [Your main answer]
- **Reasoning**: [Step-by-step explanation of your reasoning process]
- **Sources Used**: [List the specific sources that contributed to your answer]
- **Confidence**: [High/Medium/Low based on evidence quality]
- **Limitations**: [Any caveats or areas where more information would be helpful]
"""

        llm = get_llm()
        response = llm.invoke(enhanced_prompt)
        
        # Generate reasoning explanation if requested
        reasoning_trace = ""
        if enable_reasoning_trace:
            try:
                reasoning_trace = explain_reasoning(user_query, retrieved_docs, response)
            except Exception as e:
                log("Error generating reasoning trace", extra={"error": str(e)})
                reasoning_trace = f"Reasoning trace generation failed: {str(e)}"
        
        # Enhanced output formatting
        result = {
            "answer": response,
            "sources": list(set(doc.metadata.get('source', 'unknown') for doc in retrieved_docs)),
            "reasoning_trace": reasoning_trace,
            "conflicts_detected": conflicts,
            "documents_analyzed": len(retrieved_docs),
            "confidence_indicators": {
                "source_diversity": len(set(doc.metadata.get('source', 'unknown') for doc in retrieved_docs)),
                "total_chunks": len(retrieved_docs)
            },
            "performance_metrics": {
                "unique_sources": len(set(doc.metadata.get('source', 'unknown') for doc in retrieved_docs)),
                "processing_time_seconds": time.time() - time.time()  # Will be updated with actual timing
            },
            "metadata": {
                "query": user_query,
                "reasoning_enabled": enable_reasoning_trace,
                "cache_used": use_cache
            }
        }
        
        log("Response generated", extra={
            "query": user_query,
            "sources_used": len(result["sources"]),
            "conflicts": len(conflicts)
        })
        
        return json.dumps(result, indent=4, ensure_ascii=False)
    
    except Exception as e:
        log("Error in run_assistant", extra={"error": str(e)})
        error_result = {
            "error": str(e),
            "answer": f"I encountered an error while processing your query: {str(e)}",
            "sources": [],
            "documents_analyzed": 0,
            "reasoning_trace": "",
            "conflicts_detected": []
        }
        return json.dumps(error_result, indent=4, ensure_ascii=False)

def run_batch_queries(queries: List[str], 
                     data_dir: str = "data",
                     enable_reasoning: bool = False,
                     use_cache: bool = True) -> Dict[str, Any]:
    """
    Process multiple queries efficiently with caching.
    """
    results = {}
    batch_summary = {
        "total_queries": len(queries),
        "successful_queries": 0,
        "failed_queries": 0,
        "total_processing_time": 0
    }
    
    try:
        # Load documents once for all queries
        documents = load_documents(data_dir)
        
        # FIX: Use IndexManager for batch processing
        embedding_model = get_embedding_model()
        index_manager = get_index_manager()
        
        # Build index once for all queries
        index = index_manager.build_index(
            documents, 
            embedding_model.model,
            "batch_index",
            force_rebuild=False
        )
        
        log(f"Processing batch of {len(queries)} queries")
        
        start_time = time.time()
        
        for i, query in enumerate(queries):
            log(f"Processing query {i+1}/{len(queries)}: {query}")
            try:
                query_start = time.time()
                
                # Use IndexManager for search
                retrieved_docs = index_manager.search_index(
                    index, 
                    query, 
                    k=8,
                    use_cache=use_cache
                )
                
                if not retrieved_docs:
                    results[query] = {
                        "answer": "No relevant information found for this query.",
                        "sources": [],
                        "documents_analyzed": 0,
                        "error": "No relevant documents"
                    }
                    batch_summary["failed_queries"] += 1
                    continue
                
                # Simple processing for batch mode
                context_parts = []
                for j, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get('source', 'unknown')
                    context_parts.append(f"[Source {j+1}: {source}]\n{doc.page_content}")
                
                context = "\n\n---\n\n".join(context_parts)
                
                prompt = f"""Based on the following context, answer this question: {query}

Context:
{context}

Provide a clear, concise answer:"""
                
                llm = get_llm()
                response = llm.invoke(prompt)
                
                query_time = time.time() - query_start
                
                results[query] = {
                    "answer": response,
                    "sources": list(set(doc.metadata.get('source', 'unknown') for doc in retrieved_docs)),
                    "documents_analyzed": len(retrieved_docs),
                    "processing_time": query_time
                }
                
                batch_summary["successful_queries"] += 1
                
            except Exception as e:
                log(f"Error processing query '{query}': {str(e)}")
                results[query] = {
                    "error": str(e),
                    "answer": f"Error processing query: {str(e)}",
                    "sources": [],
                    "documents_analyzed": 0
                }
                batch_summary["failed_queries"] += 1
        
        batch_summary["total_processing_time"] = time.time() - start_time
        
    except Exception as e:
        log(f"Critical error in batch processing: {str(e)}")
        batch_summary["error"] = str(e)
    
    return {
        "results": results,
        "batch_summary": batch_summary
    }

# Backward compatibility functions (if needed by other parts of the system)
def build_index(documents, embeddings):
    """Backward compatibility wrapper."""
    index_manager = get_index_manager()
    return index_manager.build_index(documents, embeddings, "compatibility_index")

def search_index(index, query, k=5):
    """Backward compatibility wrapper."""
    index_manager = get_index_manager()
    return index_manager.search_index(index, query, k)