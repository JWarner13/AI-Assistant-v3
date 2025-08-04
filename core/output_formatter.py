"""
Enhanced output formatting utilities for the AI Document Assistant.
Provides multiple output formats and response transformations.
"""

import json
import yaml
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime
import xml.etree.ElementTree as ET

class ResponseFormatter:
    """
    Advanced response formatting with multiple output formats and customization options.
    """
    
    @staticmethod
    def format_json(data: Dict[str, Any], 
                   compact: bool = False, 
                   include_metadata: bool = True) -> str:
        """
        Format response as JSON with options for compactness and metadata inclusion.
        """
        if not include_metadata:
            # Strip metadata for cleaner output
            cleaned_data = {
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
                "confidence_level": data.get("confidence_metrics", {}).get("confidence_level", "UNKNOWN")
            }
            data = cleaned_data
        
        if compact:
            return json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        else:
            return json.dumps(data, indent=4, ensure_ascii=False)
    
    @staticmethod
    def format_yaml(data: Dict[str, Any]) -> str:
        """
        Format response as YAML for better human readability.
        """
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, indent=2)
    
    @staticmethod
    def format_markdown(data: Dict[str, Any]) -> str:
        """
        Format response as Markdown for documentation or reports.
        """
        md_parts = []
        
        # Title
        query = data.get("query", "AI Assistant Response")
        md_parts.append(f"# {query}\n")
        
        # Answer
        answer = data.get("answer", "No answer available")
        md_parts.append(f"## Answer\n\n{answer}\n")
        
        # Sources
        sources = data.get("sources", [])
        if sources:
            md_parts.append("## Sources\n")
            for i, source in enumerate(sources, 1):
                md_parts.append(f"{i}. {source}")
            md_parts.append("")
        
        # Confidence
        confidence_metrics = data.get("confidence_metrics", {})
        if confidence_metrics:
            level = confidence_metrics.get("confidence_level", "UNKNOWN")
            score = confidence_metrics.get("confidence_score", 0)
            md_parts.append(f"## Confidence\n\n**Level:** {level} (Score: {score:.2f})\n")
        
        # Conflicts
        conflicts = data.get("conflicts_detected", [])
        if conflicts:
            md_parts.append("## Conflicts Detected\n")
            for conflict in conflicts:
                desc = conflict.get("description", "Unknown conflict")
                severity = conflict.get("severity", "unknown")
                md_parts.append(f"- **{severity.upper()}:** {desc}")
            md_parts.append("")
        
        # Performance metrics
        perf_metrics = data.get("performance_metrics", {})
        if perf_metrics:
            md_parts.append("## Performance Metrics\n")
            proc_time = perf_metrics.get("processing_time_seconds", "N/A")
            docs_analyzed = perf_metrics.get("documents_analyzed", "N/A")
            md_parts.append(f"- Processing time: {proc_time}s")
            md_parts.append(f"- Documents analyzed: {docs_analyzed}")
            md_parts.append("")
        
        # Reasoning (if present)
        reasoning = data.get("reasoning_trace", "")
        if reasoning:
            md_parts.append("## Reasoning Process\n")
            # Truncate very long reasoning for markdown readability
            if len(reasoning) > 2000:
                reasoning = reasoning[:2000] + "\n\n*[Truncated for brevity]*"
            md_parts.append(f"{reasoning}\n")
        
        return "\n".join(md_parts)
    
    @staticmethod
    def format_xml(data: Dict[str, Any]) -> str:
        """
        Format response as XML for system integrations.
        """
        root = ET.Element("ai_response")
        
        # Query
        query_elem = ET.SubElement(root, "query")
        query_elem.text = data.get("query", "")
        
        # Answer
        answer_elem = ET.SubElement(root, "answer")
        answer_elem.text = data.get("answer", "")
        
        # Sources
        sources_elem = ET.SubElement(root, "sources")
        for source in data.get("sources", []):
            source_elem = ET.SubElement(sources_elem, "source")
            source_elem.text = source
        
        # Confidence
        confidence_metrics = data.get("confidence_metrics", {})
        if confidence_metrics:
            conf_elem = ET.SubElement(root, "confidence")
            conf_elem.set("level", confidence_metrics.get("confidence_level", "UNKNOWN"))
            conf_elem.set("score", str(confidence_metrics.get("confidence_score", 0)))
        
        # Metadata
        metadata_elem = ET.SubElement(root, "metadata")
        metadata = data.get("metadata", {})
        for key, value in metadata.items():
            meta_elem = ET.SubElement(metadata_elem, key)
            meta_elem.text = str(value)
        
        return ET.tostring(root, encoding='unicode', method='xml')
    
    @staticmethod
    def format_csv_summary(data: Dict[str, Any]) -> str:
        """
        Format response as CSV summary (useful for batch processing results).
        """
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Query", "Answer_Preview", "Sources_Count", "Confidence_Level", 
            "Processing_Time", "Conflicts_Count"
        ])
        
        # Data row
        query = data.get("query", "")
        answer_preview = data.get("answer", "")[:100] + "..." if len(data.get("answer", "")) > 100 else data.get("answer", "")
        sources_count = len(data.get("sources", []))
        confidence_level = data.get("confidence_metrics", {}).get("confidence_level", "UNKNOWN")
        processing_time = data.get("performance_metrics", {}).get("processing_time_seconds", "N/A")
        conflicts_count = len(data.get("conflicts_detected", []))
        
        writer.writerow([
            query, answer_preview, sources_count, confidence_level, processing_time, conflicts_count
        ])
        
        return output.getvalue()
    
    @staticmethod
    def format_plain_text(data: Dict[str, Any], 
                         include_sources: bool = True,
                         include_confidence: bool = True) -> str:
        """
        Format response as plain text with customizable sections.
        """
        text_parts = []
        
        # Query
        query = data.get("query", "")
        if query:
            text_parts.append(f"Query: {query}")
            text_parts.append("-" * 50)
        
        # Answer
        answer = data.get("answer", "No answer available")
        text_parts.append(f"Answer: {answer}")
        
        # Sources
        if include_sources:
            sources = data.get("sources", [])
            if sources:
                text_parts.append(f"\nSources ({len(sources)}):")
                for i, source in enumerate(sources, 1):
                    text_parts.append(f"  {i}. {source}")
        
        # Confidence
        if include_confidence:
            confidence_metrics = data.get("confidence_metrics", {})
            if confidence_metrics:
                level = confidence_metrics.get("confidence_level", "UNKNOWN")
                score = confidence_metrics.get("confidence_score", 0)
                text_parts.append(f"\nConfidence: {level} ({score:.2f})")
        
        return "\n".join(text_parts)
    
    @staticmethod
    def format_executive_summary(data: Dict[str, Any]) -> str:
        """
        Format response as executive summary for business use.
        """
        summary_parts = []
        
        # Header
        timestamp = data.get("metadata", {}).get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        summary_parts.append("AI DOCUMENT ANALYSIS - EXECUTIVE SUMMARY")
        summary_parts.append(f"Generated: {timestamp}")
        summary_parts.append("=" * 60)
        
        # Key findings
        answer = data.get("answer", "")
        summary_parts.append("\nKEY FINDINGS:")
        summary_parts.append(answer)
        
        # Data quality assessment
        confidence_metrics = data.get("confidence_metrics", {})
        if confidence_metrics:
            level = confidence_metrics.get("confidence_level", "UNKNOWN")
            factors = confidence_metrics.get("factors", {})
            
            summary_parts.append(f"\nDATA QUALITY: {level}")
            
            if factors.get("multiple_sources"):
                summary_parts.append("✓ Multiple sources consulted")
            if factors.get("comprehensive_content"):
                summary_parts.append("✓ Comprehensive content analyzed")
            if factors.get("conflicts_present"):
                summary_parts.append("⚠ Conflicting information detected")
        
        # Source summary
        sources = data.get("sources", [])
        if sources:
            summary_parts.append(f"\nSOURCES ANALYZED: {len(sources)}")
            for source in sources[:5]:  # Show top 5 sources
                summary_parts.append(f"• {source}")
            if len(sources) > 5:
                summary_parts.append(f"• ... and {len(sources) - 5} more")
        
        # Performance summary
        perf_metrics = data.get("performance_metrics", {})
        if perf_metrics:
            proc_time = perf_metrics.get("processing_time_seconds", "N/A")
            docs_analyzed = perf_metrics.get("documents_analyzed", "N/A")
            summary_parts.append(f"\nPROCESSING: {proc_time}s | {docs_analyzed} documents analyzed")
        
        return "\n".join(summary_parts)

class BatchFormatter:
    """
    Specialized formatter for batch processing results.
    """
    
    @staticmethod
    def format_batch_summary(batch_results: Dict[str, Any], format_type: str = "json") -> str:
        """
        Format batch processing results in various formats.
        """
        if format_type == "csv":
            return BatchFormatter._format_batch_csv(batch_results)
        elif format_type == "markdown":
            return BatchFormatter._format_batch_markdown(batch_results)
        else:
            return ResponseFormatter.format_json(batch_results)
    
    @staticmethod
    def _format_batch_csv(batch_results: Dict[str, Any]) -> str:
        """Format batch results as CSV."""
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            "Query", "Status", "Answer_Preview", "Sources_Count", 
            "Confidence", "Processing_Time", "Conflicts"
        ])
        
        # Process each result
        results = batch_results.get("results", {})
        for query, result in results.items():
            status = "Success" if "error" not in result else "Error"
            
            if status == "Success":
                answer_preview = result.get("answer", "")[:100] + "..." if len(result.get("answer", "")) > 100 else result.get("answer", "")
                sources_count = len(result.get("sources", []))
                confidence = result.get("confidence_metrics", {}).get("confidence_level", "UNKNOWN")
                proc_time = result.get("performance_metrics", {}).get("processing_time_seconds", "N/A")
                conflicts = len(result.get("conflicts_detected", []))
            else:
                answer_preview = f"ERROR: {result.get('error', 'Unknown error')}"
                sources_count = 0
                confidence = "N/A"
                proc_time = "N/A"
                conflicts = 0
            
            writer.writerow([
                query, status, answer_preview, sources_count, 
                confidence, proc_time, conflicts
            ])
        
        return output.getvalue()
    
    @staticmethod
    def _format_batch_markdown(batch_results: Dict[str, Any]) -> str:
        """Format batch results as Markdown report."""
        md_parts = []
        
        # Title
        md_parts.append("# Batch Processing Report\n")
        
        # Summary
        batch_summary = batch_results.get("batch_summary", {})
        total_queries = batch_summary.get("total_queries", 0)
        successful = batch_summary.get("successful_queries", 0)
        failed = batch_summary.get("failed_queries", 0)
        total_time = batch_summary.get("total_processing_time", 0)
        
        md_parts.append("## Summary\n")
        md_parts.append(f"- **Total Queries:** {total_queries}")
        md_parts.append(f"- **Successful:** {successful}")
        md_parts.append(f"- **Failed:** {failed}")
        md_parts.append(f"- **Success Rate:** {(successful/total_queries*100):.1f}%" if total_queries > 0 else "- **Success Rate:** N/A")
        md_parts.append(f"- **Total Processing Time:** {total_time:.2f}s")
        md_parts.append(f"- **Average Time per Query:** {(total_time/total_queries):.2f}s\n" if total_queries > 0 else "- **Average Time per Query:** N/A\n")
        
        # Individual results
        md_parts.append("## Individual Results\n")
        
        results = batch_results.get("results", {})
        for i, (query, result) in enumerate(results.items(), 1):
            md_parts.append(f"### {i}. {query}\n")
            
            if "error" not in result:
                answer = result.get("answer", "No answer")
                sources = result.get("sources", [])
                confidence = result.get("confidence_metrics", {}).get("confidence_level", "UNKNOWN")
                
                md_parts.append(f"**Answer:** {answer}\n")
                md_parts.append(f"**Sources:** {', '.join(sources) if sources else 'None'}")
                md_parts.append(f"**Confidence:** {confidence}\n")
            else:
                error_msg = result.get("error", "Unknown error")
                md_parts.append(f"**Status:** ❌ Error")
                md_parts.append(f"**Error:** {error_msg}\n")
        
        return "\n".join(md_parts)

# Legacy compatibility function
def format_output(answer: str, documents: List[Any]) -> str:
    """
    Legacy compatibility function - maintains backward compatibility.
    """
    return json.dumps({
        "answer": answer,
        "sources": list(set(doc.metadata['source'] for doc in documents))
    }, indent=4)

# Convenience functions for common use cases
def format_for_api(data: Dict[str, Any]) -> str:
    """Format response optimized for API consumption."""
    return ResponseFormatter.format_json(data, compact=True, include_metadata=False)

def format_for_human(data: Dict[str, Any]) -> str:
    """Format response optimized for human readability."""
    return ResponseFormatter.format_plain_text(data, include_sources=True, include_confidence=True)

def format_for_report(data: Dict[str, Any]) -> str:
    """Format response optimized for reports and documentation."""
    return ResponseFormatter.format_markdown(data)

def format_for_business(data: Dict[str, Any]) -> str:
    """Format response optimized for business stakeholders."""
    return ResponseFormatter.format_executive_summary(data)