from typing import List, Dict, Any, Optional, Tuple
import re
import json
from collections import defaultdict, Counter
from core.llm import get_llm
from core.logger import log

def explain_reasoning(query: str, docs: List[Any], response: str) -> str:
    """
    Generate detailed reasoning explanation for the answer using chain-of-thought.
    
    Args:
        query: Original user question
        docs: Retrieved documents used for the answer
        response: Generated response
    
    Returns:
        Detailed reasoning explanation
    """
    try:
        # Extract key information from documents
        doc_summaries = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            content_preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            
            # Extract key phrases/entities from the document
            key_phrases = _extract_key_phrases(doc.page_content, query)
            
            doc_summaries.append({
                "document_id": i + 1,
                "source": source,
                "content_preview": content_preview,
                "key_phrases": key_phrases,
                "relevance_score": _calculate_relevance_score(doc.page_content, query)
            })
        
        # Build reasoning chain
        reasoning_prompt = f"""
You are an AI reasoning analyst. Explain the step-by-step reasoning process used to answer a question based on provided documents.

ORIGINAL QUESTION: {query}

FINAL ANSWER PROVIDED: {response}

DOCUMENT ANALYSIS:
{json.dumps(doc_summaries, indent=2)}

Your task is to create a clear reasoning chain that shows:

1. **INFORMATION EXTRACTION**: What key information was identified in each document?
2. **RELEVANCE ASSESSMENT**: How did each piece of information relate to the question?
3. **SYNTHESIS PROCESS**: How were different pieces of information combined?
4. **LOGICAL STEPS**: What logical reasoning steps led to the conclusion?
5. **EVIDENCE EVALUATION**: How strong is the evidence for each claim?
6. **UNCERTAINTY HANDLING**: What assumptions were made or what information was missing?

Provide a clear, step-by-step explanation that someone could follow to understand how the answer was derived.

FORMAT YOUR RESPONSE AS:
## Reasoning Process

### Step 1: Information Gathering
[Explain what information was extracted from each source]

### Step 2: Relevance Analysis  
[Explain how each piece of information relates to the question]

### Step 3: Synthesis and Integration
[Explain how information was combined across sources]

### Step 4: Logical Reasoning
[Explain the logical steps taken to reach the conclusion]

### Step 5: Confidence Assessment
[Explain the strength of evidence and any limitations]
"""
        
        llm = get_llm()
        reasoning_explanation = llm.invoke(reasoning_prompt)
        
        log("Reasoning explanation generated", extra={
            "query": query,
            "docs_analyzed": len(docs),
            "explanation_length": len(reasoning_explanation)
        })
        
        return reasoning_explanation
        
    except Exception as e:
        log("Error generating reasoning explanation", extra={"error": str(e)})
        return f"Could not generate detailed reasoning explanation due to: {str(e)}"

def detect_conflicts(docs: List[Any], query: str) -> List[Dict[str, Any]]:
    """
    Enhanced conflict detection using multiple strategies.
    
    Args:
        docs: Retrieved documents
        query: User question for context
    
    Returns:
        List of detected conflicts with detailed information
    """
    conflicts = []
    
    try:
        # Group documents by source for comparison
        content_by_source = defaultdict(list)
        for doc in docs:
            source = doc.metadata.get('source', 'unknown')
            content_by_source[source].append(doc.page_content)
        
        # Strategy 1: LLM-based conflict detection for multiple sources
        if len(content_by_source) > 1:
            llm_conflicts = _detect_conflicts_with_llm(content_by_source, query)
            conflicts.extend(llm_conflicts)
        
        # Strategy 2: Heuristic-based conflict detection
        heuristic_conflicts = _heuristic_conflict_detection(docs, query)
        conflicts.extend(heuristic_conflicts)
        
        # Strategy 3: Factual contradiction detection
        factual_conflicts = _detect_factual_contradictions(docs)
        conflicts.extend(factual_conflicts)
        
        # Remove duplicates and merge similar conflicts
        conflicts = _merge_similar_conflicts(conflicts)
        
        log("Conflict detection completed", extra={
            "total_conflicts": len(conflicts),
            "sources_analyzed": len(content_by_source)
        })
        
        return conflicts
        
    except Exception as e:
        log("Error in conflict detection", extra={"error": str(e)})
        return []

def perform_multi_hop_reasoning(query: str, docs: List[Any]) -> Dict[str, Any]:
    """
    Perform sophisticated multi-hop reasoning analysis.
    
    Args:
        query: User question
        docs: Retrieved documents
    
    Returns:
        Analysis of reasoning requirements and strategy
    """
    try:
        # Analyze query complexity and reasoning requirements
        reasoning_requirements = _analyze_reasoning_requirements(query)
        
        # Map document relationships
        doc_relationships = _map_document_relationships(docs)
        
        # Identify reasoning chains
        reasoning_chains = _identify_reasoning_chains(query, docs, reasoning_requirements)
        
        # Determine optimal reasoning strategy
        strategy = _determine_reasoning_strategy(reasoning_requirements, doc_relationships)
        
        analysis = {
            "reasoning_type": reasoning_requirements["primary_type"],
            "complexity_level": reasoning_requirements["complexity"],
            "reasoning_strategy": strategy,
            "document_relationships": doc_relationships,
            "reasoning_chains": reasoning_chains,
            "multi_hop_required": reasoning_requirements["multi_hop_required"],
            "synthesis_points": reasoning_requirements["synthesis_points"]
        }
        
        log("Multi-hop reasoning analysis completed", extra={
            "reasoning_type": analysis["reasoning_type"],
            "complexity": analysis["complexity_level"],
            "multi_hop": analysis["multi_hop_required"]
        })
        
        return analysis
        
    except Exception as e:
        log("Error in multi-hop reasoning analysis", extra={"error": str(e)})
        return {"reasoning_type": "error", "error": str(e)}

# Helper functions for reasoning analysis

def _extract_key_phrases(text: str, query: str) -> List[str]:
    """Extract key phrases relevant to the query from text."""
    # Simple keyword extraction - could be enhanced with NLP libraries
    query_words = set(query.lower().split())
    text_words = text.lower().split()
    
    # Find phrases that contain query words
    key_phrases = []
    for i, word in enumerate(text_words):
        if word in query_words and i < len(text_words) - 2:
            phrase = " ".join(text_words[i:i+3])
            key_phrases.append(phrase)
    
    return list(set(key_phrases))[:5]  # Return top 5 unique phrases

def _calculate_relevance_score(text: str, query: str) -> float:
    """Calculate relevance score between text and query."""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    
    if not query_words:
        return 0.0
    
    overlap = len(query_words.intersection(text_words))
    return overlap / len(query_words)

def _detect_conflicts_with_llm(content_by_source: Dict[str, List[str]], query: str) -> List[Dict[str, Any]]:
    """Use LLM to detect sophisticated conflicts between sources."""
    conflicts = []
    
    try:
        # Prepare sources for comparison
        sources_text = []
        source_names = list(content_by_source.keys())
        
        for source, contents in content_by_source.items():
            combined_content = " ".join(contents)[:1500]  # Limit for token efficiency
            sources_text.append(f"=== SOURCE: {source} ===\n{combined_content}")
        
        conflict_prompt = f"""
Analyze the following sources for conflicts, contradictions, or inconsistencies related to: "{query}"

{chr(10).join(sources_text)}

Look for:
1. DIRECT CONTRADICTIONS: Opposing factual statements
2. METHODOLOGICAL DIFFERENCES: Different approaches to the same problem
3. INCONSISTENT DATA: Conflicting numbers, dates, or measurements  
4. CONFLICTING RECOMMENDATIONS: Different suggested actions
5. PERSPECTIVE DIFFERENCES: Different interpretations of the same information

Respond ONLY with valid JSON in this exact format:
{{
    "conflicts_found": true/false,
    "conflicts": [
        {{
            "type": "contradiction|methodological|data|recommendation|perspective",
            "description": "Clear description of the conflict",
            "sources_involved": ["source1", "source2"],
            "severity": "high|medium|low",
            "specific_claims": ["claim from source 1", "conflicting claim from source 2"]
        }}
    ]
}}

If no conflicts are found, return: {{"conflicts_found": false, "conflicts": []}}
"""
        
        llm = get_llm()
        response = llm.invoke(conflict_prompt)
        
        # Parse JSON response
        try:
            conflict_data = json.loads(response)
            if conflict_data.get("conflicts_found", False):
                for conflict in conflict_data.get("conflicts", []):
                    conflicts.append({
                        "detection_method": "llm_analysis",
                        "type": conflict.get("type", "unknown"),
                        "description": conflict.get("description", ""),
                        "sources_involved": conflict.get("sources_involved", []),
                        "severity": conflict.get("severity", "medium"),
                        "specific_claims": conflict.get("specific_claims", [])
                    })
        except json.JSONDecodeError:
            # Fallback: treat any substantive response as indicating conflicts
            if "conflict" in response.lower() or "contradiction" in response.lower():
                conflicts.append({
                    "detection_method": "llm_analysis",
                    "type": "general_conflict",
                    "description": response[:200] + "..." if len(response) > 200 else response,
                    "sources_involved": source_names,
                    "severity": "medium"
                })
    
    except Exception as e:
        log("Error in LLM conflict detection", extra={"error": str(e)})
    
    return conflicts

def _heuristic_conflict_detection(docs: List[Any], query: str) -> List[Dict[str, Any]]:
    """Detect conflicts using heuristic patterns."""
    conflicts = []
    
    # Conflict indicator patterns
    conflict_patterns = [
        (r'\b(however|but|although|while|whereas)\b', "contradiction"),
        (r'\b(on the other hand|in contrast|alternatively)\b', "alternative_view"),
        (r'\b(different|various|multiple)\s+(?:approaches|methods|ways)\b', "methodological_difference"),
        (r'\b(?:not|no|never|cannot)\b.*\b(?:accurate|correct|true)\b', "accuracy_dispute"),
        (r'\b(?:instead|rather than|unlike)\b', "opposing_approach")
    ]
    
    # Number/date conflicts
    number_pattern = r'\b(\d+(?:\.\d+)?(?:%|percent|million|billion|thousand)?)\b'
    date_pattern = r'\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b'
    
    sources_with_indicators = defaultdict(list)
    sources_with_numbers = defaultdict(list)
    
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        content = doc.page_content.lower()
        
        # Check for conflict indicators
        for pattern, conflict_type in conflict_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                sources_with_indicators[conflict_type].append({
                    "source": source,
                    "matches": matches
                })
        
        # Extract numbers for comparison
        numbers = re.findall(number_pattern, content)
        if numbers:
            sources_with_numbers[source].extend(numbers)
    
    # Generate conflict reports
    for conflict_type, sources in sources_with_indicators.items():
        if len(set(item["source"] for item in sources)) > 1:
            conflicts.append({
                "detection_method": "heuristic_pattern",
                "type": conflict_type,
                "description": f"Multiple sources contain {conflict_type} indicators",
                "sources_involved": list(set(item["source"] for item in sources)),
                "severity": "low",
                "evidence": [item["matches"] for item in sources]
            })
    
    # Check for number conflicts (simplified)
    if len(sources_with_numbers) > 1:
        # Very basic number conflict detection
        all_numbers = []
        for source, numbers in sources_with_numbers.items():
            all_numbers.extend(numbers)
        
        # If we have very different ranges of numbers, flag as potential conflict
        if len(set(all_numbers)) > len(all_numbers) * 0.5:  # More than 50% unique numbers
            conflicts.append({
                "detection_method": "heuristic_numbers",
                "type": "data_inconsistency", 
                "description": "Sources contain significantly different numerical data",
                "sources_involved": list(sources_with_numbers.keys()),
                "severity": "medium"
            })
    
    return conflicts

def _detect_factual_contradictions(docs: List[Any]) -> List[Dict[str, Any]]:
    """Detect direct factual contradictions using simple patterns."""
    conflicts = []
    
    # Look for statements that directly contradict each other
    fact_patterns = [
        r'\b(?:is|are|was|were)\s+(?:not\s+)?(\w+)\b',
        r'\b(?:can|cannot|will|will not|should|should not)\s+(\w+)\b',
        r'\b(\w+)\s+(?:is|are)\s+(?:the\s+)?(?:best|worst|most|least)\b'
    ]
    
    statements_by_source = defaultdict(list)
    
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        content = doc.page_content
        
        for pattern in fact_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            statements_by_source[source].extend(matches)
    
    # Simple contradiction detection (could be much more sophisticated)
    if len(statements_by_source) > 1:
        sources = list(statements_by_source.keys())
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                source1, source2 = sources[i], sources[j]
                # Very basic check for opposing terms
                statements1 = set(statements_by_source[source1])
                statements2 = set(statements_by_source[source2])
                
                # Look for negation patterns
                potential_conflicts = []
                for stmt1 in statements1:
                    for stmt2 in statements2:
                        if stmt1.lower() != stmt2.lower() and len(stmt1) > 2 and len(stmt2) > 2:
                            # Simple check for potential contradictions
                            if (stmt1.lower() in stmt2.lower() or stmt2.lower() in stmt1.lower()):
                                potential_conflicts.append((stmt1, stmt2))
                
                if potential_conflicts:
                    conflicts.append({
                        "detection_method": "factual_pattern",
                        "type": "potential_contradiction",
                        "description": f"Potential factual contradictions between {source1} and {source2}",
                        "sources_involved": [source1, source2],
                        "severity": "medium",
                        "evidence": potential_conflicts[:3]  # Limit to first 3
                    })
    
    return conflicts

def _merge_similar_conflicts(conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge similar conflicts to avoid duplicates."""
    if not conflicts:
        return conflicts
    
    merged = []
    used_indices = set()
    
    for i, conflict1 in enumerate(conflicts):
        if i in used_indices:
            continue
            
        similar_conflicts = [conflict1]
        used_indices.add(i)
        
        for j, conflict2 in enumerate(conflicts[i+1:], i+1):
            if j in used_indices:
                continue
                
            # Check if conflicts are similar
            if (_are_conflicts_similar(conflict1, conflict2)):
                similar_conflicts.append(conflict2)
                used_indices.add(j)
        
        # Merge similar conflicts
        if len(similar_conflicts) > 1:
            merged_conflict = _merge_conflict_group(similar_conflicts)
            merged.append(merged_conflict)
        else:
            merged.append(conflict1)
    
    return merged

def _are_conflicts_similar(conflict1: Dict, conflict2: Dict) -> bool:
    """Check if two conflicts are similar enough to merge."""
    # Same type and overlapping sources
    return (conflict1.get("type") == conflict2.get("type") and 
            set(conflict1.get("sources_involved", [])) & set(conflict2.get("sources_involved", [])))

def _merge_conflict_group(conflicts: List[Dict]) -> Dict:
    """Merge a group of similar conflicts."""
    base_conflict = conflicts[0].copy()
    
    # Combine descriptions
    descriptions = [c.get("description", "") for c in conflicts]
    base_conflict["description"] = "; ".join(set(descriptions))
    
    # Combine sources
    all_sources = []
    for c in conflicts:
        all_sources.extend(c.get("sources_involved", []))
    base_conflict["sources_involved"] = list(set(all_sources))
    
    # Take highest severity
    severities = [c.get("severity", "low") for c in conflicts]
    severity_order = {"low": 1, "medium": 2, "high": 3}
    max_severity = max(severities, key=lambda x: severity_order.get(x, 1))
    base_conflict["severity"] = max_severity
    
    base_conflict["merged_from"] = len(conflicts)
    
    return base_conflict

def _analyze_reasoning_requirements(query: str) -> Dict[str, Any]:
    """Analyze what type of reasoning the query requires."""
    query_lower = query.lower()
    
    # Determine primary reasoning type
    reasoning_indicators = {
        "comparison": ["compare", "difference", "versus", "vs", "similar", "different", "contrast"],
        "causal": ["why", "because", "cause", "reason", "result", "effect", "lead to"],
        "procedural": ["how", "process", "steps", "procedure", "method", "way to"],
        "analytical": ["analyze", "analysis", "evaluate", "assess", "examine", "study"],
        "synthesis": ["summarize", "combine", "overall", "general", "main", "key points"],
        "temporal": ["when", "before", "after", "during", "timeline", "sequence", "order"],
        "conditional": ["if", "unless", "provided that", "assuming", "given that"]
    }
    
    detected_types = []
    for reasoning_type, indicators in reasoning_indicators.items():
        if any(indicator in query_lower for indicator in indicators):
            detected_types.append(reasoning_type)
    
    primary_type = detected_types[0] if detected_types else "general"
    
    # Assess complexity
    complexity_indicators = {
        "high": ["complex", "comprehensive", "detailed", "thorough", "in-depth", "multiple", "various"],
        "medium": ["explain", "describe", "discuss", "analyze"],
        "low": ["what", "define", "list", "name"]
    }
    
    complexity = "low"
    for level, indicators in complexity_indicators.items():
        if any(indicator in query_lower for indicator in indicators):
            complexity = level
            break
    
    # Check if multi-hop reasoning is needed
    multi_hop_indicators = ["and", "also", "additionally", "furthermore", "moreover", "both", "either"]
    multi_hop_required = any(indicator in query_lower for indicator in multi_hop_indicators)
    
    # Identify synthesis points
    synthesis_points = []
    if "overall" in query_lower or "general" in query_lower:
        synthesis_points.append("general_overview")
    if "main" in query_lower or "key" in query_lower:
        synthesis_points.append("key_points")
    if "summary" in query_lower or "summarize" in query_lower:
        synthesis_points.append("summarization")
    
    return {
        "primary_type": primary_type,
        "all_types": detected_types,
        "complexity": complexity,
        "multi_hop_required": multi_hop_required,
        "synthesis_points": synthesis_points,
        "word_count": len(query.split()),
        "question_marks": query.count("?")
    }

def _map_document_relationships(docs: List[Any]) -> Dict[str, Any]:
    """Map relationships between documents."""
    relationships = {
        "source_distribution": {},
        "content_overlap": {},
        "topic_clustering": {}
    }
    
    # Source distribution
    sources = [doc.metadata.get('source', 'unknown') for doc in docs]
    source_counts = Counter(sources)
    relationships["source_distribution"] = dict(source_counts)
    
    # Simple content overlap analysis
    relationships["content_overlap"] = _calculate_content_overlap(docs)
    
    # Basic topic clustering
    relationships["topic_clustering"] = _basic_topic_clustering(docs)
    
    return relationships

def _calculate_content_overlap(docs: List[Any]) -> Dict[str, float]:
    """Calculate content overlap between documents."""
    overlap_scores = {}
    
    for i, doc1 in enumerate(docs):
        for j, doc2 in enumerate(docs[i+1:], i+1):
            source1 = doc1.metadata.get('source', f'doc_{i}')
            source2 = doc2.metadata.get('source', f'doc_{j}')
            
            words1 = set(doc1.page_content.lower().split())
            words2 = set(doc2.page_content.lower().split())
            
            if words1 and words2:
                overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                overlap_scores[f"{source1} <-> {source2}"] = round(overlap, 3)
    
    return overlap_scores

def _basic_topic_clustering(docs: List[Any]) -> Dict[str, List[str]]:
    """Basic topic clustering based on common words."""
    # Extract common meaningful words (excluding stop words)
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'])
    
    word_freq = Counter()
    doc_words = {}
    
    for i, doc in enumerate(docs):
        words = [w.lower() for w in doc.page_content.split() if len(w) > 3 and w.lower() not in stop_words]
        doc_words[i] = set(words)
        word_freq.update(words)
    
    # Find most common words as topics
    top_words = [word for word, count in word_freq.most_common(10) if count > 1]
    
    clusters = {}
    for word in top_words:
        clusters[word] = []
        for i, words in doc_words.items():
            if word in words:
                source = docs[i].metadata.get('source', f'doc_{i}')
                clusters[word].append(source)
    
    # Remove clusters with only one document
    clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    
    return clusters

def _identify_reasoning_chains(query: str, docs: List[Any], requirements: Dict) -> List[Dict[str, Any]]:
    """Identify potential reasoning chains across documents."""
    chains = []
    
    if requirements["primary_type"] == "comparison":
        chains.extend(_identify_comparison_chains(docs))
    elif requirements["primary_type"] == "causal":
        chains.extend(_identify_causal_chains(docs, query))
    elif requirements["primary_type"] == "procedural":
        chains.extend(_identify_procedural_chains(docs))
    
    return chains

def _identify_comparison_chains(docs: List[Any]) -> List[Dict[str, Any]]:
    """Identify comparison chains between documents."""
    chains = []
    
    # Look for documents that can be compared
    sources = list(set(doc.metadata.get('source', 'unknown') for doc in docs))
    
    if len(sources) > 1:
        chains.append({
            "type": "comparison",
            "description": f"Compare information across {len(sources)} sources",
            "sources": sources,
            "steps": [
                "Extract key information from each source",
                "Identify similarities and differences", 
                "Synthesize comparative analysis"
            ]
        })
    
    return chains

def _identify_causal_chains(docs: List[Any], query: str) -> List[Dict[str, Any]]:
    """Identify causal reasoning chains."""
    chains = []
    
    # Look for causal indicators in documents
    causal_indicators = ["because", "due to", "caused by", "results in", "leads to", "therefore"]
    
    causal_docs = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        if any(indicator in content_lower for indicator in causal_indicators):
            causal_docs.append(doc.metadata.get('source', 'unknown'))
    
    if causal_docs:
        chains.append({
            "type": "causal",
            "description": "Trace cause-effect relationships",
            "sources": causal_docs,
            "steps": [
                "Identify cause-effect statements",
                "Map causal relationships",
                "Build logical chain to answer query"
            ]
        })
    
    return chains

def _identify_procedural_chains(docs: List[Any]) -> List[Dict[str, Any]]:
    """Identify procedural reasoning chains."""
    chains = []
    
    # Look for step indicators
    step_indicators = ["step", "first", "second", "then", "next", "finally", "process", "procedure"]
    
    procedural_docs = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        if any(indicator in content_lower for indicator in step_indicators):
            procedural_docs.append(doc.metadata.get('source', 'unknown'))
    
    if procedural_docs:
        chains.append({
            "type": "procedural",
            "description": "Follow procedural steps across sources",
            "sources": procedural_docs,
            "steps": [
                "Extract procedural information",
                "Order steps logically",
                "Synthesize complete procedure"
            ]
        })
    
    return chains

def _determine_reasoning_strategy(requirements: Dict, relationships: Dict) -> Dict[str, Any]:
    """Determine the optimal reasoning strategy."""
    strategy = {
        "approach": requirements["primary_type"],
        "complexity_handling": requirements["complexity"],
        "multi_source_synthesis": len(relationships["source_distribution"]) > 1,
        "conflict_resolution_needed": False,  # Will be updated based on conflicts
        "evidence_aggregation": "weighted" if requirements["complexity"] == "high" else "simple"
    }
    
    # Adjust strategy based on source diversity
    if len(relationships["source_distribution"]) > 3:
        strategy["source_prioritization"] = "diversity_weighted"
    else:
        strategy["source_prioritization"] = "relevance_weighted"
    
    # Adjust for multi-hop requirements
    if requirements["multi_hop_required"]:
        strategy["reasoning_depth"] = "multi_hop"
        strategy["synthesis_method"] = "chain_of_thought"
    else:
        strategy["reasoning_depth"] = "single_hop"
        strategy["synthesis_method"] = "direct_synthesis"
    
    return strategy
                    