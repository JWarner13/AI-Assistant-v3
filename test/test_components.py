from core.cache import CacheManager
import json
import time
from pathlib import Path

def test_document_loading():
    """Test document loading and processing."""
    print("🔍 Testing Document Loading...")
    
    try:
        from core.document_loader import load_documents, get_document_summary
        
        # Load documents
        documents = load_documents("data")
        
        if not documents:
            print("❌ No documents loaded! Make sure documents exist in data/ folder")
            return False
        
        print(f"✅ Loaded {len(documents)} document chunks")
        
        # Get summary
        summary = get_document_summary(documents)
        print(f"✅ Document summary: {summary['unique_sources']} sources, {summary['content_statistics']['total_words']} words")
        
        # Show first document preview
        if documents:
            first_doc = documents[0]
            print(f"📄 Sample chunk: {first_doc['content'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM system failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_system():
    """Test embedding generation."""
    print("\n🔗 Testing Embedding System...")
    
    try:
        from core.embedding import get_embedding_model
        
        # Initialize embedding model
        embedding_model = get_embedding_model()
        
        # Test single embedding
        test_text = "What is machine learning?"
        start_time = time.time()
        embedding = embedding_model.embed_query(test_text)
        end_time = time.time()
        
        print(f"✅ Generated embedding: {len(embedding)} dimensions in {end_time - start_time:.2f}s")
        
        # Test batch embeddings
        test_texts = [
            "Machine learning fundamentals",
            "Deep learning neural networks", 
            "Natural language processing"
        ]
        
        start_time = time.time()
        embeddings = embedding_model.embed_documents(test_texts)
        end_time = time.time()
        
        print(f"✅ Generated {len(embeddings)} batch embeddings in {end_time - start_time:.2f}s")
        
        # Test caching
        start_time = time.time()
        cached_embedding = embedding_model.embed_query(test_text)  # Should be cached
        end_time = time.time()
        
        print(f"✅ Cached embedding retrieved in {end_time - start_time:.4f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM system failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_indexing_system():
    """Test vector indexing."""
    print("\n🗃️ Testing Indexing System...")
    
    try:
        from core.document_loader import load_documents
        from core.embedding import get_embedding_model
        from core.index import get_index_manager
        
        # Load documents and create embeddings
        documents = load_documents("data")
        embedding_model = get_embedding_model()
        
        # Build index
        index_manager = get_index_manager()
        start_time = time.time()
        index = index_manager.build_index(documents, embedding_model.model, "test_index")
        end_time = time.time()
        
        print(f"✅ Built index with {len(documents)} chunks in {end_time - start_time:.2f}s")
        
        # Test search
        test_query = "What is machine learning?"
        start_time = time.time()
        results = index_manager.search_index(index, test_query, k=3)
        end_time = time.time()
        
        print(f"✅ Search returned {len(results)} results in {end_time - start_time:.4f}s")
        
        # Show search results
        for i, doc in enumerate(results[:2]):
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:100] + "..."
            print(f"   {i+1}. {source}: {preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM system failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_system():
    """Test LLM integration."""
    print("\n🤖 Testing LLM System...")
    
    try:
        from core.llm import get_llm
        
        # Initialize LLM
        llm = get_llm()
        
        # Test simple generation
        test_prompt = "Explain machine learning in one sentence."
        start_time = time.time()
        response = llm.invoke(test_prompt)
        end_time = time.time()
        
        print(f"✅ LLM response generated in {end_time - start_time:.2f}s")
        print(f"📝 Response: {response[:150]}...")
        
        # Test with template
        context = "Machine learning is a method of data analysis that automates analytical model building."
        query = "What is machine learning?"
        
        templated_response = llm.generate_with_template(context, query, "default")
        print(f"✅ Template-based response generated")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM system failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reasoning_system():
    """Test reasoning and conflict detection."""
    print("\n🧠 Testing Reasoning System...")
    
    try:
        from core.reasoning import detect_conflicts, explain_reasoning
        from core.document_loader import load_documents
        from core.embedding import get_embedding_model
        from core.index import get_index_manager
        
        # Get some documents for testing
        documents = load_documents("data")
        embedding_model = get_embedding_model()
        index_manager = get_index_manager()
        index = index_manager.build_index(documents, embedding_model.model, "reasoning_test")
        
        # Test conflict detection with a query that should find conflicts
        conflict_query = "What is the best approach for model validation?"
        retrieved_docs = index_manager.search_index(index, conflict_query, k=5)
        
        conflicts = detect_conflicts(retrieved_docs, conflict_query)
        print(f"✅ Conflict detection: {len(conflicts)} conflicts found")
        
        if conflicts:
            for i, conflict in enumerate(conflicts[:2]):
                print(f"   {i+1}. {conflict.get('type', 'unknown')}: {conflict.get('description', 'N/A')[:100]}...")
        
        # Test reasoning explanation
        if retrieved_docs:
            sample_response = "Model validation can be done using cross-validation or holdout methods."
            reasoning = explain_reasoning(conflict_query, retrieved_docs[:2], sample_response)
            print(f"✅ Reasoning explanation generated: {len(reasoning)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM system failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_caching_system():
    """Test caching functionality."""
    print("\n💾 Testing Caching System...")
    
    try:
        from core.cache import CacheManager
        
        # Create cache manager
        cache = CacheManager()
        
        # Test basic operations
        cache.set("test_key", "test_value")
        retrieved = cache.get("test_key")
        
        if retrieved == "test_value":
            print("✅ Basic cache operations working")
        else:
            print("❌ Cache retrieval failed")
            return False
        
        # Test TTL
        cache.set("ttl_test", "expires_soon", ttl=1)
        time.sleep(1.1)
        expired = cache.get("ttl_test")
        
        if expired is None:
            print("✅ TTL expiration working")
        else:
            print("❌ TTL not working properly")
        
        # Test stats
        stats = cache.get_stats()
        print(f"✅ Cache stats: {stats['hits']} hits, {stats['misses']} misses")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM system failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all component tests."""
    print("🧪 AI Document Assistant - Component Testing")
    print("=" * 50)
    
    tests = [
        test_document_loading,
        test_embedding_system,
        test_indexing_system,
        test_llm_system,
        test_reasoning_system,
        test_caching_system
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    test_names = [
        "Document Loading",
        "Embedding System", 
        "Indexing System",
        "LLM System",
        "Reasoning System",
        "Caching System"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All components working! Ready for full pipeline testing.")
        return True
    else:
        print("⚠️  Some components failed. Fix issues before full pipeline testing.")
        return False

if __name__ == "__main__":
    main()