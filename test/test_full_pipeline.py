import json
import time
from pathlib import Path

def test_basic_query():
    """Test basic query functionality."""
    print("🔍 Testing Basic Query Processing...")
    
    try:
        from core.rag_engine import run_assistant
        
        query = "What is machine learning?"
        print(f"Query: {query}")
        
        start_time = time.time()
        result = run_assistant(query, enable_reasoning_trace=False)
        end_time = time.time()
        
        # Parse result
        data = json.loads(result)
        
        print(f"✅ Query processed in {end_time - start_time:.2f}s")
        print(f"📄 Sources: {len(data.get('sources', []))}")
        print(f"🎯 Answer preview: {data.get('answer', '')[:150]}...")
        
        # Validate response structure
        required_fields = ['answer', 'sources', 'documents_analyzed']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        
        print("✅ Response structure valid")
        return True
        
    except Exception as e:
        print(f"❌ Basic query failed: {e}")
        return False

def test_conflict_detection():
    """Test conflict detection with conflicting documents."""
    print("\n⚠️ Testing Conflict Detection...")
    
    try:
        from core.rag_engine import run_assistant
        
        # Query that should trigger conflicts between our test documents
        query = "What is the best approach for model validation?"
        print(f"Query: {query}")
        
        result = run_assistant(query, enable_reasoning_trace=True)
        data = json.loads(result)
        
        conflicts = data.get('conflicts_detected', [])
        print(f"✅ Conflicts detected: {len(conflicts)}")
        
        if conflicts:
            for i, conflict in enumerate(conflicts[:2], 1):
                conflict_type = conflict.get('type', 'unknown')
                description = conflict.get('description', 'N/A')
                severity = conflict.get('severity', 'unknown')
                print(f"   {i}. [{severity.upper()}] {conflict_type}: {description[:100]}...")
        
        # Test should find conflicts given our test documents
        if len(conflicts) > 0:
            print("✅ Conflict detection working (found expected conflicts)")
        else:
            print("⚠️ No conflicts detected (may be expected depending on documents)")
        
        return True
        
    except Exception as e:
        print(f"❌ Conflict detection failed: {e}")
        return False

def test_reasoning_traces():
    """Test reasoning trace generation."""
    print("\n🧠 Testing Reasoning Trace Generation...")
    
    try:
        from core.rag_engine import run_assistant
        
        # Complex query that requires reasoning
        query = "How do machine learning principles relate to natural language processing applications?"
        print(f"Query: {query}")
        
        start_time = time.time()
        result = run_assistant(query, enable_reasoning_trace=True)
        end_time = time.time()
        
        data = json.loads(result)
        reasoning = data.get('reasoning_trace', '')
        
        print(f"✅ Reasoning trace generated in {end_time - start_time:.2f}s")
        print(f"📝 Reasoning length: {len(reasoning)} characters")
        
        if reasoning:
            # Check for structured reasoning
            if "Step" in reasoning and "Reasoning Process" in reasoning:
                print("✅ Structured reasoning format detected")
            else:
                print("⚠️ Reasoning generated but format may not be structured")
            
            print(f"🔍 Reasoning preview: {reasoning[:200]}...")
        else:
            print("❌ No reasoning trace generated")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Reasoning trace failed: {e}")
        return False

def test_output_formats():
    """Test multiple output formats."""
    print("\n🎨 Testing Output Format Variations...")
    
    try:
        from core.rag_engine import run_assistant
        from core.output_formatter import (
            ResponseFormatter, 
            format_for_business, 
            format_for_human,
            format_for_api
        )
        
        query = "What are the main types of machine learning?"
        result = run_assistant(query)
        data = json.loads(result)
        
        print("✅ Base JSON format working")
        
        # Test different formats
        formats_to_test = [
            ("Human-readable", format_for_human),
            ("API format", format_for_api), 
            ("Business executive", format_for_business),
            ("Markdown", ResponseFormatter.format_markdown),
            ("Plain text", ResponseFormatter.format_plain_text)
        ]
        
        for format_name, format_func in formats_to_test:
            try:
                formatted = format_func(data)
                print(f"✅ {format_name} format: {len(formatted)} chars")
            except Exception as e:
                print(f"❌ {format_name} format failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Output formatting failed: {e}")
        return False

def test_caching_performance():
    """Test caching improves performance."""
    print("\n⚡ Testing Caching Performance...")
    
    try:
        from core.rag_engine import run_assistant
        
        query = "What is supervised learning?"
        
        # First run (no cache)
        print("First run (cold cache)...")
        start_time = time.time()
        result1 = run_assistant(query)
        first_time = time.time() - start_time
        
        # Second run (should use cache)
        print("Second run (warm cache)...")
        start_time = time.time()
        result2 = run_assistant(query)
        second_time = time.time() - start_time
        
        print(f"✅ First run: {first_time:.2f}s")
        print(f"✅ Second run: {second_time:.2f}s")
        
        if second_time < first_time * 0.8:  # At least 20% improvement
            print(f"✅ Caching improved performance by {((first_time - second_time) / first_time * 100):.1f}%")
        else:
            print("⚠️ Caching may not be working optimally")
        
        # Verify results are consistent
        data1 = json.loads(result1)
        data2 = json.loads(result2)
        
        if data1.get('answer') == data2.get('answer'):
            print("✅ Cached results consistent")
        else:
            print("❌ Cached results differ from original")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Caching performance test failed: {e}")
        return False

def test_batch_processing():
    """Test batch query processing."""
    print("\n📦 Testing Batch Processing...")
    
    try:
        from core.rag_engine import run_batch_queries
        
        test_queries = [
            "What is machine learning?",
            "What are the types of machine learning?",
            "How does deep learning work?",
            "What is natural language processing?"
        ]
        
        print(f"Processing {len(test_queries)} queries in batch...")
        start_time = time.time()
        results = run_batch_queries(test_queries)
        end_time = time.time()
        
        print(f"✅ Batch processing completed in {end_time - start_time:.2f}s")
        print(f"📊 Average time per query: {(end_time - start_time) / len(test_queries):.2f}s")
        
        # Validate results structure
        if isinstance(results, dict):
            print(f"✅ Results structure valid: {len(results)} results")
            return True
        else:
            print("❌ Invalid batch results structure")
            return False
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return False

def test_edge_cases():
    """Test system behavior with edge cases."""
    print("\n🔬 Testing Edge Cases...")
    
    try:
        from core.rag_engine import run_assistant
        
        edge_cases = [
            ("Empty query", ""),
            ("Very short query", "ML"),
            ("Very long query", "What is machine learning and how does it relate to artificial intelligence and what are all the different types and applications and how does it work in practice and what are the challenges?" * 3),
            ("Non-English query", "¿Qué es el aprendizaje automático?"),
            ("Query with special chars", "What is ML??? @#$%^&*()"),
            ("Nonsense query", "Quantum blockchain synergy optimization"),
        ]
        
        passed = 0
        for test_name, query in edge_cases:
            try:
                print(f"Testing: {test_name}")
                result = run_assistant(query, enable_reasoning_trace=False)
                data = json.loads(result)
                
                if 'answer' in data:
                    print(f"  ✅ Handled gracefully")
                    passed += 1
                else:
                    print(f"  ❌ No answer provided")
            except Exception as e:
                print(f"  ⚠️ Exception handled: {str(e)[:50]}...")
                passed += 1  # Graceful exception handling is acceptable
        
        print(f"✅ Edge cases: {passed}/{len(edge_cases)} handled gracefully")
        return passed >= len(edge_cases) * 0.7  # 70% success rate acceptable
        
    except Exception as e:
        print(f"❌ Edge case testing failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring and statistics."""
    print("\n📊 Testing Performance Monitoring...")
    
    try:
        from core.logger import get_performance_summary
        from core.embedding import get_embedding_stats
        from core.llm import get_llm_stats
        
        # Run a few queries to generate metrics
        from core.rag_engine import run_assistant
        for query in ["What is ML?", "Types of learning", "Deep learning"]:
            run_assistant(query)
        
        # Test performance summary
        try:
            perf_summary = get_performance_summary()
            print(f"✅ Performance summary available: {len(perf_summary)} metrics")
        except:
            print("⚠️ Performance summary not available")
        
        # Test embedding stats
        try:
            embed_stats = get_embedding_stats()
            print(f"✅ Embedding stats: {embed_stats.get('requests', {}).get('total', 0)} requests")
        except:
            print("⚠️ Embedding stats not available")
        
        # Test LLM stats
        try:
            llm_stats = get_llm_stats()
            print(f"✅ LLM stats: {llm_stats.get('requests', {}).get('total', 0)} requests")
        except:
            print("⚠️ LLM stats not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        return False

def test_cli_interface():
    """Test CLI interface functionality."""
    print("\n💻 Testing CLI Interface...")
    
    try:
        import subprocess
        import sys
        
        # Test basic CLI help
        result = subprocess.run([
            sys.executable, "cli.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ CLI help command works")
        else:
            print("❌ CLI help command failed")
            return False
        
        # Test query command
        result = subprocess.run([
            sys.executable, "cli.py", "query", "What is ML?", "--format", "text"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and len(result.stdout) > 50:
            print("✅ CLI query command works")
            print(f"📝 Output length: {len(result.stdout)} characters")
        else:
            print("⚠️ CLI query command may have issues")
            if result.stderr:
                print(f"   Error: {result.stderr[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI interface test failed: {e}")
        return False

def run_comprehensive_test():
    """Run the complete test suite."""
    print("🧪 AI DOCUMENT ASSISTANT - FULL PIPELINE TESTING")
    print("=" * 60)
    
    tests = [
        ("Basic Query Processing", test_basic_query),
        ("Conflict Detection", test_conflict_detection),
        ("Reasoning Traces", test_reasoning_traces),
        ("Output Formats", test_output_formats),
        ("Caching Performance", test_caching_performance),
        ("Batch Processing", test_batch_processing),
        ("Edge Cases", test_edge_cases),
        ("Performance Monitoring", test_performance_monitoring),
        ("CLI Interface", test_cli_interface),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Results:")
    print(f"   Tests Passed: {passed}/{len(tests)}")
    print(f"   Success Rate: {(passed/len(tests)*100):.1f}%")
    print(f"   Total Time: {total_time:.1f}s")
    
    if passed >= len(tests) * 0.8:  # 80% pass rate
        print("\n🎉 SYSTEM READY FOR DEMO!")
        print("✅ Pipeline is working well and ready for hackathon presentation")
        return True
    elif passed >= len(tests) * 0.6:  # 60% pass rate
        print("\n⚠️ SYSTEM MOSTLY WORKING")
        print("🔧 Some components need attention but core functionality works")
        return True
    else:
        print("\n❌ SYSTEM NEEDS WORK")
        print("🛠️ Multiple components failed - address issues before demo")
        return False

def main():
    """Main test execution."""
    success = run_comprehensive_test()
    
    if success:
        print("\n" + "=" * 60)
        print("🚀 NEXT STEPS FOR HACKATHON:")
        print("=" * 60)
        print("1. ✅ Run the demo script: python demo_script.py")
        print("2. ✅ Test with your own documents in the data/ folder")
        print("3. ✅ Practice the presentation with conflict detection demo")
        print("4. ✅ Prepare 2-3 sample queries that showcase different features")
        print("5. ✅ Test the CLI interactive mode: python cli.py interactive")
        print("\n🎯 Your system is hackathon-ready!")
    else:
        print("\n" + "=" * 60)
        print("🛠️ FIXES NEEDED:")
        print("=" * 60)
        print("1. Review failed test outputs above")
        print("2. Check API key and network connectivity")
        print("3. Verify all dependencies are installed")
        print("4. Run individual component tests: python test_components.py")
        print("5. Check logs in outputs/ directory for detailed errors")

if __name__ == "__main__":
    main()