#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path
from core.rag_engine import run_assistant
from core.logger import log

def validate_environment():
    """Validate environment setup before running."""
    errors = []
    warnings = []
    
    # Check .env file
    if not Path(".env").exists():
        errors.append("❌ .env file not found! Create it with: HUGGINGFACEHUB_API_TOKEN=your_token_here")
    else:
        # Check if API token is set
        from dotenv import load_dotenv
        load_dotenv()
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token or token == "your_token_here":
            warnings.append("⚠️  HuggingFace API token not properly configured")
    
    # Check data directory
    data_dir = Path("data")
    if not data_dir.exists():
        errors.append(f"❌ Data directory '{data_dir}' not found!")
        return errors, warnings
    
    # Check for documents
    doc_files = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.txt"))
    if not doc_files:
        errors.append(f"❌ No PDF or TXT files found in '{data_dir}'!")
    
    return errors, warnings

def create_sample_data():
    """Create sample data for testing."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    sample_file = data_dir / "sample_ml_guide.txt"
    sample_content = """# Machine Learning Guide

## Introduction
Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence (AI) based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled examples to train algorithms to classify data or predict outcomes accurately. Common algorithms include:
- Linear Regression
- Decision Trees
- Random Forest
- Support Vector Machines
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns or intrinsic structures in input data. Types include:
- Clustering (K-means, Hierarchical)
- Association Rules
- Dimensionality Reduction (PCA, t-SNE)

### Reinforcement Learning
Reinforcement learning is about taking suitable action to maximize reward in a particular situation. It is employed by various software and machines to find the best possible behavior or path it should take in a specific situation.

## Best Practices
1. **Data Quality**: Ensure your data is clean, relevant, and representative
2. **Feature Engineering**: Create meaningful features that help the model learn
3. **Model Selection**: Choose the right algorithm for your problem
4. **Validation**: Use proper cross-validation techniques
5. **Monitoring**: Continuously monitor model performance in production

## Common Challenges
- Overfitting and underfitting
- Data quality issues
- Feature selection
- Model interpretability
- Scalability concerns
"""
    
    sample_file.write_text(sample_content, encoding='utf-8')
    print(f"✅ Created sample document: {sample_file}")
    
    # Create .env template if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = "HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here\n"
        env_file.write_text(env_content)
        print(f"✅ Created .env template: {env_file}")
        print("   Please edit .env and add your actual HuggingFace token")

def display_results(data):
    """Display results in a user-friendly format."""
    print("\n" + "="*60)
    print("🤖 AI DOCUMENT ASSISTANT RESPONSE")
    print("="*60)
    
    # Main answer
    print("\n📝 ANSWER:")
    print("-" * 40)
    print(data.get('answer', 'No answer provided'))
    
    # Sources
    sources = data.get('sources', [])
    if sources:
        print(f"\n📚 SOURCES ({len(sources)}):")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source}")
    
    # Performance metrics
    metrics = data.get('performance_metrics', {})
    if metrics:
        print("\n⚡ PERFORMANCE:")
        proc_time = metrics.get('processing_time_seconds', 'N/A')
        docs_analyzed = metrics.get('documents_analyzed', 'N/A')
        print(f"  • Processing time: {proc_time} seconds")
        print(f"  • Documents analyzed: {docs_analyzed}")
    
    # Conflicts
    conflicts = data.get('conflicts_detected', [])
    if conflicts:
        print(f"\n⚠️  CONFLICTS DETECTED ({len(conflicts)}):")
        for i, conflict in enumerate(conflicts, 1):
            desc = conflict.get('description', 'Unknown conflict')
            severity = conflict.get('severity', 'unknown')
            print(f"  {i}. [{severity.upper()}] {desc}")
    
    # Reasoning trace (abbreviated)
    reasoning = data.get('reasoning_trace', '')
    if reasoning and len(reasoning) > 100:
        print("\n🧠 REASONING PROCESS:")
        print("-" * 40)
        # Show first 500 characters
        preview = reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
        print(preview)

def main():
    """Enhanced main function with proper error handling."""
    print("🚀 AI Document Assistant")
    print("=" * 40)
    
    try:
        # Validate environment
        errors, warnings = validate_environment()
        
        # Display warnings
        for warning in warnings:
            print(warning)
        
        # Handle errors
        if errors:
            print("\n❌ Setup Issues Found:")
            for error in errors:
                print(error)
            
            print("\n🔧 Quick Fix:")
            print("Run this to create sample data and setup:")
            print("  python -c \"from main import create_sample_data; create_sample_data()\"")
            return 1
        
        # Get query
        if len(sys.argv) > 1:
            user_query = " ".join(sys.argv[1:])
        else:
            user_query = "What are the main types of machine learning and their key characteristics?"
        
        print(f"❓ Query: {user_query}")
        print("\n🔄 Processing... (this may take a moment)")
        
        # Run the assistant
        result = run_assistant(
            user_query, 
            enable_reasoning_trace=True,
            data_dir="data",
            use_cache=True
        )
        
        # Parse and display results
        try:
            response_data = json.loads(result)
            display_results(response_data)
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing response: {e}")
            print("Raw response:", result[:500] + "..." if len(result) > 500 else result)
            return 1
        
        print(f"\n💡 Try different queries by running:")
        print(f"  python main.py \"Your question here\"")
        print(f"  python cli.py interactive  # For interactive mode")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        return 0
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Show helpful debugging info
        print(f"\n🔍 Debug Info:")
        print(f"  • Python version: {sys.version}")
        print(f"  • Current directory: {os.getcwd()}")
        print(f"  • Data directory exists: {Path('data').exists()}")
        
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)