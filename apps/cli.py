import argparse
import json
import sys
import os
from pathlib import Path
from core.rag_engine import run_assistant, run_batch_queries
from core.logger import log

def main():
    parser = argparse.ArgumentParser(description="AI Document Assistant - RAG-based Q&A system")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single query command
    query_parser = subparsers.add_parser('query', help='Ask a single question')
    query_parser.add_argument('question', type=str, help='The question to ask')
    query_parser.add_argument('--data-dir', type=str, default='data', 
                            help='Directory containing documents (default: data)')
    query_parser.add_argument('--output', type=str, 
                            help='Output file path (default: print to stdout)')
    query_parser.add_argument('--reasoning', action='store_true', 
                            help='Enable detailed reasoning traces')
    query_parser.add_argument('--format', choices=['json', 'text'], default='json',
                            help='Output format')
    
    # Batch query command
    batch_parser = subparsers.add_parser('batch', help='Process multiple queries from file')
    batch_parser.add_argument('queries_file', type=str, 
                            help='JSON file containing list of queries')
    batch_parser.add_argument('--output', type=str, 
                            help='Output file path (default: queries_results.json)')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup the environment')
    setup_parser.add_argument('--create-sample-data', action='store_true',
                            help='Create sample data directory and files')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    interactive_parser.add_argument('--data-dir', type=str, default='data',
                                  help='Directory containing documents')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'query':
            handle_single_query(args)
        elif args.command == 'batch':
            handle_batch_queries(args)
        elif args.command == 'setup':
            handle_setup(args)
        elif args.command == 'interactive':
            handle_interactive_mode(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def handle_single_query(args):
    """Handle single query command."""
    # Validate data directory
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory '{args.data_dir}' not found. Run 'python cli.py setup --create-sample-data' first.")
    
    # Check for documents
    doc_files = [f for f in os.listdir(args.data_dir) if f.endswith(('.pdf', '.txt'))]
    if not doc_files:
        raise FileNotFoundError(f"No PDF or TXT files found in '{args.data_dir}'. Please add some documents.")
    
    print(f"Processing query: {args.question}")
    print(f"Using documents from: {args.data_dir}")
    print(f"Found {len(doc_files)} documents: {', '.join(doc_files)}")
    print("=" * 50)
    
    # Set data directory (you'd need to modify your code to accept this parameter)
    result = run_assistant(args.question, enable_reasoning_trace=args.reasoning)
    
    if args.format == 'text':
        # Convert JSON to readable text
        data = json.loads(result)
        output = format_as_text(data)
    else:
        output = result
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Results saved to: {args.output}")
    else:
        print(output)

def handle_batch_queries(args):
    """Handle batch query processing."""
    if not os.path.exists(args.queries_file):
        raise FileNotFoundError(f"Queries file '{args.queries_file}' not found.")
    
    with open(args.queries_file, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    if isinstance(queries_data, list):
        queries = queries_data
    elif isinstance(queries_data, dict) and 'queries' in queries_data:
        queries = queries_data['queries']
    else:
        raise ValueError("Queries file must contain a list of queries or a dict with 'queries' key.")
    
    print(f"Processing {len(queries)} queries...")
    results = run_batch_queries(queries)
    
    output_file = args.output or 'queries_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"Batch results saved to: {output_file}")

def handle_setup(args):
    """Handle setup command."""
    if args.create_sample_data:
        create_sample_data()
    
    # Check environment
    check_environment()

def handle_interactive_mode(args):
    """Handle interactive mode."""
    print("AI Document Assistant - Interactive Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("=" * 50)
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"Warning: Data directory '{args.data_dir}' not found.")
        print("Run 'python cli.py setup --create-sample-data' to create sample data.")
        return
    
    while True:
        try:
            user_input = input("\nYour question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print_interactive_help()
                continue
            elif not user_input:
                continue
            
            print("Processing...")
            result = run_assistant(user_input, enable_reasoning_trace=True)
            data = json.loads(result)
            print("\n" + format_as_text(data))
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def create_sample_data():
    """Create sample data directory and files."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    sample_txt = data_dir / 'sample_document.txt'
    if not sample_txt.exists():
        sample_content = """
# Sample Technical Document

## Introduction
This is a sample technical document for testing the AI Document Assistant.
It contains information about machine learning and natural language processing.

## Machine Learning Concepts
Machine learning is a subset of artificial intelligence that focuses on algorithms
that can learn from and make decisions or predictions based on data.

### Types of Machine Learning
1. Supervised Learning: Uses labeled data to train models
2. Unsupervised Learning: Finds patterns in unlabeled data
3. Reinforcement Learning: Learns through interaction with environment

## Natural Language Processing
NLP is a field of AI that helps computers understand, interpret, and manipulate
human language. Key applications include:
- Text classification
- Sentiment analysis
- Machine translation
- Question answering systems

## Best Practices
When implementing ML systems:
- Always validate your data quality
- Use appropriate evaluation metrics
- Consider model interpretability
- Plan for model maintenance and updates
"""
        sample_txt.write_text(sample_content, encoding='utf-8')
        print(f"Created sample document: {sample_txt}")
    
    # Create sample queries file
    queries_file = Path('sample_queries.json')
    if not queries_file.exists():
        sample_queries = {
            "queries": [
                "What are the main types of machine learning?",
                "What is natural language processing used for?",
                "What are the best practices for ML systems?",
                "How does supervised learning work?"
            ]
        }
        queries_file.write_text(json.dumps(sample_queries, indent=4), encoding='utf-8')
        print(f"Created sample queries file: {queries_file}")
    
    print("Sample data created successfully!")

def check_environment():
    """Check if the environment is properly configured."""
    print("Checking environment...")
    
    # Check .env file
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        print("   Create .env file with: OPENAI_API_KEY=your-api-key-here")
    else:
        print("✅ .env file found")
    
    # Check required packages
    required_packages = ['openai', 'langchain', 'faiss-cpu', 'tiktoken', 'PyMuPDF']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} not installed")
    
    if missing_packages:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")

def format_as_text(data):
    """Convert JSON response to readable text format."""
    output = []
    output.append("🤖 AI ASSISTANT RESPONSE")
    output.append("=" * 50)
    
    # Main answer
    if 'answer' in data:
        output.append("\n📝 ANSWER:")
        output.append(data['answer'])
    
    # Sources
    if 'sources' in data and data['sources']:
        output.append(f"\n📚 SOURCES USED ({len(data['sources'])}):")
        for i, source in enumerate(data['sources'], 1):
            output.append(f"  {i}. {source}")
    
    # Reasoning trace
    if 'reasoning_trace' in data and data['reasoning_trace']:
        output.append("\n🧠 REASONING PROCESS:")
        output.append(data['reasoning_trace'])
    
    # Conflicts
    if 'conflicts_detected' in data and data['conflicts_detected']:
        output.append("\n⚠️  CONFLICTS DETECTED:")
        for conflict in data['conflicts_detected']:
            output.append(f"  • {conflict.get('description', 'Unknown conflict')}")
    
    # Confidence indicators
    if 'confidence_indicators' in data:
        indicators = data['confidence_indicators']
        output.append(f"\n📊 ANALYSIS STATS:")
        output.append(f"  • Documents analyzed: {data.get('documents_analyzed', 'N/A')}")
        output.append(f"  • Unique sources: {indicators.get('source_diversity', 'N/A')}")
        output.append(f"  • Text chunks processed: {indicators.get('total_chunks', 'N/A')}")
    
    return "\n".join(output)

def print_interactive_help():
    """Print help for interactive mode."""
    help_text = """
Available commands:
  help    - Show this help message
  quit    - Exit the program
  
Simply type your question and press Enter to get an answer.
The system will analyze your documents and provide detailed responses.
"""
    print(help_text)

if __name__ == '__main__':
    main()
