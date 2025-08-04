import os
from pathlib import Path

def create_test_documents():
    """Create test documents with potential conflicts for comprehensive testing."""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Document 1: ML Fundamentals (Positive on Deep Learning)
    doc1_content = """
# Machine Learning Fundamentals

## Introduction
Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It has revolutionized how we approach complex problem-solving across industries.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled training data to learn a mapping function from inputs to outputs. The algorithm learns from input-output pairs and can then predict outputs for new, unseen inputs.

**Key characteristics:**
- Requires labeled training data
- Performance can be measured against known correct answers
- Common algorithms include linear regression, decision trees, and neural networks
- Best for prediction and classification tasks

**Applications:**
- Email spam detection
- Image recognition
- Medical diagnosis
- Financial fraud detection

### Unsupervised Learning
Unsupervised learning finds hidden patterns in data without labeled examples. It discovers structure in data where you don't know the outcome you're looking for.

**Key characteristics:**
- No labeled training data required
- Discovers hidden patterns and relationships
- Common algorithms include clustering, association rules, and dimensionality reduction
- Best for exploratory data analysis

**Applications:**
- Customer segmentation
- Anomaly detection
- Recommendation systems
- Data compression

### Reinforcement Learning
Reinforcement learning trains agents to make decisions by rewarding good actions and penalizing bad ones. The agent learns through trial and error interaction with an environment.

**Applications:**
- Game playing (Chess, Go)
- Autonomous vehicles
- Robot control
- Trading algorithms

## Deep Learning Advantages
Deep learning, a subset of machine learning using neural networks, has shown remarkable success in recent years. It excels at:
- Image and speech recognition
- Natural language processing
- Complex pattern recognition
- Automatic feature extraction

Deep learning models can achieve superhuman performance on many tasks and are the current state-of-the-art approach for most AI applications.

## Best Practices for ML Projects

### Data Quality
High-quality data is absolutely critical for machine learning success. Poor data quality will inevitably lead to poor model performance, regardless of the algorithm used.

**Data quality factors:**
- Accuracy: Data should be correct and error-free
- Completeness: Minimal missing values
- Consistency: Uniform formatting and standards
- Relevance: Data should be pertinent to the problem
- Timeliness: Data should be current and up-to-date

### Model Validation
Cross-validation is the gold standard for model validation. It provides a robust estimate of model performance by testing on multiple data subsets.

**Recommended approach:**
1. Split data into training (70%), validation (15%), and test (15%) sets
2. Use k-fold cross-validation on training data
3. Fine-tune hyperparameters using validation set
4. Final evaluation on test set only once

### Performance Metrics
Choose metrics that align with business objectives:
- Classification: Accuracy, Precision, Recall, F1-score
- Regression: RMSE, MAE, R-squared
- Consider class imbalance and business costs

## Production Deployment
Successful ML deployment requires careful consideration of:
- Model monitoring and retraining schedules
- A/B testing for model updates
- Scalability and latency requirements
- Data pipeline robustness
- Model interpretability and explainability
"""

    # Document 2: AI Ethics and Limitations (Critical of Deep Learning)
    doc2_content = """
# AI Ethics and Current Limitations

## The Reality of AI Limitations

While artificial intelligence has made impressive strides, it's crucial to understand its current limitations and potential risks. Many claims about AI capabilities are overstated, and deep learning approaches face significant challenges.

## Deep Learning Concerns

### Data Dependency Issues
Deep learning models require massive amounts of labeled data, which is often:
- Expensive and time-consuming to collect
- Prone to bias and representation issues
- Not available for many specialized domains
- Quickly becomes outdated

### Interpretability Problems
Deep learning models are essentially "black boxes" that make it difficult to:
- Understand decision-making processes
- Debug errors and failures
- Meet regulatory requirements
- Build trust with stakeholders

**This is particularly problematic in:**
- Medical diagnosis
- Financial lending decisions
- Criminal justice applications
- Safety-critical systems

### Overfitting and Generalization
Deep learning models often:
- Memorize training data rather than learning generalizable patterns
- Fail catastrophically on slightly different data
- Require extensive regularization techniques
- Need careful validation to avoid overfitting

## Alternative Approaches Often Overlooked

### Traditional Machine Learning Advantages
Classical algorithms like decision trees, SVMs, and linear models often provide:
- Better interpretability
- Faster training and inference
- Lower computational requirements
- More stable performance with limited data
- Easier debugging and maintenance

### When to Avoid Deep Learning
Consider simpler approaches when:
- Data is limited (less than 10,000 samples)
- Interpretability is critical
- Computational resources are constrained
- Problem complexity doesn't warrant neural networks
- Fast iteration and debugging are priorities

## Model Validation Best Practices - Alternative View

### The Holdout Method Debate
While cross-validation is popular, simple train-test splits (holdout method) are often more practical and realistic for production scenarios.

**Advantages of holdout validation:**
- Simpler to implement and understand
- Faster computation, especially for large datasets
- More realistic simulation of production deployment
- Avoids data leakage issues common in cross-validation

**Recommended approach:**
1. Single train (80%) and test (20%) split
2. Use temporal splits for time-series data
3. Stratify splits to maintain class distributions
4. Validate on truly independent test data

### Metric Selection Nuances
Common metrics can be misleading:
- Accuracy is meaningless with imbalanced datasets
- F1-score may not reflect business value
- Cross-validation scores often overestimate performance
- Consider domain-specific metrics and business impact

## Production Deployment Challenges

### The 90% Problem
Most ML projects fail in production due to:
- Insufficient focus on data engineering
- Underestimating deployment complexity
- Lack of monitoring and maintenance planning
- Overemphasis on model accuracy vs. business value

### Alternative Deployment Strategies
- Start with simple rule-based systems
- Use ML to augment rather than replace human decisions
- Implement gradual rollouts with extensive monitoring
- Focus on robust data pipelines over complex models

## Ethical Considerations

### Bias and Fairness
AI systems can perpetuate and amplify societal biases through:
- Biased training data
- Algorithmic bias in model design
- Feedback loops that reinforce discrimination
- Lack of diverse perspectives in development teams

### Responsible AI Practices
- Regular bias audits and fairness assessments
- Diverse and inclusive development teams
- Transparent documentation of model limitations
- User consent and control over AI decisions
- Clear appeals processes for automated decisions

## Conclusion
While AI and machine learning offer powerful tools, success requires realistic expectations, careful validation, and ethical considerations. Often, simpler approaches provide better long-term value than complex deep learning solutions.
"""

    # Document 3: NLP and Advanced Applications
    doc3_content = """
# Natural Language Processing: State of the Art

## Introduction to NLP
Natural Language Processing (NLP) represents one of the most challenging and exciting frontiers in artificial intelligence. It combines computational linguistics with machine learning to enable computers to understand, interpret, and generate human language.

## Core NLP Tasks

### Text Classification
Automatically categorizing text into predefined classes:
- Sentiment analysis (positive, negative, neutral)
- Topic classification (sports, politics, technology)
- Intent recognition in chatbots
- Spam detection in emails

### Named Entity Recognition (NER)
Identifying and classifying entities in text:
- Person names (John Smith, Marie Curie)
- Organizations (Google, United Nations)
- Locations (New York, Mount Everest)
- Dates, monetary values, percentages

### Information Extraction
Extracting structured information from unstructured text:
- Relationship extraction (Person X works for Company Y)
- Event extraction (Meeting scheduled for Tuesday)
- Fact extraction for knowledge bases

### Machine Translation
Automatically translating text between languages:
- Statistical machine translation (older approach)
- Neural machine translation (current state-of-the-art)
- Zero-shot translation for low-resource languages

## Advanced NLP Applications

### Question Answering Systems
Modern QA systems can:
- Extract answers from large document collections
- Synthesize information from multiple sources
- Handle complex, multi-hop reasoning questions
- Provide confidence scores for answers

**Key technologies:**
- BERT and transformer architectures
- Dense passage retrieval
- Reading comprehension models
- Knowledge graph integration

### Conversational AI
Building systems that can engage in natural dialogue:
- Task-oriented chatbots for customer service
- Open-domain conversational agents
- Multi-turn dialogue management
- Context awareness and memory

### Text Generation
Creating human-like text for various purposes:
- Creative writing assistance
- Automated report generation
- Code generation from natural language descriptions
- Personalized content creation

## NLP Challenges and Solutions

### Ambiguity Resolution
Natural language is inherently ambiguous:
- Lexical ambiguity (bank = financial institution or river bank)
- Syntactic ambiguity (multiple parse trees)
- Semantic ambiguity (multiple meanings)
- Pragmatic ambiguity (context-dependent interpretation)

**Solution approaches:**
- Context-aware models
- Word sense disambiguation
- Dependency parsing
- Coreference resolution

### Handling Low-Resource Languages
Most NLP research focuses on English, but solutions exist for other languages:
- Cross-lingual transfer learning
- Multilingual pretrained models
- Data augmentation techniques
- Unsupervised learning approaches

### Domain Adaptation
Models trained on general text often fail in specialized domains:
- Medical text processing
- Legal document analysis
- Scientific literature mining
- Social media text analysis

**Adaptation strategies:**
- Domain-specific pretraining
- Fine-tuning on domain data
- Few-shot learning techniques
- Active learning for labeling efficiency

## Evaluation in NLP

### Intrinsic vs. Extrinsic Evaluation
- **Intrinsic**: Evaluating components in isolation (POS tagging accuracy)
- **Extrinsic**: Evaluating end-to-end system performance (user satisfaction)

### Common Evaluation Metrics
- **Classification tasks**: Precision, Recall, F1-score, Accuracy
- **Generation tasks**: BLEU, ROUGE, METEOR, human evaluation
- **Information Retrieval**: MAP, MRR, NDCG
- **Question Answering**: Exact Match, F1, METEOR

### Human Evaluation Considerations
Automated metrics don't capture all aspects of quality:
- Fluency and naturalness
- Factual correctness
- Relevance to user needs
- Potential for harm or bias

## Current Trends and Future Directions

### Large Language Models
Recent advances in transformer-based models:
- GPT family (GPT-3, GPT-4)
- BERT and its variants
- T5, PaLM, and other large-scale models
- Implications for few-shot and zero-shot learning

### Multimodal NLP
Combining text with other modalities:
- Vision-language models (image captioning, VQA)
- Speech-text integration
- Video understanding with text
- Embodied AI with language grounding

### Efficient NLP
Making NLP more accessible and sustainable:
- Model compression and distillation
- Efficient architectures (MobileBERT, DistilBERT)
- Few-shot learning techniques
- Edge deployment considerations

## Best Practices for NLP Projects

### Data Preparation
- Careful text preprocessing and cleaning
- Handling of noisy, user-generated content
- Appropriate tokenization strategies
- Character encoding and normalization

### Model Selection
- Start with pretrained models when possible
- Consider computational constraints
- Evaluate multiple architectures
- Balance performance with interpretability needs

### Evaluation Strategy
- Use appropriate train/validation/test splits
- Consider temporal aspects for time-sensitive data
- Include human evaluation for critical applications
- Monitor for dataset shift and model degradation

### Deployment Considerations
- Latency requirements for real-time applications
- Handling of out-of-vocabulary words
- Robustness to adversarial inputs
- Monitoring and maintenance in production

## Conclusion
NLP continues to evolve rapidly, with new breakthroughs regularly pushing the boundaries of what's possible. Success in NLP requires understanding both the technical capabilities and limitations of current approaches, as well as careful consideration of real-world deployment challenges.
"""

    # Write the documents
    with open(data_dir / "ml_fundamentals.txt", "w", encoding="utf-8") as f:
        f.write(doc1_content)
    
    with open(data_dir / "ai_ethics_limitations.txt", "w", encoding="utf-8") as f:
        f.write(doc2_content)
    
    with open(data_dir / "nlp_advanced.txt", "w", encoding="utf-8") as f:
        f.write(doc3_content)
    
    print("✅ Test documents created successfully!")
    print(f"📁 Documents location: {data_dir.absolute()}")
    print("📄 Created files:")
    print("   1. ml_fundamentals.txt (Pro-deep learning perspective)")
    print("   2. ai_ethics_limitations.txt (Critical perspective)")
    print("   3. nlp_advanced.txt (Comprehensive NLP overview)")
    print("\n🔍 These documents contain intentional conflicts for testing conflict detection!")

if __name__ == "__main__":
    create_test_documents()