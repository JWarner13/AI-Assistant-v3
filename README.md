# 📄 AI Document Assistant

The **AI Document Assistant** is an intelligent tool that helps you analyze and understand technical documents (like research papers, API documentation, or policies). You can ask it questions, and it will search through your documents, figure out the answer, and respond with clear, structured information.

This project uses a technique called **Retrieval-Augmented Generation (RAG)** to combine document search with a powerful language model (LLM), giving you accurate answers that are backed by your uploaded files.

---

## ✨ Features

✅ Load and analyze technical documents (PDF, text)  
✅ Ask questions in natural language (like “What is the key difference between X and Y?”)  
✅ Get responses in structured JSON format  
✅ Supports complex reasoning across multiple documents  
✅ Built-in caching for faster repeated queries  
✅ Easy to run from the command line

---

## 📁 Project Structure

```
AI-Assistant/
│
├── core/               # Core engine components (RAG, embeddings, LLM, etc.)
├── apps/               # CLI interface for running the assistant
├── configs/            # Configuration files
├── sample_queries.json # Examples of questions you can ask
├── main.py             # Entry point for the application
├── requirements.txt    # Python dependencies
└── README.md           # You're here!
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/JWarner13/AI-Assistant.git
cd AI-Assistant
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Your `.env` File

Create a `.env` file in the root directory and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_key_here
```

---

## 📘 How to Use It

Run the assistant from the command line:

```bash
python apps/cli.py
```

You'll be prompted to:
- Upload documents (PDF or `.txt`)
- Ask a question
- Receive a structured JSON answer with key info and reasoning

---

## 💬 Example Query

Ask:
```text
"What are the key differences between ISO 27001 and NIST SP 800-53?"
```

Response (simplified):
```json
{
  "answer": "ISO 27001 focuses on ISMS certification, while NIST SP 800-53 offers detailed controls for federal systems.",
  "sources": ["DocumentA.pdf", "DocumentB.txt"],
  "reasoning_summary": "Compared scope, structure, and application of both standards."
}
```

---

## 🧠 What's Happening Under the Hood?

This tool combines:
- 📚 **Embedding-based search** to understand document meaning
- 🧠 **Language model reasoning** to answer your questions smartly
- 🧾 **JSON output formatting** for clean, structured results
- ⚡ **Caching** to speed up repeated queries

---

## 🧪 Sample Scenarios

Use `sample_queries.json` to test:
- Factual lookups
- Cross-document comparisons
- Complex reasoning or ambiguous questions

---

## 🛠️ Future Improvements

- Web UI version
- Support for more file formats
- Fine-tuned domain-specific models
- More robust ambiguity detection

---

## 🤝 Contributing

Pull requests welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License
