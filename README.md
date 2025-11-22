# Document Intelligence System

This project is an AI-powered document assistant that enables users to interact with PDF files through natural language queries.  
The system extracts text from documents, indexes it using embeddings, and answers questions using a retrieval-augmented generation (RAG) pipeline.

---

## 🚀 Features

- Upload and process PDF documents  
- Automatic text extraction and segmentation  
- Embedding-based document search  
- Natural language question answering  
- Support for multi-page and large PDFs  
- Clean and simple user interface  

---

## 🧠 How It Works

1. **Text Extraction** — The PDF file is converted into raw text.  
2. **Chunking** — The text is split into semantic chunks.  
3. **Embedding Generation** — Each chunk is encoded into a vector representation.  
4. **Similarity Search** — Relevant document chunks are retrieved based on the user’s question.  
5. **Answer Generation** — A large language model (LLM) summarizes and answers using retrieved content.

This architecture follows the standard **Retrieval-Augmented Generation (RAG)** approach used in modern document AI systems.

---

## 🛠️ Technologies Used

- Python  
- LangChain or LlamaIndex (optional)  
- Vector database (FAISS, ChromaDB, or Pinecone)  
- OpenAI API / HuggingFace models  
- Streamlit / Flask / FastAPI (choose what fits your implementation)

---

## 📦 Installation

```bash
# Cloner le dépôt depuis GitHub
git clone https://github.com/safouane498/Ai-document-assistant.git

# Se déplacer dans le dossier du projet
cd Ai-document-assistant

# Installer les dépendances
pip install -r requirements.txt

