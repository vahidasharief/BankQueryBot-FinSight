# FinSight — BankQueryBot

**FinSight** (aka *BankQueryBot-FinSight*) is a retrieval-augmented banking chatbot that answers user questions from bank PDF documents (HSBC, HDFC, ICICI, Canara).  
It uses local sentence embeddings + FAISS vector search and a local LLM via **Ollama (Gemma 3)** to produce citation-backed answers — all processing can be done locally (no cloud APIs required).

---

## 🚀 Features
- Local SBERT embeddings (`all-MiniLM-L6-v2`) for private embedding generation  #you can chose any availabe best model according to your local machine configuration.
- Per-bank FAISS vector stores for fast similarity search  
- Ollama + Gemma (local LLM) for grounded answer generation  
- Deterministic fallback: skips model generation if no relevant context found  
- Simple Flask API and static frontend
---
