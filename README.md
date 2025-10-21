# 🧠 FinSight — Intelligent Banking Chatbot (RAG-based)

**FinSight** (BankQueryBot-FinSight) is an intelligent **Retrieval-Augmented Generation (RAG)** chatbot designed to handle real-world **banking domain queries**.  
It provides accurate, document-grounded answers to customer questions using official bank PDFs — with full citation transparency.

---

## 🚀 Core Functionalities

### 🏦 1. Bank-Specific Knowledge
FinSight supports multiple major Indian banks:
- **HSBC**
- **HDFC**
- **ICICI**
- **Canara Bank**

Each bank has its own FAISS vector index built from official PDFs, enabling domain-specific responses.  
No cross-contamination — HSBC answers never come from HDFC docs.

---

### ⚙️ 2. Local Embedding + Vector Search
- Uses **SentenceTransformer (all-MiniLM-L6-v2)** to generate semantic embeddings.  
- Employs **FAISS** for vector similarity search to retrieve the most relevant document chunks.  
- Retrieval is precise, efficient, and runs locally — **no external APIs or internet** required.

---

### 💬 3. Local LLM via Ollama
- Powered by **Gemma 3:1B** model through **Ollama**, running completely offline.  
- Generates context-aware, citation-based answers using retrieved content.  
- If no relevant context is found (score < 0.10), the LLM step is skipped — ensuring **zero hallucinations**.

---

### 🧩 4. RAG Pipeline Overview

**Step 1: Query Embedding**
User question → converted into embedding via SBERT.

**Step 2: Retrieval**
FAISS searches the respective bank index for top matching chunks.

**Step 3: Context Assembly**
Top results + metadata are fed into a dynamic, structured prompt.

**Step 4: Generation**
Ollama (Gemma 3) analyzes the context and produces a cited, explanatory answer.

**Step 5: Fallback**
If retrieval confidence is too low → immediate fallback message:
> “No sufficiently relevant context found in the HDFC Bank documents. Please contact the bank at 1800 202 6161.”

---

### 🧠 5. Smart Context Handling
- Dynamically formats context for each query using `prompts.py`.
- Citations automatically generated as `[file.pdf - page X]`.
- Expands on related sections for better readability.
- Avoids hallucinations with strict document grounding.

---

### 🔍 6. Adaptive Response Control
- **Score thresholding:** LLM only invoked when top similarity ≥ 0.10.  
- **Top-k retrieval:** Configurable to fine-tune performance vs. precision.  
- **Bank separation:** Each bank’s FAISS index is independent.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Static HTML/JS (Flask static) |
| **Backend Framework** | Flask (Python) |
| **Embeddings** | SentenceTransformers (SBERT) |
| **Vector Database** | FAISS |
| **LLM** | Gemma 3:1B via Ollama |
| **Storage** | JSON metadata + FAISS index per bank |
| **Language** | Python 3.9+ |

---

## 🧱 Architecture Diagram (Conceptual)

            ┌────────────────────┐
            │  User Query (Text) │
            └─────────┬──────────┘
                      │
             (1) Generate Embedding
                      │
         ┌────────────▼────────────┐
         │   FAISS Vector Store    │
         │ (Per Bank PDF Indexes)  │
         └────────────┬────────────┘
                      │
             (2) Retrieve Context
                      │
         ┌────────────▼────────────┐
         │   Prompt Constructor    │
         │  (make_retrieval_prompt)│
         └────────────┬────────────┘
                      │
             (3) Generate Answer
                      │
         ┌────────────▼────────────┐
         │  Ollama + Gemma 3:1B    │
         │  Local LLM Inference    │
         └────────────┬────────────┘
                      │
           (4) Return Final Answer

---

## 🧩 Example Outputs

### Example 1 — Relevant Query
