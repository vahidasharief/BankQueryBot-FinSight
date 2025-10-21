# 🧠 FinSight — Intelligent Banking Chatbot (RAG-based)

**FinSight** (BankQueryBot-FinSight) is an intelligent **Retrieval-Augmented Generation (RAG)** chatbot designed to handle real-world **banking domain queries**.  
It provides accurate, document-grounded answers to customer questions using official bank PDFs — with full citation transparency.

---

## 🚀 Core Functionalities

### 🏦 1. Bank-Specific Knowledge
FinSight supports multiple major Indian banks:
- **HSBC**
- **Canara Bank**
- **HDFC**
- **ICICI**


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
**Query:**  
> “How much house loan hsbc provides”

**Response:**  
<img width="3200" height="1900" alt="resposne_hsbc1" src="https://github.com/user-attachments/assets/3c670f5b-82d6-40e0-a091-c88b4b1a163d" />


### Example 2 — Relavant Query 2
**Query:**  
> “What are the intrest changes on the Personal Loan amout for EMI”

**Response:**  
<img width="3196" height="1936" alt="response2_hsbc" src="https://github.com/user-attachments/assets/90b9c362-bd42-4a61-bbc6-c305784d36b0" />

### Example 3 — Relavant Query 3
**Query:**  
> “What are the documents required for personal loan?”

**Response:**  
<img width="3200" height="1998" alt="HSDF_RES1" src="https://github.com/user-attachments/assets/e86ed8a4-aadc-4792-b5c3-76ff86bdaede" />

.
### Example 3 — Irrelavant Query 1
**Query:**  
> “Who invented AI?”

**Response:**  
<img width="3200" height="1928" alt="irrevant" src="https://github.com/user-attachments/assets/b5294287-7a55-492f-ab50-980eeeb4fd68" />

### Example 4 — Irrelavant Query 2
**Query:**  
> “Who won recent cricket match between ind vs pak”

**Response:**  
<img width="3188" height="1946" alt="irrelavant2" src="https://github.com/user-attachments/assets/507eca1a-489b-4aab-a4da-d60b61ec4e43" />





## 🧠 Key Strengths
- 100% Local & Private — no cloud calls, no data leakage.  
- Modular — each bank operates on its own index.  
- Configurable — easily extend to more banks or domains.  
- Interview-ready project — demonstrates RAG design, embeddings, LLM integration, and safe fallbacks.

---

## 🏁 Summary
FinSight showcases a real-world RAG workflow implemented end-to-end:
> **Retrieval (FAISS)** → **Augmentation (context assembly)** → **Generation (Gemma 3 LLM)**

It’s a perfect example of blending information retrieval with generative AI —  
scalable, auditable, and practical for enterprise banking solutions.

---

## 📸 Screenshots Placeholder
*(Add them before committing for visual impact)*  
