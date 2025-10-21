# app.py
import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from vectorstore import VectorStore
from prompts import make_retrieval_prompt, BANK_SUPPORT
from dotenv import load_dotenv
import pathlib
from ollama_helper import generate_chat_completion, check_ollama_health, format_chat_prompt

# Bank-specific vector stores
BANK_VS = {}

# local embedding libs
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

load_dotenv()

# Configuration
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")  # default to local SBERT
LOCAL_EMBED_PATH = os.getenv("LOCAL_EMBED_PATH", r"R:\BankingQueryChatbot\models\all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

# Check Ollama availability
if not check_ollama_health():
    print("[WARNING] Ollama service is not available or gemma3:1b model is not installed. "
          "Please ensure Ollama is running and the model is installed using: ollama pull gemma3:1b")

app = Flask(__name__, static_folder="static", static_url_path="/static")
BANK_VS = {}

def load_bank_vectorstores():
    """Load vector stores for all available banks"""
    index_dir = Path("index")
    if not index_dir.exists():
        return

    # Load bank list
    try:
        with open(index_dir / "banks.json") as f:
            banks = json.load(f)["banks"]
    except:
        banks = ["hsbc", "canara", "icici", "hdfc"]  # default list

    # Load vector store for each bank
    for bank in banks:
        bank_index_dir = index_dir / bank
        if bank_index_dir.exists():
            try:
                BANK_VS[bank] = VectorStore(
                    index_path=str(bank_index_dir / "faiss.index"),
                    meta_path=str(bank_index_dir / "metadata.json"),
                    dim=EMBED_DIM
                )
                print(f"[INFO] Loaded vector store for {bank.upper()}")
            except Exception as e:
                print(f"[WARNING] Failed to load vector store for {bank}: {e}")

# Initialize vector stores
load_bank_vectorstores()

# load local embedding model once (if configured)
_sbert = None
def get_local_embedder():
    global _sbert
    if _sbert is not None:
        return _sbert
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed but EMBED_MODEL is set to local.")
    # Prefer explicit local path if it exists; otherwise try model name
    if os.path.isdir(LOCAL_EMBED_PATH):
        print(f"[INFO] Loading SentenceTransformer from local path: {LOCAL_EMBED_PATH}")
        _sbert = SentenceTransformer(LOCAL_EMBED_PATH)
    else:
        print(f"[INFO] Loading SentenceTransformer by model name: {EMBED_MODEL}")
        _sbert = SentenceTransformer(EMBED_MODEL)
    return _sbert

def embed_texts_for_query(texts):
    """
    Use local SBERT embeddings for query embedding
    """
    model = get_local_embedder()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [v.astype("float32").tolist() for v in vecs]

@app.route("/api/query", methods=["POST"])
def query():
    data = request.json or {}
    question = data.get("question")
    top_k = int(data.get("top_k", 5))
    bank = data.get("bank", "hsbc").lower()
    
    if not question:
        return jsonify({"error": "question required"}), 400
    if bank not in BANK_VS:
        return jsonify({"error": f"No data available for {bank.upper()} bank"}), 400

    try:
        # Generate embeddings and search
        q_vec = embed_texts_for_query([question])[0]
        hits = BANK_VS[bank].search(q_vec, top_k=top_k)
        retrieved = [{"source": h['source'], "text": h.get('text',''), "score": h.get('score',0), 
                     "chunk_id": h.get('chunk_id', 0)} for h in hits]
        
        # Create context from retrieved documents
        context = "\n\n---\n".join(
            f"Source: {c['source']} (chunk {c['chunk_id']})\n{c['text']}" 
            for c in retrieved
        )
        
        # Generate response using Ollama (skip generation if no good context)
        try:
            # --- Skip if no results at all ---
            if not retrieved:
                return jsonify({
                    "answer": f"No context found in the {bank.upper()} Bank documents. "
                              f"Please reach out to the bank at {BANK_SUPPORT[bank]}.",
                    "retrieved": retrieved,
                    "from_llm": False
                })

            # --- Skip if best match score is too low ---
            try:
                top_score = max(c.get("score", 0) for c in retrieved)
            except Exception:
                top_score = 0.0

            SCORE_THRESHOLD = 0.10  # below 10% similarity → skip generation
            if top_score < SCORE_THRESHOLD:
                return jsonify({
                    "answer": (f"No sufficiently relevant context found in the {bank.upper()} Bank documents "
                               f"(top match score={top_score:.3f}). Please contact the bank at {BANK_SUPPORT[bank]}."),
                    "retrieved": retrieved,
                    "from_llm": False,
                    "top_score": top_score
                })

            # --- Otherwise, build prompt and generate as usual ---
            formatted_prompt = format_chat_prompt(
                system_prompt=make_retrieval_prompt(question, retrieved, bank),
                user_query=question,
                context=context
            )

            answer = generate_chat_completion(
                prompt=formatted_prompt,
                model=OLLAMA_MODEL,
                temperature=0.0
            )

            return jsonify({
                "answer": answer,
                "retrieved": retrieved,
                "from_llm": True,
                "top_score": top_score
            })

        except ConnectionError as e:
            return jsonify({
                "error": "Failed to connect to Ollama service. Please ensure it's running.",
                "retrieved": retrieved,
                "fallback_response": (
                    "I found some relevant information but cannot generate a response "
                    "because the local LLM service is not available. Please check the "
                    "retrieved content directly or ensure Ollama is running."
                )
            }), 503
            
        except Exception as e:
            return jsonify({
                "error": str(e),
                "retrieved": retrieved,
                "fallback_response": "I encountered an error while generating the response. Please review the retrieved information directly or try again later."
            }), 503
            
    except Exception as e:
        return jsonify({
            "error": f"Error during search: {str(e)}",
            "retrieved": [],
            "fallback_response": "An error occurred while searching the knowledge base. Please try again later."
        }), 500

@app.route("/api/ingest", methods=["POST"])
def ingest():
    if 'file' not in request.files:
        return jsonify({"error": "file required"}), 400
    f = request.files['file']
    fname = f.filename
    save_dir = pathlib.Path("pdfs")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / fname
    f.save(save_path)
    # call ingestion
    from ingest_pdfs import ingest_all
    ingest_all()
    return jsonify({"status": "ingested", "file": fname})

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    if path == "" or path == "index.html":
        return send_from_directory("static", "index.html")
    return send_from_directory("static", path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
