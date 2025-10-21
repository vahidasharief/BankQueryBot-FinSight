import os
import sys
import time
import json
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from vectorstore import VectorStore

# Constants
PDF_DIR = Path(__file__).parent / "pdfs"
INDEX_DIR = Path(__file__).parent / "index"
BANKS = ['hsbc', 'canara', 'icici', 'hdfc']

def process_bank_pdfs(bank: str):
    """Process all PDFs for a specific bank and create its vector store."""
    print(f"\nProcessing {bank.upper()} documents...")
    bank_dir = PDF_DIR / bank
    bank_index_dir = INDEX_DIR / bank
    
    if not bank_dir.exists():
        print(f"No documents found for {bank}")
        return False
        
    # Create bank-specific index directory
    bank_index_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize vector store for this bank
    vs = VectorStore(
        index_path=str(bank_index_dir / "faiss.index"),
        meta_path=str(bank_index_dir / "metadata.json"),
        dim=EMBED_DIM
    )
    
    # Process each PDF in the bank's directory
    all_chunks = []
    pdf_files = list(bank_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {bank_dir}")
        return False
        
    print(f"Found {len(pdf_files)} PDF files for {bank}")
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        chunks = extract_text_from_pdf(pdf_path)
        
        for chunk in chunks:
            chunk['bank'] = bank  # Add bank identifier to metadata
        
        all_chunks.extend(chunks)
    
    if not all_chunks:
        print(f"No text extracted from {bank} PDFs")
        return False
        
    print(f"Embedding {len(all_chunks)} chunks for {bank}...")
    
    # Prepare texts for embedding
    texts = [chunk['text'] for chunk in all_chunks]
    
    # Generate embeddings
    embeddings = embed_texts(
        texts,
        local_path=r"R:\BankingQueryChatbot\models\all-MiniLM-L6-v2"
    )
    
    if not embeddings:
        print(f"Failed to generate embeddings for {bank}")
        return False
        
    # Add to vector store
    vs.add(embeddings, all_chunks)
    vs.save()
    
    print(f"✓ Successfully processed {bank.upper()} documents:")
    print(f"  - Processed {len(pdf_files)} PDF files")
    print(f"  - Created {len(embeddings)} text chunks")
    print(f"  - Saved index to {bank_index_dir}")
    
    return True

def cleanup_indices():
    """Remove all existing vector store files."""
    try:
        if INDEX_DIR.exists():
            for bank in BANKS:
                bank_index_dir = INDEX_DIR / bank
                if bank_index_dir.exists():
                    for f in bank_index_dir.glob("*"):
                        f.unlink()
                    bank_index_dir.rmdir()
            print("Cleaned up existing indices")
    except Exception as e:
        print(f"Error during cleanup: {e}")
EMBED_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Recommended small batch size for CPU machines
EMBED_BATCH_SIZE = 8  # override previous 32 if present

_local_sbert = None

def _get_sbert(local_path=None, model_name="all-MiniLM-L6-v2"):
    """
    Load SentenceTransformer from a local path if provided, otherwise from the hub.
    Apply CPU-thread limits for Windows to avoid thrashing.
    """
    global _local_sbert
    if _local_sbert is not None:
        return _local_sbert

    # Limit MKL/OpenMP threads to avoid CPU oversubscription on Windows
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Try loading from local path first
    if local_path:
        try:
            print(f"[INFO] Loading SentenceTransformer from local path: {local_path}")
            _local_sbert = SentenceTransformer(local_path)
        except Exception as e:
            print(f"[WARN] Failed to load local SBERT at {local_path}: {e}. Falling back to hub model.", file=sys.stderr)
            _local_sbert = SentenceTransformer(model_name)
    else:
        print(f"[INFO] Loading SentenceTransformer model: {model_name}")
        _local_sbert = SentenceTransformer(model_name)

    # Warm-up: small encode to initialize layers & cache
    try:
        with torch.no_grad():
            t0 = time.time()
            _ = _local_sbert.encode(["hello world"], show_progress_bar=False, convert_to_numpy=True, batch_size=4)
            print(f"[INFO] SBERT warm-up done in {time.time()-t0:.2f}s")
    except Exception as e:
        print("[WARN] SBERT warm-up encode failed:", e, file=sys.stderr)

    return _local_sbert


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from PDF file and return list of chunks with metadata."""
    chunks = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                # Split into smaller chunks if needed (you can implement chunk splitting logic here)
                chunks.append({
                    'text': text.strip(),
                    'page': page_num + 1,
                    'source': pdf_path.name
                })
        return chunks
        print(f"[INFO] Processing PDF: {pdf_path.name}")
        doc = fitz.open(str(pdf_path))
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                # Split into smaller chunks if text is too long
                # This is a simple split - you might want to use a more sophisticated method
                words = text.split()
                for i in range(0, len(words), 150):
                    chunk = " ".join(words[i:i + 150])
                    if chunk.strip():
                        chunks.append(chunk)
                        
        doc.close()
        print(f"[INFO] Extracted {len(chunks)} text chunks from {pdf_path.name}")
        return chunks
    except Exception as e:
        print(f"[ERROR] Failed to process PDF {pdf_path}: {e}", file=sys.stderr)
        return []

def embed_texts(texts: list, local_path=r"R:\\BankingQueryChatbot\\models\\all-MiniLM-L6-v2", batch_size=EMBED_BATCH_SIZE):
    """
    Encode texts in small CPU-friendly batches and return list of float32 vectors.
    Use local_path to load a local model. Avoid big batches on CPU.
    """
    if not texts:
        return []

    sbert = _get_sbert(local_path=local_path, model_name=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))

    embs = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        t0 = time.time()
        try:
            with torch.no_grad():
                arr = sbert.encode(batch, show_progress_bar=False, convert_to_numpy=True, batch_size=batch_size)
        except Exception as e:
            print(f"[WARN] encode failed for batch at index {i}: {e}. Trying single-item fallback.", file=sys.stderr)
            # fallback: try one-by-one
            for s in batch:
                try:
                    with torch.no_grad():
                        single = sbert.encode([s], show_progress_bar=False, convert_to_numpy=True, batch_size=1)
                        embs.append(single[0].astype("float32").tolist())
                except Exception as e2:
                    print(f"[ERROR] single encode failed for chunk: {e2}", file=sys.stderr)
            print(f"[INFO] completed fallback single-item encodes for batch starting at {i}")
            continue

        # arr is numpy array (batch, dim)
        embs.extend([v.astype("float32").tolist() for v in arr])
        print(f"[INFO] Embedded batch {i//batch_size + 1}/{(n+batch_size-1)//batch_size} size={len(batch)} in {time.time()-t0:.2f}s")

    return embs


def cleanup():
    """
    Remove all vector store files to start fresh
    """
    try:
        index_file = INDEX_DIR/"faiss.index"
        meta_file = INDEX_DIR/"metadata.json"
        
        if index_file.exists():
            print(f"[INFO] Removing {index_file}")
            index_file.unlink()
            
        if meta_file.exists():
            print(f"[INFO] Removing {meta_file}")
            meta_file.unlink()
            
        print("[INFO] Cleanup completed successfully")
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}", file=sys.stderr)

def validate_embeddings(embeddings, dim=EMBED_DIM):
    """
    Validate that embeddings match expected dimension
    """
    if not embeddings:
        return False
    if isinstance(embeddings[0], list):
        actual_dim = len(embeddings[0])
    else:
        actual_dim = len(embeddings)
    
    if actual_dim != dim:
        print(f"[ERROR] Embedding dimension mismatch. Expected {dim}, got {actual_dim}", file=sys.stderr)
        return False
    return True

def ingest_all(reprocess=False):
    """Process PDFs for all banks."""
    if reprocess:
        cleanup_indices()
    
    # Create main index directory
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track successful processing
    processed_banks = []
    
    # Process each bank
    for bank in BANKS:
        if process_bank_pdfs(bank):
            processed_banks.append(bank)
    
    # Save list of processed banks
    with open(INDEX_DIR / "banks.json", "w") as f:
        json.dump({
            "banks": processed_banks,
            "total_banks": len(processed_banks),
            "last_updated": str(Path(__file__).stat().st_mtime)
        }, f, indent=2)
    
    if processed_banks:
        print("\n✓ Indexing complete!")
        print(f"Successfully processed {len(processed_banks)} banks: {', '.join(processed_banks)}")
    else:
        print("\n⚠ No banks were successfully processed")
        print("Please ensure PDF documents are present in the bank-specific folders")
    """
    Process all PDFs in the pdfs directory
    :param reprocess: If True, force reprocessing of all files
    """
    # Ensure directories exist
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up if reprocessing
    if reprocess:
        cleanup()

    # Find all PDF files
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    # Initialize vector store
    vs = VectorStore(index_path=str(INDEX_DIR/"faiss.index"), meta_path=str(INDEX_DIR/"metadata.json"), dim=EMBED_DIM)
    metas = vs.load_metadata() or []

    if not pdf_files:
        return

    # Process each PDF
    for pdf_path in pdf_files:
        pdf_name = pdf_path.name
        print(f"\n[INFO] Processing {pdf_name}")

        # Skip if already processed (unless reprocess=True)
        if not reprocess and any(meta.get("source") == pdf_name for meta in metas):
            print(f"[INFO] Skipping {pdf_name} - already processed")
            continue

        # Extract text chunks from PDF
        chunks = extract_text_from_pdf(pdf_path)
        if not chunks:
            print(f"[WARN] No text extracted from {pdf_name}", file=sys.stderr)
            continue

        print(f"[INFO] Embedding {len(chunks)} chunks from {pdf_name}")
        # Generate embeddings
        embeddings = embed_texts(chunks)
        if not embeddings:
            print(f"[ERROR] Failed to generate embeddings for {pdf_name}", file=sys.stderr)
            continue

        # Create metadata for each chunk
        chunk_metas = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_metas.append({
                "text": chunk,
                "source": pdf_name,
                "chunk_id": i
            })

        # Validate embeddings before adding
        if not validate_embeddings(embeddings):
            print(f"[ERROR] Skipping {pdf_name} due to invalid embeddings", file=sys.stderr)
            continue

        try:
            # Add to vector store
            vs.add(embeddings, chunk_metas)
            print(f"[INFO] Successfully added {len(embeddings)} vectors from {pdf_name}")
            # Save after each file successfully processed
            vs.save()
        except Exception as e:
            print(f"[ERROR] Failed to add vectors for {pdf_name}: {e}", file=sys.stderr)
            continue

    print("[INFO] ingest_all completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process PDFs and create vector store')
    parser.add_argument('--reprocess', action='store_true', help='Force reprocessing of all files')
    parser.add_argument('--cleanup', action='store_true', help='Clean up vector store files and exit')
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
    else:
        ingest_all(reprocess=args.reprocess)