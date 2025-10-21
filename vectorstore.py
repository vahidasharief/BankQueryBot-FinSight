# vectorstore.py
import faiss
import numpy as np
import json
from pathlib import Path

class VectorStore:
    def __init__(self, index_path="index/faiss.index", meta_path="index/metadata.json", dim=1536):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.dim = dim
        self.index = None
        self.metadata = []
        self._load_or_create()

    def _load_or_create(self):
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.metadata = self._load_meta()
            except Exception:
                # fallback to new index if corrupted
                self.index = faiss.IndexFlatIP(self.dim)
                self.metadata = []
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    def _load_meta(self):
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def add(self, vectors, metas):
        """
        Add vectors and metadata to the index
        :param vectors: List of vectors or single vector
        :param metas: List of metadata dicts or single dict
        """
        # Convert to numpy array and ensure 2D
        v = np.array(vectors, dtype='float32')
        if v.ndim == 1:
            v = v.reshape(1, -1)
            metas = [metas]
        
        # Validate dimensions
        if v.shape[1] != self.dim:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dim}, got {v.shape[1]}")
        
        # Normalize vectors
        faiss.normalize_L2(v)
        
        # Add to index and metadata
        self.index.add(v)
        self.metadata.extend(metas)

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def search(self, vector, top_k=5):
        v = np.array(vector, dtype='float32').reshape(1, -1)
        faiss.normalize_L2(v)
        D, I = self.index.search(v, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx].copy()
            meta['score'] = float(score)
            results.append(meta)
        return results

    # compatibility helpers
    def save_metadata(self, metas):
        self.metadata = metas
        self.save()

    def load_metadata(self):
        return self._load_meta()

    def commit(self):
        self.save()
