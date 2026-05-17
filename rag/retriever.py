#!/usr/bin/env python3
"""
RAG retriever for Mira
Loads corpus + FAISS index, returns top-K matches
"""

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

class CBTRetriever:
    def __init__(self,
                 corpus_path: str = "/app/rag/corpus.json",
                 index_path: str = "/app/rag/corpus.faiss",
                 encoder_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):

        # Load corpus
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path) as f:
            self.corpus = json.load(f)

        # Load FAISS index
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)

        # Load encoder
        print(f"Loading encoder: {encoder_name}...")
        self.encoder = SentenceTransformer(encoder_name)

        embedding_dim = self.encoder.get_sentence_embedding_dimension()

        print(f"✓ CBT Retriever loaded: {len(self.corpus)} entries, dim={embedding_dim}")

    def retrieve(self, query: str, k: int = 3) -> list:
        """Retrieve top-K most relevant references"""
        # Encode query
        query_emb = self.encoder.encode([query], convert_to_numpy=True).astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(query_emb)

        # Search
        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            entry = self.corpus[idx]
            results.append({
                "score": float(score),
                "source": entry.get("source", "unknown"),
                "user_input": entry.get("user_input", ""),
                "counselor_response": entry.get("counselor_response", ""),
                "topics": entry.get("topics", []),
                "technique": entry.get("technique_inferred", "general")
            })

        return results

# Singleton instance
_retriever = None

def get_retriever() -> CBTRetriever:
    """Get or create singleton retriever instance"""
    global _retriever
    if _retriever is None:
        _retriever = CBTRetriever()
    return _retriever


if __name__ == "__main__":
    # Test retriever
    print("=" * 60)
    print("Testing CBT Retriever")
    print("=" * 60)

    retriever = get_retriever()

    test_query = "ผมเครียดงานมาก ไม่ได้นอน"
    print(f"\nTest query: {test_query}")

    results = retriever.retrieve(test_query, k=3)

    print(f"\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result['score']:.3f}")
        print(f"    Topics: {', '.join(result['topics'])}")
        print(f"    Technique: {result['technique']}")
        print(f"    User: {result['user_input'][:100]}...")
        print(f"    Response: {result['counselor_response'][:100]}...")
