"""
Smart CBT retriever for Mira.
Uses existing `technique_inferred` and `topics` tags to filter
corpus by conversation phase, returning more appropriate references.
"""

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from typing import Optional


PHASE_TECHNIQUE_MAP = {
    'rapport': ['validation', 'general'],
    'assessment': ['validation', 'socratic_questioning', 'general'],
    'intervention': ['reframing', 'cbt', 'socratic_questioning'],
    'skills': ['behavioral_activation', 'cbt'],
    'crisis': []
}


class SmartCBTRetriever:
    """
    Phase + technique aware retriever for CBT-style therapy.
    Uses existing corpus tags - no data changes needed.
    """

    def __init__(
        self,
        corpus_path: str = "/app/rag/corpus.json",
        index_path: str = "/app/rag/corpus.faiss"
    ):
        with open(corpus_path) as f:
            self.corpus = json.load(f)

        self.faiss_index = faiss.read_index(index_path)

        print('Loading encoder: paraphrase-multilingual-MiniLM-L12-v2...')
        self.encoder = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )

        self.technique_indices = self._build_technique_indices()
        self.topic_indices = self._build_topic_indices()

        print(f'✓ SmartCBT Retriever loaded')
        print(f'  Corpus: {len(self.corpus)} entries')
        print(f'  Techniques: {dict((k, len(v)) for k, v in self.technique_indices.items())}')

    def _build_technique_indices(self):
        indices = defaultdict(list)
        for i, entry in enumerate(self.corpus):
            tech = entry.get('technique_inferred', 'general').lower()
            indices[tech].append(i)
        return dict(indices)

    def _build_topic_indices(self):
        indices = defaultdict(list)
        for i, entry in enumerate(self.corpus):
            for topic in entry.get('topics', []):
                indices[topic].append(i)
        return dict(indices)

    def retrieve(
        self,
        query: str,
        phase: str = 'rapport',
        topic_hint: Optional[str] = None,
        k: int = 3
    ):
        query_emb = self.encoder.encode([query])[0].astype('float32')
        query_emb = query_emb / np.linalg.norm(query_emb)

        technique_filter = PHASE_TECHNIQUE_MAP.get(phase, ['general', 'validation'])

        candidate_indices = set()
        for tech in technique_filter:
            candidate_indices.update(self.technique_indices.get(tech, []))

        if topic_hint and topic_hint in self.topic_indices:
            topic_indices_set = set(self.topic_indices[topic_hint])
            intersection = candidate_indices & topic_indices_set
            if len(intersection) >= k * 5:
                candidate_indices = intersection

        if not candidate_indices:
            return self._fallback_search(query_emb, k)

        scores, indices = self.faiss_index.search(
            query_emb.reshape(1, -1),
            min(len(self.corpus), 100)
        )

        filtered = [
            (float(score), int(idx))
            for score, idx in zip(scores[0], indices[0])
            if int(idx) in candidate_indices
        ]

        top = filtered[:k]

        results = []
        for score, idx in top:
            entry = self.corpus[idx]
            results.append({
                'id': entry['id'],
                'user_input': entry['user_input'],
                'counselor_response': entry['counselor_response'],
                'technique': entry.get('technique_inferred', 'general'),
                'topics': entry.get('topics', []),
                'score': score,
            })

        return results

    def _fallback_search(self, query_emb, k):
        scores, indices = self.faiss_index.search(
            query_emb.reshape(1, -1), k
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            entry = self.corpus[int(idx)]
            results.append({
                'id': entry['id'],
                'user_input': entry['user_input'],
                'counselor_response': entry['counselor_response'],
                'technique': entry.get('technique_inferred', 'general'),
                'topics': entry.get('topics', []),
                'score': float(score),
            })
        return results


_retriever = None

def get_smart_retriever():
    global _retriever
    if _retriever is None:
        _retriever = SmartCBTRetriever()
    return _retriever