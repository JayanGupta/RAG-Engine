"""Sparse keyword search using BM25 (Best Matching 25).

Implemented cleanly in pure Python with optional rank_bm25 fallback:
Zero external dependencies required!

Why BM25?
While vector embeddings capture semantic meaning (synonyms, conceptual similarity),
BM25 captures exact lexical matches (rare keywords, acronyms, specific numbers).
"""

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+")


def tokenize(text):
    """Clean and tokenize text into lowercase words/terms."""
    if not text:
        return []
    return [t.lower() for t in TOKEN_RE.findall(text)]


class SimpleBM25:
    """Pure Python implementation of Okapi BM25.
    
    Formula:
      IDF(q) = ln((N - n(q) + 0.5) / (n(q) + 0.5) + 1)
      Score(D, Q) = sum( IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl))) )
    """

    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.corpus_size)

        # Document frequencies for each term
        self.doc_freqs = Counter()
        self.doc_term_counts = []

        for doc in corpus:
            counts = Counter(doc)
            self.doc_term_counts.append(counts)
            for term in counts.keys():
                self.doc_freqs[term] += 1

    def get_scores(self, query_tokens):
        """Calculate BM25 relevance score for each document in the corpus."""
        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.doc_freqs:
                continue

            # Inverse Document Frequency (IDF)
            df = self.doc_freqs[token]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

            # Term frequency & length normalization for each doc
            for doc_idx, term_counts in enumerate(self.doc_term_counts):
                tf = term_counts.get(token, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_len)))
                scores[doc_idx] += idf * (numerator / denominator)

        return scores


class BM25SearchIndex:
    """Keyword search index over chunks."""

    def __init__(self):
        self.chunks = []
        self.corpus_tokens = []
        self.bm25 = None

    def build(self, chunks):
        """Index a list of TextChunk objects."""
        if not chunks:
            self.chunks = []
            self.corpus_tokens = []
            self.bm25 = None
            return

        self.chunks = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            # Include text and document title for matching
            combined_text = f"{chunk.title} {chunk.text}"
            self.corpus_tokens.append(tokenize(combined_text))

        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except ImportError:
            # Use pure Python BM25 fallback
            self.bm25 = SimpleBM25(self.corpus_tokens)

    def search(self, query, top_k=5):
        """Find the top_k best matching chunks by BM25 keyword relevance.
        
        Returns:
            List of dicts: [{"chunk": TextChunk, "score": float, "source": "bm25"}]
        """
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        scored = list(enumerate(scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored[:top_k]:
            if score <= 0:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": round(float(score), 4),
                "source": "bm25",
            })
        return results

    def __len__(self):
        return len(self.chunks)
