from __future__ import annotations

import operator
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder


class ScoredCrossEncoderReranker(BaseDocumentCompressor):
    """Document compressor that uses CrossEncoder for reranking."""

    model: BaseCrossEncoder
    """CrossEncoder model to use for scoring similarity
      between the query and documents."""
    top_n: int = 7
    """Number of documents to return."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    # Add the long_context_reorder function here
    def long_context_reorder(self, documents):
        """
        Reorders documents such that the most important ones are at the beginning
        and the end of the list (interleaved ordering).

        Args:
            documents: List of tuples [(Document, score), ...]

        Returns:
            List of reordered documents [(Document, score), ...]
        """
        n = len(documents)
        reordered = []

        # Interleave the documents: start with the most important, alternate between high and low.
        for i in range((n + 1) // 2):  # Go halfway through the list
            reordered.append(documents[i])  # Add the i-th most important document
            if i != n - i - 1:  # Avoid duplicating the middle document for odd-length lists
                reordered.append(documents[n - i - 1])  # Add the i-th least important document

        return reordered

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using CrossEncoder and apply long-context reordering.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        # Score documents using the CrossEncoder
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores))

        # Sort documents by their scores in descending order
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)

        # Apply long-context reordering
        reordered_result = self.long_context_reorder(result)

        # Return the reordered documents with updated metadata
        return [
            doc.copy(update={"metadata": {**doc.metadata, "relevance_score": score}})
            for doc, score in reordered_result[:self.top_n]
        ]