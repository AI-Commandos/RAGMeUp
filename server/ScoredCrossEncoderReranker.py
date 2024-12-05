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
    top_n: int = 3
    """Number of documents to return."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        

    def get_feedback(self, document_id: str):
        """Fetch feedback for a specific document from the feedback database."""
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()

        cursor.execute(
            "SELECT query, answer, rating, timestamp FROM Feedback WHERE document_id = ?",
            (document_id,)
        )
        feedback = cursor.fetchall()

        conn.close()

        # Format feedback as a list of dictionaries
        feedback_list = [
            {"query": row[0], "answer": row[1], "rating": row[2], "timestamp": row[3]} for row in feedback
        ]

        return feedback_list
    

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using CrossEncoder.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """

        # scores = self.model.score([(query, doc.page_content) for doc in documents])
        # docs_with_scores = list(zip(documents, scores))
        
        # Score documents using CrossEncoder model
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores))

        print("docs_with_scores", docs_with_scores)
        print('Start with reranking en feedback')
        # Get feedback for each document and adjust score if needed
        for i, (doc, score) in enumerate(docs_with_scores):
            feedback = get_feedback(doc.metadata.get("document_id", ""))
            # You could adjust the score based on feedback here, e.g., adding the average rating to the score
            print('feedback in crossencoder', feedback)
            if feedback:
                avg_rating = sum(f["rating"] for f in feedback) / len(feedback)
                # Modify score based on feedback (this is just an example)
                docs_with_scores[i] = (doc, score + avg_rating)
        
        
        print('End with reranking en feedback')
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        print('result', result)
        return [doc.copy(update={"metadata": {**doc.metadata, "relevance_score": score}}) for doc, score in result[:self.top_n]]