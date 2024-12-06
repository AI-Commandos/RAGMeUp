from __future__ import annotations

import operator
from typing import Optional, Sequence, Dict, Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder

from Reranker import Reranker


class FeedbackAwareCrossEncoderReranker(BaseDocumentCompressor):
    """Document compressor that uses CrossEncoder for reranking with feedback integration."""

    model: BaseCrossEncoder
    """CrossEncoder model to use for scoring similarity between the query and documents."""
    top_n: int = 3
    """Number of documents to return."""
    feedback_df: Optional[Any] = None
    """DataFrame containing document feedback ratings."""
    feedback_weight: float = 0.5
    """Weight given to feedback ratings in final scoring (0-1)."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def __init__(self, model: BaseCrossEncoder, feedback_df: pd.DataFrame, feedback_weight: float = 0.5, top_n: int = 3):
        self.model = model
        self.feedback_df = Reranker.get_feedback_reranker()
        self.feedback_weight = feedback_weight
        self.top_n = top_n

        

    def _get_document_feedback_score(self, document: Document) -> float:
        """
        Retrieve the feedback score for a given document.
        
        Args:
            document: The document to find a feedback score for.
        
        Returns:
            The average feedback rating for the document, or 0.5 if no rating exists.
        """
        if self.feedback_df is None:
            print('No feedback data available in get_document_feedback_score')
            return 0.5

        # Assume the document source is the filename
        source = document.metadata.get('source', '')
        
        # Find matching rows in the feedback DataFrame
        matching_rows = self.feedback_df[self.feedback_df['document_id'] == source]
        print('matching_rows:', matching_rows)
        
        if matching_rows.empty:
            return 0.5
        
        # Calculate the average rating
        return matching_rows['total_rating'].mean()

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using CrossEncoder with integrated feedback scoring.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        # Get CrossEncoder relevance scores
        cross_encoder_scores = self.model.score([(query, doc.page_content) for doc in documents])
        
        # Compute combined scores
        docs_with_scores = []
        for doc, cross_encoder_score in zip(documents, cross_encoder_scores):
            # Get feedback score
            feedback_score = self._get_document_feedback_score(doc)
            
            # Combine scores with weighted average
            combined_score = (
                (1 - self.feedback_weight) * cross_encoder_score + 
                self.feedback_weight * feedback_score
            )
            
            docs_with_scores.append((doc, combined_score))
        
        # Sort documents by the combined score in descending order
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        
        # Return top N documents with updated metadata
        return [
            doc.copy(update={
                "metadata": {
                    **doc.metadata, 
                    "relevance_score": score,
                    "cross_encoder_score": cross_encoder_score,
                    "feedback_score": self._get_document_feedback_score(doc)
                }
            }) 
            for (doc, score), (_, cross_encoder_score) in zip(result[:self.top_n], result[:self.top_n])
        ]
