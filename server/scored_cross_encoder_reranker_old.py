from __future__ import annotations
import pandas as pd

import operator
from typing import Optional, Sequence, Dict, Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder

from server.Reranker_1 import Reranker

print('Running scored-encoder-reranker.py')

class FeedbackAwareCrossEncoderReranker(BaseDocumentCompressor):
    """Document compressor that uses CrossEncoder for reranking with feedback integration."""

    model: BaseCrossEncoder
    """CrossEncoder model to use for scoring similarity between the query and documents."""
    top_n: int = 3
    """Number of documents to return."""
    encoder_feedback_df: Optional[Any] = None
    """DataFrame containing document feedback ratings."""
    feedback_weight: float = 0.5
    """Weight given to feedback ratings in final scoring (0-1)."""

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def __init__(self, model: BaseCrossEncoder, feedback_db='feedback.db', feedback_weight= 0.5, top_n = 3):
        super().__init__()
        self.model = model
        self.feedback_db = feedback_db        
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
        encoder_feedback_df = Reranker.get_feedback_reranker()
        
        if encoder_feedback_df is None:
            print('No feedback data available in get_document_feedback_score')
            return 0.5

        # Assume the document source is the filename
        source = document.metadata.get('source', '')
        
        # Find matching rows in the feedback DataFrame
        matching_rows = encoder_feedback_df[encoder_feedback_df['document_id'] == source]
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
            print('feedback_score:', feedback_score)
            
            # Combine scores with weighted average
            combined_score = (
                (1 - self.feedback_weight) * cross_encoder_score + 
                self.feedback_weight * feedback_score
            )
            
            docs_with_scores.append((doc, combined_score))
            print('docs_with_scores:', docs_with_scores)
        
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
        
    def main_in_class(self):
        docs = self.compress_documents()
        # documents_lst = self.get_documents_reranker()
        # combined_df = self.combiner(feedback_df, documents_lst)
        # query = 'What is Word2Vec?'
        # rerank_df = self.rerank_documents_with_feedback(query, documents_lst, feedback_df, combined_df)
        print('docs:', docs)

        
def main():
    # Example usage
    try:      
        Encoder_reranker = FeedbackAwareCrossEncoderReranker(
        model=BaseCrossEncoder,
        feedback_db='feedback.db',
        feedback_weight=0.7,
        top_n=5
        )
        Encoder_reranker.main_in_class(
        )
    
    except Exception as e:
        print(f"Reranking failed: {e}")
    

if __name__ == "__main__":
    main()