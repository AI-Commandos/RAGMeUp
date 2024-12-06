from __future__ import annotations

import operator
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder

print('Running ScoredCrossEncoderReranker.py')
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
        print('Running compress_documents')
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = list(zip(documents, scores))
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        return [doc.copy(update={"metadata": {**doc.metadata, "relevance_score": score}}) for doc, score in result[:self.top_n]]


# from __future__ import annotations

# import operator
# from typing import Optional, Sequence

# from langchain_core.callbacks import Callbacks
# from langchain_core.documents import BaseDocumentCompressor, Document

# from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder


# class ScoredCrossEncoderReranker(BaseDocumentCompressor):
#     """Document compressor that uses CrossEncoder for reranking."""

#     model: BaseCrossEncoder
#     """CrossEncoder model to use for scoring similarity
#       between the query and documents."""
#     top_n: int = 3
#     """Number of documents to return."""

#     print('Running ScoredCrossEncoderReranker.py')

#     class Config:
#         arbitrary_types_allowed = True
#         extra = "forbid"
        
 

#     def compress_documents(
#         self,
#         documents: Sequence[Document],
#         query: str,
#         callbacks: Optional[Callbacks] = None,
#     ) -> Sequence[Document]:
#         """
#         Rerank documents using CrossEncoder.

#         Args:
#             documents: A sequence of documents to compress.
#             query: The query to use for compressing the documents.
#             callbacks: Callbacks to run during the compression process.

#         Returns:
#             A sequence of compressed documents.
#         """

#         # scores = self.model.score([(query, doc.page_content) for doc in documents])
#         # docs_with_scores = list(zip(documents, scores))
        
#         # Score documents using CrossEncoder model
#         scores = self.model.score([(query, doc.page_content) for doc in documents])
#         # Voeg min-max score toe als van o-1 wil je de score tussen 0 en 1 hebben
#         docs_with_scores = list(zip(documents, scores))

#         print("docs_with_scores", docs_with_scores)
#         print('Start with reranking en feedback')
#         # Get feedback for each document and adjust score if needed
#         for i, (doc, score) in enumerate(docs_with_scores):
#             feedback = get_feedback(doc.metadata.get("document_id", ""))
#             # You could adjust the score based on feedback here, e.g., adding the average rating to the score
#             print('feedback in crossencoder', feedback)
#             if feedback:
#                 avg_rating = sum(f["rating"] for f in feedback) / len(feedback)
#                 # Modify score based on feedback (this is just an example)
#                 docs_with_scores[i] = (doc, score + avg_rating)
        
        
#         result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
#         return [doc.copy(update={"metadata": {**doc.metadata, "relevance_score": score}}) for doc, score in result[:self.top_n]]
    
    
#     def main_in_class(self):
#         print('Running main_in_class')
#         docs = self.compress_documents()
#         print('docs:', docs)

        
# def main():
#     # Example usage
#     try:      
#         model = BaseCrossEncoder()  # Initialize your model here
#         print('model:', model)
#         reranker = ScoredCrossEncoderReranker(
#             model=model
#         )        
#         reranker.main_in_class(
#         )
    
#     except Exception as e:
#         print(f"Reranking failed: {e}")
    


# if __name__ == "__main__":
#     main()