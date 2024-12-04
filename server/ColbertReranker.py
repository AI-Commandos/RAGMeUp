from llama_index.core import VectorStoreIndex
from llama_index.core import Document as LlamaDocument
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from typing import Optional, Sequence
import operator


class ColbertReranker(BaseDocumentCompressor):

    top_n: int = 3
    """Number of documents to return."""

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks]=None
    ) -> Sequence[Document]:
        
        """
        Rerank documents using ColbertReranker.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        def convert_to_llama_documents(documents):
            return [
                LlamaDocument(text=doc.page_content, metadata=doc.metadata)
                for doc in documents
            ]
        llama_docs = convert_to_llama_documents(documents)
        # build index
        index = VectorStoreIndex.from_documents(documents=llama_docs)
        
        colbert_reranker = ColbertRerank(
            top_n=5,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            keep_retrieval_score=True,
        )

        query_engine = index.as_query_engine(
            similarity_top_k=10,
            node_postprocessors=[colbert_reranker],
        )
        # Perform the query
        response = query_engine.query(query)

        # Extract reranked documents and their scores
        docs_with_scores = [
            (doc.node, doc.score)  # doc.node is the document, doc.score is the reranking score
            for doc in response.source_nodes
        ]

        # Sort the documents by reranking score in descending order
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)

        # Return documents in the specified format
        return [
            doc.copy(
                update={
                    "metadata": {**doc.metadata, "relevance_score": score}
                }
            )
            for doc, score in result[:colbert_reranker.top_n]
        ]