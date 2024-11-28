from typing import List, Dict, Any
from langchain_core.documents import Document, BaseDocumentCompressor
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert import Indexer, Searcher
import torch
import os
from ragatouille import RAGPretrainedModel


class ColBERTReranker(BaseDocumentCompressor):
    """
    A document compressor that uses ColBERT for reranking documents.
    """

    top_n: int = 5
    model: str = "colbert-ir/colbertv2.0"

    def compress_documents(
        self, documents: List[Document], query: str, callbacks=None
    ) -> List[Document]:
        """
        Rerank the documents using ColBERT.

        Args:
            documents (List[Document]): List of documents to rerank
            query (str): Query string
            callbacks: Callbacks for tracking progress (unused)

        Returns:
            List[Document]: Reranked documents
        """
        if not documents:
            return []

        # Prepare documents for reranking
        doc_texts = [doc.page_content for doc in documents]
        doc_ids = [str(i) for i in range(len(documents))]

        colbert = RAGPretrainedModel.from_pretrained(self.model)
        colbert.index(
            index_name="/content/RAGMeUp/server/indexed",
            collection=doc_texts,
            document_ids=doc_ids,
            document_metadatas=[doc.metadata for doc in documents],
        )
        results = colbert.search(query)

        # Return top_n documents
        return [
            {
                "page_content": result["content"],
                "metadata": {
                    **result["document_metadata"],
                    "relevance_score": result["score"],
                },
            }
            for result in sorted(results, key=lambda x: x["score"], reverse=True)[
                : self.top_n
            ]
        ]
