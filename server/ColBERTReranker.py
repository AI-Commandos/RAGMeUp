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

    def __init__(
            self,
            model_name: str = "colbert-ir/colbertv2.0",
            top_n: int = 5,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            nbits: int = 2,
            doc_maxlen: int = 180,
            query_maxlen: int = 32,
    ):
        """
        Initialize the ColBERT reranker.

        Args:
            model_name (str): Name or path of the ColBERT model to use
            top_n (int): Number of documents to return after reranking
            device (str): Device to run the model on ('cuda' or 'cpu')
            nbits (int): Number of bits for quantization
            doc_maxlen (int): Maximum document length
            query_maxlen (int): Maximum query length
        """
        super().__init__()

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

        kutding = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        kutding.index(index_name='/content/RAGMeUp/server/indexed',
                      collection=doc_texts,
                      document_ids=doc_ids,
                      document_metadatas=[doc.metadata for doc in documents])
        results = kutding.search(query)

        # Return top_n documents
        return [{'page_content': result['content'],'metadata': {**result['document_metadata'], 'relevance_score': result['score']}} for result in
                sorted(results, key=lambda x: x['score'], reverse = True)[:self.top_n]]
