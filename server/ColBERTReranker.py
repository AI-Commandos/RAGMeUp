from typing import List, Dict, Any
from langchain.retrievers.document_compressors import BaseDocumentCompressor
from langchain_core.documents import Document
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert import Indexer, Searcher
import torch
import os


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

        self.top_n = top_n
        self.device = device

        # Initialize ColBERT configuration
        self.colbert_config = ColBERTConfig(
            model_name=model_name,
            nbits=nbits,
            doc_maxlen=doc_maxlen,
            query_maxlen=query_maxlen,
        )

        # Initialize tokenizers
        self.query_tokenizer = QueryTokenizer(self.colbert_config)
        self.doc_tokenizer = DocTokenizer(self.colbert_config)

        # Load model
        with Run().context(RunConfig(nranks=1)):
            self.searcher = Searcher(
                index=None, config=self.colbert_config, device=self.device
            )

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

        # Tokenize query and documents
        Q = self.query_tokenizer.tokenize(query)
        D = [self.doc_tokenizer.tokenize(doc) for doc in doc_texts]

        # Move tensors to device
        Q = Q.to(self.device)
        D = [d.to(self.device) for d in D]

        # Get scores
        with torch.no_grad():
            Q_encodings = self.searcher.model.query(Q)
            D_encodings = [self.searcher.model.doc(d) for d in D]
            scores = [
                self.searcher.model.score(Q_encodings, d_enc).cpu().item()
                for d_enc in D_encodings
            ]

        # Sort documents by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top_n documents
        return [doc for doc, _ in scored_docs[: self.top_n]]
