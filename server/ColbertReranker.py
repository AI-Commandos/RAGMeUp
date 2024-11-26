import operator
import torch
from typing import Sequence, Optional
from transformers import AutoTokenizer
from colbert.modeling.colbert import ColBERT
from colbert.infra.config import ColBERTConfig
from colbert.utils.utils import print_message
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from langchain.retrievers.document_compressors.cross_encoder import BaseCrossEncoder

class ColBERTReranker(BaseDocumentCompressor):
    """
    Document compressor that uses ColBERT for reranking.
    """

    model: ColBERT
    tokenizer: AutoTokenizer
    config: ColBERTConfig
    top_n: int = 3

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    def __init__(self, model_name_or_path: str = "/colbertv2.0/pytorch_mobel.bin", top_n: int = 3, colbert_config: Optional[ColBERTConfig] = None):
        """
        Initialize the ColBERTReranker with a ColBERT model.

        Args:
            model_name_or_path: Path to the pretrained ColBERT model.
            top_n: Number of top documents to return.
            colbert_config: Configuration for the ColBERT model.
        """
        self.top_n = top_n
        self.config = colbert_config or ColBERTConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = ColBERT.from_pretrained(model_name_or_path, config=self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Rerank documents using ColBERT.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        assert all(doc.page_content for doc in documents), "All documents must have non-empty content."

        # Tokenize the query
        query_tokens = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=self.config.query_maxlen).to(self.device)

        # Tokenize all documents in a batch
        document_contents = [doc.page_content for doc in documents]
        document_tokens = self.tokenizer(document_contents, return_tensors='pt', padding=True, truncation=True, max_length=self.config.doc_maxlen).to(self.device)

        # Compute query and document embeddings
        with torch.no_grad():
            query_embedding = self.model.query(query_tokens['input_ids'], query_tokens['attention_mask'])
            doc_embeddings = self.model.doc(document_tokens['input_ids'], document_tokens['attention_mask'])
            
            # Compute scores in parallel
            scores = self.model.score(query_embedding, doc_embeddings)

        # Pair documents with scores
        scores = scores.cpu().numpy()
        docs_with_scores = list(zip(documents, scores))

        # Sort and return top_n documents
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        return [doc.copy(update={"metadata": {**doc.metadata, "relevance_score": score}}) for doc, score in result[:self.top_n]]
