from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from typing import Optional, Sequence

import operator


class ColbertR():
    top_n: int = 3


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
        Rerank documents using ColbertReranker.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        # build index
        index = VectorStoreIndex.from_documents(documents=documents)
        
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

        response = query_engine.query(query)
        scores = [node.score for node in response.source_nodes] 
        print("What are scores")
        print(scores)
        docs_with_scores = list(zip(documents, scores))
        print("What are docs_with_scores")
        print(docs_with_scores)
        result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
        return [doc.copy(update={"metadata": {**doc.metadata, "relevance_score": score}}) for doc, score in result[:self.top_n]]