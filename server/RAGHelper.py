import os
import torch
from tqdm import tqdm
import hashlib

from provenance import (compute_attention, compute_rerank_provenance, compute_llm_provenance, DocumentSimilarityAttribution)
from ScoredCrossEncoderReranker import ScoredCrossEncoderReranker

from transformers import BitsAndBytesConfig
from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig,
  pipeline,
)

from langchain_core.documents.base import Document
from langchain.chains.llm import LLMChain
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_milvus.vectorstores import Milvus
from langchain_postgres.vectorstores import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

from lxml import etree
import re
import pickle

# Make documents look a bit better than default
def formatDocuments(docs):
    doc_strings = []
    for i, doc in enumerate(docs):
        metadata_string = ", ".join([f"{md}: {doc.metadata[md]}" for md in doc.metadata.keys()])
        doc_strings.append(f"Document {i} content: {doc.page_content}\nDocument {i} metadata: {metadata_string}")
    return "\n\n<NEWDOC>\n\n".join(doc_strings)

class RAGHelper:
    # Loads the data and chunks it into an ensemble retriever
    def loadData(self):
        sparse_db_path = f"{os.getenv('vector_store_sparse')}"
        if os.path.exists(sparse_db_path):
            with open(sparse_db_path, 'rb') as f:
                self.chunked_documents = pickle.load(f)
        else:
            # Load PDF files if need be
            docs = []
            data_dir = os.getenv('data_directory')
            file_types = os.getenv("file_types").split(",")
            if "pdf" in file_types:
                loader = PyPDFDirectoryLoader(data_dir)
                docs = docs + loader.load()
            # Load JSON
            if "json" in file_types:
                text_content = True
                if os.getenv("json_text_content").lower() == 'false':
                    text_content = False
                loader_kwargs = {
                    'jq_schema': os.getenv("json_schema"),
                    'text_content': text_content
                }
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.json",
                    loader_cls=JSONLoader,
                    loader_kwargs=loader_kwargs,
                )
                docs = docs + loader.load()
            # Load CSV
            if "csv" in file_types:
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.csv",
                    loader_cls=CSVLoader,
                )
                docs = docs + loader.load()
            # Load MS Word
            if "docx" in file_types:
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.docx",
                    loader_cls=Docx2txtLoader,
                )
                docs = docs + loader.load()
            # Load MS Excel
            if "xlsx" in file_types:
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.xlsx",
                    loader_cls=UnstructuredExcelLoader,
                )
                docs = docs + loader.load()
            # Load MS PPT
            if "pptx" in file_types:
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.pptx",
                    loader_cls=UnstructuredPowerPointLoader,
                )
                docs = docs + loader.load()
            # Load XML, which is nasty
            if "xml" in file_types:
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.xml",
                    loader_cls=TextLoader,
                )
                xmldocs = loader.load()
                newdocs = []
                for index, doc in enumerate(xmldocs):
                    xmltree = etree.fromstring(doc.page_content.encode('utf-8'))
                    elements = xmltree.xpath(os.getenv("xml_xpath"))
                    elements = [etree.tostring(element, pretty_print=True).decode() for element in elements]
                    metadata = doc.metadata
                    metadata['index'] = index
                    newdocs = newdocs + [Document(page_content=doc, metadata=metadata) for doc in elements]
                docs = docs + newdocs

            if os.getenv('splitter') == 'RecursiveCharacterTextSplitter':
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=int(os.getenv('chunk_size')),
                    chunk_overlap=int(os.getenv('chunk_overlap')),
                    length_function=len,
                    keep_separator=True,
                    separators=[
                        "\n \n",
                        "\n\n",
                        "\n",
                        ".",
                        "!",
                        "?",
                        " ",
                        ",",
                        "\u200b",  # Zero-width space
                        "\uff0c",  # Fullwidth comma
                        "\u3001",  # Ideographic comma
                        "\uff0e",  # Fullwidth full stop
                        "\u3002",  # Ideographic full stop
                        "",
                    ],
                )
            elif os.getenv('splitter') == 'SemanticChunker':
                breakpoint_threshold_amount=None
                number_of_chunks=None
                if os.getenv('breakpoint_threshold_amount') != 'None':
                    breakpoint_threshold_amount=float(os.getenv('breakpoint_threshold_amount'))
                if os.getenv('number_of_chunks') != 'None':
                    number_of_chunks=int(os.getenv('number_of_chunks'))
                self.text_splitter = SemanticChunker(
                    self.embeddings,
                    breakpoint_threshold_type=os.getenv('breakpoint_threshold_type'),
                    breakpoint_threshold_amount=breakpoint_threshold_amount,
                    number_of_chunks=number_of_chunks
                )

            # Add a hash as ID to each document chunk's metadata
            self.chunked_documents = [
                Document(page_content=doc.page_content, 
                    metadata={**doc.metadata, 'id': hashlib.md5(doc.page_content.encode()).hexdigest()})
                        for doc in self.text_splitter.split_documents(docs)
            ]

            # Store these too, for our sparse DB
            with open(sparse_db_path, 'wb') as f:
                pickle.dump(self.chunked_documents, f)

        vector_store_uri = os.getenv('vector_store_uri')
        if os.getenv("vector_store") == "milvus":
            if os.getenv("vector_store_initial_load") == "True":
                self.db = Milvus.from_documents(
                    [], self.embeddings,
                    connection_args={"uri": vector_store_uri},
                    collection_name=os.getenv("vector_store_collection"),
                )
            else:
                # Load chunked documents into the Milvus index
                self.db = Milvus.from_documents(
                    [], self.embeddings,
                    drop_old=True,
                    connection_args={"uri": vector_store_uri},
                    collection_name=os.getenv("vector_store_collection"),
                )
        elif os.getenv("vector_store") == "postgres":
            self.db = PGVector(
                embeddings=self.embeddings,
                collection_name=os.getenv("vector_store_collection"),
                connection=vector_store_uri,
                use_jsonb=True,
            )
        else:
            raise Exception("Only milvus or postgres are supported as vector stores! Please set vector_store in your .env file.")

        # Add the documents 1 by 1 so we can track progress
        with tqdm(total=len(self.chunked_documents), desc="Vectorizing documents") as pbar:
            for d in self.chunked_documents:
                self.db.add_documents([d], ids=[d.metadata["id"]])
                pbar.update(1)

        # Now the BM25 retriever
        bm25_retriever = BM25Retriever.from_texts(
            [x.page_content for x in self.chunked_documents],
            metadatas=[x.metadata for x in self.chunked_documents]
        )

        # Set up the vector retriever
        retriever=self.db.as_retriever(
            search_type="mmr", search_kwargs = {'k': int(os.getenv("vector_store_k"))}
        )

        # Now combine them to do hybrid retrieval
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )
        # Set up the reranker
        self.rerank_retriever = None
        if os.getenv("rerank"):
            if os.getenv("rerank_model") == "flashrank":
                self.compressor = FlashrankRerank(top_n=int(os.getenv("rerank_k")))
            else:
                self.compressor = ScoredCrossEncoderReranker(
                    model=HuggingFaceCrossEncoder(model_name=os.getenv("rerank_model")),
                    top_n=int(os.getenv("rerank_k"))
                )
            
            self.rerank_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor, base_retriever=self.ensemble_retriever
            )