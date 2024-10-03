import os
import pickle
import hashlib
from tqdm import tqdm
from lxml import etree
from langchain_core.documents.base import Document
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_milvus.vectorstores import Milvus
from langchain_postgres.vectorstores import PGVector
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import (
    PyPDFLoader,
    PyPDFDirectoryLoader,
    JSONLoader,
    DirectoryLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    CSVLoader
)
from ScoredCrossEncoderReranker import ScoredCrossEncoderReranker


class RAGHelper:
    """
    A helper class to manage retrieval-augmented generation (RAG) processes,
    including data loading, chunking, vector storage, and retrieval.
    """

    def __init__(self, logger):
        """
        Initializes the RAGHelper class and loads environment variables.
        """
        self.logger = logger
        self.chunked_documents = []
        self.embeddings = None  # Placeholder for embeddings; set during initialization
        self.text_splitter = None
        self.db = None
        self.sparse_retriever = None
        self.ensemble_retriever = None
        self.rerank_retriever = None

        # Load environment variables
        self.vector_store_sparse_uri = os.getenv('vector_store_sparse_uri')
        self.vector_store_uri = os.getenv('vector_store_uri')
        self.document_chunks_pickle = os.getenv('document_chunks_pickle')
        self.data_dir = os.getenv('data_directory')
        self.file_types = os.getenv("file_types").split(",")
        self.splitter_type = os.getenv('splitter')
        self.vector_store = os.getenv("vector_store")
        self.vector_store_initial_load = os.getenv("vector_store_initial_load") == "True"
        self.rerank = os.getenv("rerank") == "True"
        self.rerank_model = os.getenv("rerank_model")
        self.rerank_k = int(os.getenv("rerank_k"))

    @staticmethod
    def format_documents(docs):
        """
        Formats the documents for better readability.

        Args:
            docs (list): List of Document objects.

        Returns:
            str: Formatted string representation of documents.
        """
        doc_strings = []
        for i, doc in enumerate(docs):
            metadata_string = ", ".join([f"{md}: {doc.metadata[md]}" for md in doc.metadata.keys()])
            doc_strings.append(f"Document {i} content: {doc.page_content}\nDocument {i} metadata: {metadata_string}")
        return "\n\n<NEWDOC>\n\n".join(doc_strings)

    def _load_chunked_documents(self):
        """Loads previously chunked documents from a pickle file."""
        with open(self.document_chunks_pickle, 'rb') as f:
            self.chunked_documents = pickle.load(f)

    def _load_json_files(self):
        """
        Loads JSON files from the data directory.

        Returns:
            list: A list of loaded Document objects from JSON files.
        """
        text_content = os.getenv("json_text_content").lower() == 'true'
        loader_kwargs = {
            'jq_schema': os.getenv("json_schema"),
            'text_content': text_content
        }
        loader = DirectoryLoader(
            path=self.data_dir,
            glob="*.json",
            loader_cls=JSONLoader,
            loader_kwargs=loader_kwargs,
            recursive=True,
            show_progress=True,
        )
        return loader.load()

    def _load_xml_files(self):
        """
        Loads XML files from the data directory and extracts relevant elements.

        Returns:
            list: A list of Document objects created from XML elements.
        """
        loader = DirectoryLoader(
            path=self.data_dir,
            glob="*.xml",
            loader_cls=TextLoader,
            recursive=True,
            show_progress=True,
        )
        xmldocs = loader.load()
        newdocs = []
        for index, doc in enumerate(xmldocs):
            try:
                xmltree = etree.fromstring(doc.page_content.encode('utf-8'))
                elements = xmltree.xpath(os.getenv("xml_xpath"))
                elements = [etree.tostring(element, pretty_print=True).decode() for element in elements]
                metadata = doc.metadata
                metadata['index'] = index
                newdocs += [Document(page_content=content, metadata=metadata) for content in elements]
            except Exception as e:
                print(f"Error processing XML document: {e}")
        return newdocs

    def _load_documents(self):
        """
        Loads documents from specified file types in the data directory.

        Returns:
            list: A list of loaded Document objects.
        """
        docs = []
        for file_type in self.file_types:
            try:
                if file_type == "pdf":
                    loader = PyPDFDirectoryLoader(self.data_dir)
                    docs += loader.load()
                elif file_type == "json":
                    docs += self._load_json_files()
                elif file_type == "csv":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.csv",
                        loader_cls=CSVLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    docs += loader.load()
                elif file_type == "docx":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.docx",
                        loader_cls=Docx2txtLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    docs += loader.load()
                elif file_type == "xlsx":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.xlsx",
                        loader_cls=UnstructuredExcelLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    docs += loader.load()
                elif file_type == "pptx":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.pptx",
                        loader_cls=UnstructuredPowerPointLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    docs += loader.load()
                elif file_type == "xml":
                    docs += self._load_xml_files()
            except Exception as e:
                print(f"Error loading {file_type} files: {e}")
        return docs

    @staticmethod
    def _load_json_document(filename):
        """Load JSON documents with specific parameters"""
        return JSONLoader(
            file_path=filename,
            jq_schema=os.getenv("json_schema"),
            text_content=os.getenv("json_text_content") == "True"
        )

    def _load_document(self, filename):
        """Load documents from the specified file based on its extension."""
        file_type = filename.lower().split('.')[-1]
        loaders = {
            'pdf': PyPDFLoader,
            'json': self._load_json_document,
            'csv': CSVLoader,
            'docx': Docx2txtLoader,
            'xlsx': UnstructuredExcelLoader,
            'pptx': UnstructuredPowerPointLoader
        }

        if file_type in loaders:
            return loaders[file_type](filename).load()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def _create_recursive_text_splitter():
        """
        Creates an instance of RecursiveCharacterTextSplitter.

        Returns:
            RecursiveCharacterTextSplitter: A configured text splitter instance.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('chunk_size')),
            chunk_overlap=int(os.getenv('chunk_overlap')),
            length_function=len,
            keep_separator=True,
            separators=[
                "\n \n", "\n\n", "\n", ".", "!", "?", " ",
                ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""
            ],
        )

    def _create_semantic_chunker(self):
        """
        Creates an instance of SemanticChunker.

        Returns:
            SemanticChunker: A configured semantic chunker instance.
        """
        breakpoint_threshold_amount = float(os.getenv('breakpoint_threshold_amount', 'None'))
        number_of_chunks = int(os.getenv('number_of_chunks', 'None'))
        return SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=os.getenv('breakpoint_threshold_type'),
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks
        )

    def _initialize_text_splitter(self):
        """Initialize the text splitter based on the environment settings."""
        if self.splitter_type == 'RecursiveCharacterTextSplitter':
            self.text_splitter = self._create_recursive_text_splitter()
        elif self.splitter_type == 'SemanticChunker':
            self.text_splitter = self._create_semantic_chunker()

    def _split_and_store_documents(self, docs):
        """
        Splits documents into chunks and stores them as a pickle file.

        Args:
            docs (list): A list of loaded Document objects.
        """
        self._initialize_text_splitter()
        self.chunked_documents = [
            Document(page_content=doc.page_content,
                     metadata={**doc.metadata, 'id': hashlib.md5(doc.page_content.encode()).hexdigest()})
            for doc in self.text_splitter.split_documents(docs)
        ]

        # Store the chunked documents
        with open(self.document_chunks_pickle, 'wb') as f:
            pickle.dump(self.chunked_documents, f)

    def _initialize_milvus(self):
        """Initializes the Milvus vector store."""
        self.db = Milvus.from_documents(
            [], self.embeddings,
            collection_name="default_collection",
            connection_string=self.vector_store_sparse_uri
        )

    def _initialize_postgres(self):
        """Initializes the Postgres vector store."""
        self.db = PGVector.from_documents(
            [], self.embeddings,
            collection_name="default_collection",
            connection_string=self.vector_store_uri
        )

    def _initialize_vector_store(self):
        """Initializes the vector store based on the specified type (Milvus or Postgres)."""
        if self.vector_store == "milvus":
            self._initialize_milvus()
        elif self.vector_store == "postgres":
            self._initialize_postgres()
        else:
            raise ValueError(
                "Only 'milvus' or 'postgres' are supported as vector stores! Please set vector_store in your "
                "environment variables.")

    def _initialize_retrievers(self):
        """Initializes the sparse retriever, ensemble retriever, and rerank retriever."""
        self.sparse_retriever = BM25Retriever(self.db)
        self.rerank_retriever = HuggingFaceCrossEncoder(model_name=self.rerank_model)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.sparse_retriever,
                        ContextualCompressionRetriever(self.sparse_retriever, self.rerank_retriever)]
        )

    def _setup_retrievers(self):
        """Sets up the retrievers based on specified configurations."""
        if self.vector_store_initial_load:
            self._initialize_retrievers()

    def _process_documents(self, docs):
        """Attach skills and personality metadata to the documents."""
        new_docs = []
        for doc in docs:
            doc.metadata['skills'] = self.parse_cv(doc)
            doc.metadata['personality'] = self.personality_predictor.predict(doc)
            new_docs.append(doc)
        return new_docs

    def _update_chunked_documents(self, new_chunks):
        """Update the chunked documents list and store them."""
        if not self.chunked_documents:
            if os.path.exists(self.document_chunks_pickle):
                self.logger.info("documents chunk pickle exists, loading it.")
                self._load_chunked_documents()
        self.chunked_documents += new_chunks
        with open(f"{os.getenv('vector_store_uri')}_sparse.pickle", 'wb') as f:
            pickle.dump(self.chunked_documents, f)

    def _add_to_vector_database(self, new_chunks):
        """Add the new document chunks to the vector database."""
        if not self.db:
            self._initialize_vector_store()
        self.db.add_documents(new_chunks)

        if os.getenv("vector_store") == "postgres":
            self.sparse_retriever.add_documents(new_chunks)
        else:
            self.sparse_retriever = BM25Retriever.from_texts(
                [x.page_content for x in self.chunked_documents],
                metadatas=[x.metadata for x in self.chunked_documents]
            )
            retriever = self.db.as_retriever(
                search_type="mmr",
                search_kwargs={'k': int(os.getenv('vector_store_k'))}
            )
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.sparse_retriever, retriever],
                weights=[0.5, 0.5]
            )

    def _initialize_reranker(self):
        """Initialize the reranking model based on environment settings."""
        if os.getenv("rerank_model") == "flashrank":
            self.compressor = FlashrankRerank(top_n=int(os.getenv("rerank_k")))
        else:
            self.compressor = ScoredCrossEncoderReranker(
                model=HuggingFaceCrossEncoder(model_name=os.getenv("rerank_model")),
                top_n=int(os.getenv("rerank_k"))
            )

    def _parse_cv(self, doc):
        """Extract skills from the CV document."""
        # Implement your skill extraction logic here
        return []

    def load_data(self):
        """
        Loads data from various file types and chunks it into an ensemble retriever.
        """
        if os.path.exists(self.document_chunks_pickle):
            self.logger.info("documents chunk pickle exists, reusing it.")
            self._load_chunked_documents()
        else:
            self.logger.info("loading the documents for the first time.")
            docs = self._load_documents()
            self.logger.info("chunking the documents.")
            self._split_and_store_documents(docs)

        self._initialize_vector_store()
        self._setup_retrievers()

    def add_document(self, filename):
        """
        Load documents from various file types, extract metadata,
        split the documents into chunks, and store them in a vector database.

        Parameters:
            filename (str): The name of the file to be loaded.

        Raises:
            ValueError: If the file type is unsupported.
        """
        docs = self._load_document(filename)
        new_docs = self._process_documents(docs)

        self._initialize_text_splitter()
        new_chunks = self.text_splitter.split_documents(new_docs)
        self._update_chunked_documents(new_chunks)

        # Add new chunks to the vector database
        self._add_to_vector_database(new_chunks)

        # Handle reranking if required
        if os.getenv("rerank") == "True":
            self._initialize_reranker()
            self.rerank_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=self.ensemble_retriever
            )
