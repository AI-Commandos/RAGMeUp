import hashlib
import os
import pickle

from langchain.retrievers import (ContextualCompressionRetriever,
                                  EnsembleRetriever)
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import (CSVLoader, DirectoryLoader,
                                                  Docx2txtLoader, JSONLoader,
                                                  PyPDFDirectoryLoader,
                                                  PyPDFLoader, TextLoader,
                                                  UnstructuredExcelLoader,
                                                  UnstructuredPowerPointLoader)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents.base import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_milvus.vectorstores import Milvus
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lxml import etree
from PostgresBM25Retriever import PostgresBM25Retriever
from ScoredCrossEncoderReranker import ScoredCrossEncoderReranker
from tqdm import tqdm
from text2_sql import TextToSQL
from document_sql_combiner import DocumentSQLCombiner
from transformers import pipeline




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
        self._batch_size = 1000
        # Load environment variables
        self.vector_store_sparse_uri = os.getenv('vector_store_sparse_uri')
        self.vector_store_uri = os.getenv('vector_store_uri')
        self.document_chunks_pickle = os.getenv('document_chunks_pickle')
        self.data_dir = os.getenv("data_directory", "/content/RAGMeUp/server/data")
        self.logger.info(f"Initialized data directory: {self.data_dir}")
        self.file_types = os.getenv("file_types").split(",")
        self.splitter_type = os.getenv('splitter')
        self.vector_store = os.getenv("vector_store")
        self.vector_store_initial_load = os.getenv("vector_store_initial_load") == "True"
        self.rerank = os.getenv("rerank") == "True"
        self.rerank_model = os.getenv("rerank_model")
        self.rerank_k = int(os.getenv("rerank_k"))
        self.vector_store_k = int(os.getenv("vector_store_k"))
        self.chunk_size = int(os.getenv("chunk_size"))
        self.chunk_overlap = int(os.getenv("chunk_overlap"))
        self.breakpoint_threshold_amount = os.getenv('breakpoint_threshold_amount', 'None')
        self.number_of_chunks = None if (value := os.getenv('number_of_chunks',
                                                            None)) is None or value.lower() == 'none' else int(value)
        self.breakpoint_threshold_type = os.getenv('breakpoint_threshold_type')
        self.vector_store_collection = os.getenv("vector_store_collection")
        self.xml_xpath = os.getenv("xml_xpath")
        self.json_text_content = os.getenv("json_text _content", "false").lower() == 'true'
        self.json_schema = os.getenv("json_schema")
        self.llm = pipeline("text-generation", model="gpt2")
        if not self.vector_store_sparse_uri:
            self.logger.error("Environment variable 'vector_store_sparse_uri' is not set.")
            raise ValueError("Missing Postgres connection URI in 'vector_store_sparse_uri'")

        # Initialize Text-to-SQL
        text_to_sql_model = os.getenv("text_to_sql_model", "suriya7/t5-base-text-to-sql")
        self.logger.info(f"Initializing TextToSQL with model: {text_to_sql_model}")
        self.text_to_sql = TextToSQL(model_name=text_to_sql_model, db_uri= self.vector_store_sparse_uri)

        # Initialize DocumentSQLCombiner (optional, only if needed)
        self.document_sql_combiner = DocumentSQLCombiner()

        self.logger.info("RAGHelper initialized successfully with Text-to-SQL support.")

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
            self.logger.info("Loading chunked documents.")
            self.chunked_documents = pickle.load(f)

    def _load_json_files(self):
        """
        Loads JSON files from the data directory.

        Returns:
            list: A list of loaded Document objects from JSON files.
        """
        text_content = self.json_text_content
        loader_kwargs = {
            'jq_schema': self.json_schema,
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
                elements = xmltree.xpath(self.xml_xpath)
                elements = [etree.tostring(element, pretty_print=True).decode() for element in elements]
                metadata = doc.metadata
                metadata['index'] = index
                newdocs += [Document(page_content=content, metadata=metadata) for content in elements]
            except Exception as e:
                self.logger.error(f"Error processing XML document: {e}")
        return newdocs

    @staticmethod
    def _filter_metadata(docs, filters=None):
        """
        Filters the metadata of documents by retaining only specified keys.

        Parameters
        ----------
        docs : list
            A list of document objects, where each document contains a metadata dictionary.
        filters : list, optional
            A list of metadata keys to retain (default is ["source"]).

        Returns
        -------
        list
            The modified list of documents with filtered metadata.

        Raises
        ------
        ValueError
            If docs is not a list or if filters is not a list.
        """
        if not isinstance(docs, list):
            raise ValueError("Expected 'docs' to be a list.")
        if filters is None:
            filters = ["source"]
        elif not isinstance(filters, list):
            raise ValueError("Expected 'filters' to be a list.")

        # Filter metadata for each document
        for doc in docs:
            doc.metadata = {key: doc.metadata.get(key) for key in filters if key in doc.metadata}

        return docs

    def _load_documents(self):
        """
        Loads documents from specified file types in the data directory.

        Returns:
            list: A list of loaded Document objects.
        """
        print(f"Data directory: {self.data_dir}")
        print(f"Files in data directory: {os.listdir(self.data_dir)}")
        self.logger.info(f"Starting to load documents from data directory: {self.data_dir}")
        if not os.path.exists(self.data_dir):
            self.logger.error(f"Data directory does not exist: {self.data_dir}")
            return []
        files_in_directory = os.listdir(self.data_dir)
        self.logger.info(f"Files in data directory: {files_in_directory}")
        docs = []
        for file_type in self.file_types:
            try:
                self.logger.info(f"Loading {file_type} document(s)....")
                if file_type == "pdf":
                    loader = PyPDFDirectoryLoader(self.data_dir)
                    docs += loader.load()
                elif file_type == "json":
                    docs += self._load_json_files()
                elif file_type == "txt":
                    loader = DirectoryLoader(
                        path=self.data_dir,
                        glob="*.txt",
                        loader_cls=TextLoader,
                        recursive=True,
                        show_progress=True,
                    )
                    docs += loader.load()
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
                self.logger.error(f"Error loading {file_type} files: {e}")
                print(f"Error loading {file_type} files: {e}")

        self.logger.info(f"Number of documents loaded: {len(docs)}")
        return self._filter_metadata(docs)

    def _load_json_document(self, filename):
        """Load JSON documents with specific parameters"""
        return JSONLoader(
            file_path=filename,
            jq_schema=self.json_schema,
            text_content=self.json_text_content
        )

    def _load_document(self, filename):
        """Load documents from the specified file based on its extension."""
        file_type = filename.lower().split('.')[-1]
        loaders = {
            'pdf': PyPDFLoader,
            'json': self._load_json_document,
            'txt': TextLoader,
            'csv': CSVLoader,
            'docx': Docx2txtLoader,
            'xlsx': UnstructuredExcelLoader,
            'pptx': UnstructuredPowerPointLoader
        }
        self.logger.info(f"Loading {file_type} document....")
        if file_type in loaders:
            docs = loaders[file_type](filename).load()
            return self._filter_metadata(docs)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _create_recursive_text_splitter(self):
        """
        Creates an instance of RecursiveCharacterTextSplitter.

        Returns:
            RecursiveCharacterTextSplitter: A configured text splitter instance.
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
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
        return SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            number_of_chunks=self.number_of_chunks
        )

    def _initialize_text_splitter(self):
        """Initialize the text splitter based on the environment settings."""
        self.logger.info(f"Initializing {self.splitter_type} splitter.")
        if self.splitter_type == 'RecursiveCharacterTextSplitter':
            self.text_splitter = self._create_recursive_text_splitter()
        elif self.splitter_type == 'SemanticChunker':
            self.text_splitter = self._create_semantic_chunker()

    def _split_documents(self, docs):
        """
        Splits documents into chunks.

        Args:
            docs (list): A list of loaded Document objects.
        """
        self._initialize_text_splitter()
        self.logger.info("Chunking document(s).")
        chunked_documents = [
            Document(page_content=doc.page_content,
                     metadata={**doc.metadata, 'id': hashlib.md5(doc.page_content.encode()).hexdigest()})
            for doc in self.text_splitter.split_documents(docs)
        ]
        self.logger.info(f"Number of chunks created: {len(chunked_documents)}")
        return chunked_documents

    def _split_and_store_documents(self, docs):
        """
        Splits documents into chunks and stores them as a pickle file.

        Args:
            docs (list): A list of loaded Document objects.
        """
        self.chunked_documents = self._split_documents(docs)
        # Store the chunked documents
        self.logger.info("Storing chunked document(s).")
        with open(self.document_chunks_pickle, 'wb') as f:
            pickle.dump(self.chunked_documents, f)

    def _initialize_milvus(self):
        """Initializes the Milvus vector store."""
        self.logger.info("Setting up Milvus Vector DB.")
        self.db = Milvus.from_documents(
            [], self.embeddings,
            drop_old=not self.vector_store_initial_load,
            connection_args={"uri": self.vector_store_uri},
            collection_name=self.vector_store_collection,
        )

    def _initialize_postgres(self):
        """Initializes the Postgres vector store."""
        self.logger.info(f"Setting up PGVector DB.")
        self.db = PGVector(
            embeddings=self.embeddings,
            collection_name=self.vector_store_collection,
            connection=self.vector_store_uri,
            use_jsonb=True
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
        if self.vector_store_initial_load:
            self.logger.info("Loading data from existing store.")
            # Add the documents 1 by 1, so we can track progress
            with tqdm(total=len(self.chunked_documents), desc="Vectorizing documents") as pbar:
                for i in range(0, len(self.chunked_documents), self._batch_size):
                    # Slice the documents for the current batch
                    batch = self.chunked_documents[i:i + self._batch_size]
                    # Prepare documents and their IDs for batch insertion
                    documents = [d for d in batch]
                    ids = [d.metadata["id"] for d in batch]

                    # Add the batch of documents to the database
                    self.db.add_documents(documents, ids=ids)

                    # Update the progress bar by the size of the batch
                    pbar.update(len(batch))

    def _initialize_bm25retriever(self):
        """Initializes in memory BM25Retriever."""
        self.logger.info(f"Number of documents loaded for BM25: {len(self.chunked_documents)}")
        if not self.chunked_documents:
            self.logger.error("No documents available to initialize BM25Retriever. Check document loading.")
            for file_type in self.file_types:
                self.logger.warning(f"Ensure {file_type} files are present in the data directory: {self.data_dir}")
            raise ValueError("No documents available to initialize BM25Retriever. Check document loading.")
        self.logger.info("Initializing BM25Retriever.")
        try:
            self.logger.info("Initializing BM25Retriever.")
            self.sparse_retriever = BM25Retriever.from_texts(
                [x.page_content for x in self.chunked_documents],
                metadatas=[x.metadata for x in self.chunked_documents]
            )
            self.logger.info("BM25Retriever successfully initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize BM25Retriever: {e}")
            raise

    def _initialize_postgresbm25retriever(self):
        """Initializes in memory PostgresBM25Retriever."""
        self.logger.info("Initializing PostgresBM25Retriever.")
        self.sparse_retriever = PostgresBM25Retriever(connection_uri=self.vector_store_sparse_uri,
                                                      table_name="sparse_vectors", k=self.vector_store_k)
        if self.vector_store_initial_load == "True":
            self.logger.info("Loading data from existing store into the PostgresBM25Retriever.")
            with tqdm(total=len(self.chunked_documents), desc="Vectorizing documents") as pbar:
                for d in self.chunked_documents:
                    self.sparse_retriever.add_documents([d], ids=[d.metadata["id"]])
                    pbar.update(1)
    def retrieve_from_sql(self, user_query):
        sql_query = self.text_to_sql.translate(user_query)
        self.logger.info(f"Received user query for SQL retrieval: {user_query}")
        try:
            sql_results = self.text_to_sql.execute(sql_query)
            self.logger.info(f"Generated SQL Query: {sql_query}")  # 打印生成的 SQL 查询
        except Exception as e:
            self.logger.error(f"Error executing SQL: {e}")
            sql_results = []
        self.logger.info(f"SQL Query Results: {sql_results}")
        return [{"type": "sql_result", "content": result} for result in sql_results]

    def _initialize_retrievers(self):
        """Initializes the sparse retriever, ensemble retriever, and rerank retriever."""
        if self.vector_store == "milvus":
            self._initialize_bm25retriever()
        elif self.vector_store == "postgres":
            self._initialize_postgresbm25retriever()
        else:
            raise ValueError(
                "Only 'milvus' or 'postgres' are supported as vector stores! Please set vector_store in your "
                "environment variables.")

    def _initialize_reranker(self):
        """Initialize the reranking model based on environment settings."""
        if self.rerank_model == "flashrank":
            self.logger.info("Setting up the FlashrankRerank.")
            self.compressor = FlashrankRerank(top_n=self.rerank_k)
        else:
            self.logger.info("Setting up the ScoredCrossEncoderReranker.")
            self.compressor = ScoredCrossEncoderReranker(
                model=HuggingFaceCrossEncoder(model_name=self.rerank_model),
                top_n=self.rerank_k
            )
        self.logger.info("Setting up the ContextualCompressionRetriever.")
        self.rerank_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.ensemble_retriever
        )

    def _setup_retrievers(self):
        """Sets up the retrievers based on specified configurations."""
        self._initialize_retrievers()
        # Set up the vector retriever
        self.logger.info("Setting up the Vector Retriever.")
        retriever = self.db.as_retriever(
            search_type="mmr", search_kwargs={'k': self.vector_store_k}
        )
        self.logger.info("Setting up the hybrid retriever.")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.sparse_retriever, retriever], weights=[0.5, 0.5]
        )
        if self.rerank:
            self._initialize_reranker()

    def _update_chunked_documents(self, new_chunks):
        """Update the chunked documents list and store them."""
        if self.vector_store == 'milvus':
            if not self.chunked_documents:
                if os.path.exists(self.document_chunks_pickle):
                    self.logger.info("documents chunk pickle exists, loading it.")
                    self._load_chunked_documents()
            self.chunked_documents += new_chunks
            with open(f"{self.vector_store_uri}_sparse.pickle", 'wb') as f:
                pickle.dump(self.chunked_documents, f)

    def _add_to_vector_database(self, new_chunks):
        """Add the new document chunks to the vector database."""
        if not self.db:
            self._initialize_vector_store()

        documents = [d for d in new_chunks]
        ids = [d.metadata["id"] for d in new_chunks]
        self.db.add_documents(documents, ids=ids)

        if self.vector_store == "postgres":
            self.sparse_retriever.add_documents(new_chunks, ids)
        else:
            # Recreate the in-memory store
            self._initialize_bm25retriever()
            # Update full retriever too
        retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={'k': self.vector_store_k}
        )
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.sparse_retriever, retriever],
            weights=[0.5, 0.5]
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
        new_docs = self._load_document(filename)

        self.logger.info("chunking the documents.")
        new_chunks = self._split_documents(new_docs)

        self._update_chunked_documents(new_chunks)

        # Add new chunks to the vector database
        self._add_to_vector_database(new_chunks)
    def inference_pipeline(self, user_query, history=None, fetch_new_documents=True):
        """
        整合稠密检索、稀疏检索和 Text-to-SQL，生成最终答案。

        Args:
            user_query (str): 用户的自然语言查询。
            history (list, optional): 上下文历史记录。
            fetch_new_documents (bool, optional): 是否需要重新检索新文档。

        Returns:
            dict: 包括生成的答案、检索到的文档和历史记录。
        """
        self.logger.info(f"Starting inference pipeline with user query: {user_query}")
        self.logger.info(f"History: {history}, Fetch New Documents: {fetch_new_documents}")
        retrieved_docs = {"dense_results": [], "sparse_results": [], "sql_results": []}


        # 如果没有历史记录或需要重新检索
        if not history or fetch_new_documents:
            self.logger.info("Starting new retrieval process.")

            # 1. 稠密向量检索 (Milvus/pgvector)
            if self.db:
                dense_results = self.db.search(user_query)
                self.logger.info(f"Dense retrieval results: {len(dense_results)} documents retrieved.")
                retrieved_docs["dense_results"] = dense_results

            # 2. 稀疏向量检索 (BM25)
            if self.sparse_retriever:
                sparse_results = self.sparse_retriever.retrieve(user_query)
                self.logger.info(f"Sparse retrieval results: {len(sparse_results)} documents retrieved.")
                retrieved_docs["sparse_results"] = sparse_results

            # 3. Text-to-SQL 检索
            self.logger.info("Forcing SQL retrieval for testing.")
            sql_results = self.retrieve_from_sql(user_query)
            self.logger.info(f"SQL retrieval results: {len(sql_results)} results retrieved.")
            retrieved_docs["sql_results"] = sql_results

        else:
            # 如果有历史记录，直接使用
            self.logger.info("Using history for retrieval context.")
            retrieved_docs = history
        
        formatted_sql_results = [
            {"type": "sql_result", "content": f"Result: {result['content']}"}
            for result in retrieved_docs["sql_results"]
            if "content" in result
        ]
        all_retrieved_docs = (
            retrieved_docs["dense_results"]
            + retrieved_docs["sparse_results"]
            + retrieved_docs["sql_results"]
        )

        if hasattr(self, "llm") and self.llm:
            self.logger.info("Generating final answer using Hugging Face LLM.")
            context = "\n".join(
                [doc["content"] for doc in all_retrieved_docs if "content" in doc]
            )
            prompt = f"{context}\n\nQuestion: {user_query}\nAnswer:"
            try:
                llm_response = self.llm(prompt, max_length=200, do_sample=True, top_p=0.95)
                answer = llm_response[0]["generated_text"] if llm_response else "No answer generated."
            except Exception as e:
                self.logger.error(f"Error generating answer: {e}")
                answer = "Error occurred while generating the answer."
        else:
            self.logger.error("LLM is not initialized. Returning only retrieved documents.")
            answer = "LLM not available. Please check your configuration."
        
        self.logger.info(f"SQL Results: {retrieved_docs['sql_results']}")
        
        return {
            "answer": answer,
            "documents": retrieved_docs,  # 结构化展示
            "all_results": all_retrieved_docs,  # 合并后的展示
        }