import os
import re
import pickle
from provenance import (compute_llm_provenance_cloud, compute_rerank_provenance, DocumentSimilarityAttribution)
from ScoredCrossEncoderReranker import ScoredCrossEncoderReranker
from RAGHelper_v2 import RAGHelper
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.document_loaders import (PyPDFLoader, JSONLoader, Docx2txtLoader,
                                                  UnstructuredExcelLoader, UnstructuredPowerPointLoader)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


def combine_results(inputs: dict) -> dict:
    """Combine the results of the user query processing.

    Args:
        inputs (dict): The input results.

    Returns:
        dict: A dictionary containing the answer, context, and question.
    """
    combined = {"answer": inputs["answer"], "question": inputs["question"]}
    if "context" in inputs and "docs" in inputs:
        combined.update({"docs": inputs["docs"], "context": inputs["context"]})
    return combined


class RAGHelperCloud(RAGHelper):
    def __init__(self, logger):
        """Initialize the RAGHelperCloud instance with required models and configurations."""
        super().__init__(logger)
        self.logger = logger
        self.llm = self.initialize_llm()
        self.embeddings = self.initialize_embeddings()

        # Load the data
        self.load_data()
        self.initialize_rag_chains()
        self.initialize_provenance_attribution()
        self.initialize_rewrite_loops()

    def initialize_llm(self):
        """Initialize the Language Model based on environment configurations."""
        if os.getenv("use_openai") == "True":
            self.logger.info("Initializing OpenAI conversation.")
            return ChatOpenAI(model=os.getenv("openai_model_name"), temperature=0, max_tokens=None, timeout=None,
                              max_retries=2)
        elif os.getenv("use_gemini") == "True":
            self.logger.info("Initializing Gemini conversation.")
            return ChatGoogleGenerativeAI(model=os.getenv("gemini_model_name"), convert_system_message_to_human=True)
        elif os.getenv("use_azure") == "True":
            self.logger.info("Initializing Azure OpenAI conversation.")
            return AzureChatOpenAI(openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                                   azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"])
        elif os.getenv("use_ollama") == "True":
            self.logger.info("Initializing Ollama conversation.")
            return OllamaLLM(model=os.getenv("ollama_model"))
        else:
            self.logger.error("No valid LLM configuration found.")
            raise ValueError("No valid LLM configuration found.")

    def initialize_embeddings(self):
        """Initialize the embeddings based on the CPU/GPU configuration."""
        embedding_model = os.getenv('embedding_model')
        model_kwargs = {'device': 'cpu'} if os.getenv('force_cpu') == "True" else {'device': 'cuda'}
        self.logger.info(f"Initializing embedding model {embedding_model} with params {model_kwargs}.")
        return HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs=model_kwargs)

    def initialize_rag_chains(self):
        """Create the RAG chain for fetching new documents."""
        rag_thread = [
            ('system', os.getenv('rag_fetch_new_instruction')),
            ('human', os.getenv('rag_fetch_new_question'))
        ]
        self.logger.info("Initializing RAG chains for fetching new documents.")
        rag_prompt = ChatPromptTemplate.from_messages(rag_thread)
        rag_llm_chain = rag_prompt | self.llm
        self.rag_fetch_new_chain = {"question": RunnablePassthrough()} | rag_llm_chain

    def initialize_provenance_attribution(self):
        """Initialize the provenance attribution method based on the environment configuration."""
        if os.getenv("provenance_method") == "similarity":
            self.attributor = DocumentSimilarityAttribution()

    def initialize_rewrite_loops(self):
        """Create rewrite loop LLM chains if enabled."""
        if os.getenv("use_rewrite_loop") == "True":
            self.rewrite_ask_chain = self.create_rewrite_ask_chain()
            self.rewrite_chain = self.create_rewrite_chain()

    def create_rewrite_ask_chain(self):
        """Create the chain to ask if a rewrite is needed."""
        rewrite_ask_thread = [
            ('system', os.getenv('rewrite_query_instruction')),
            ('human', os.getenv('rewrite_query_question'))
        ]
        rewrite_ask_prompt = ChatPromptTemplate.from_messages(rewrite_ask_thread)
        rewrite_ask_llm_chain = rewrite_ask_prompt | self.llm
        context_retriever = self.ensemble_retriever if os.getenv("rerank") != "True" else self.rerank_retriever
        return {"context": context_retriever | RAGHelper.format_documents(),
                "question": RunnablePassthrough()} | rewrite_ask_llm_chain

    def create_rewrite_chain(self):
        """Create the chain to perform the actual rewrite."""
        rewrite_thread = [('human', os.getenv('rewrite_query_prompt'))]
        rewrite_prompt = ChatPromptTemplate.from_messages(rewrite_thread)
        rewrite_llm_chain = rewrite_prompt | self.llm
        return {"question": RunnablePassthrough()} | rewrite_llm_chain

    def handle_rewrite(self, user_query: str) -> str:
        """Determine if a rewrite is needed and perform it if required.

        Args:
            user_query (str): The original user query.

        Returns:
            str: The potentially rewritten user query.
        """
        if os.getenv("use_rewrite_loop") == "True":
            response = self.rewrite_ask_chain.invoke(user_query)
            response = self.extract_response_content(response)

            if re.sub(r'\W+ ', '', response).lower().startswith('yes'):
                return self.extract_response_content(self.rewrite_chain.invoke(user_query))
            else:
                return user_query
        return user_query

    def handle_user_interaction(self, user_query: str, history: list) -> tuple:
        """Handle user interaction by processing their query and maintaining conversation history.

        Args:
            user_query (str): The user's query.
            history (list): The history of previous interactions.

        Returns:
            tuple: A tuple containing the conversation thread and the reply.
        """
        fetch_new_documents = self.should_fetch_new_documents(user_query, history)

        thread = self.create_interaction_thread(user_query, history, fetch_new_documents)
        prompt = ChatPromptTemplate.from_messages(thread)
        llm_chain = prompt | self.llm

        if fetch_new_documents:
            user_query = self.handle_rewrite(user_query)
            context_retriever = self.ensemble_retriever if os.getenv("rerank") != "True" else self.rerank_retriever
            retriever_chain = {
                "docs": context_retriever,
                "context": context_retriever | RAGHelper.format_documents(),
                "question": RunnablePassthrough()
            }
            llm_chain = prompt | self.llm | StrOutputParser()
            rag_chain = self.create_rag_chain(retriever_chain, llm_chain)
        else:
            retriever_chain = {"question": RunnablePassthrough()}
            llm_chain = prompt | self.llm | StrOutputParser()
            rag_chain = self.create_rag_chain(retriever_chain, llm_chain)

        # Check if we need to apply Re2 to mention the question twice
        if os.getenv("use_re2") == "True":
            user_query = f'{user_query}\n{os.getenv("re2_prompt")}{user_query}'

        # Invoke RAG pipeline
        reply = rag_chain.invoke(user_query)

        # Track provenance if needed
        self.track_provenance(reply, fetch_new_documents, user_query)

        return (thread, reply)

    def should_fetch_new_documents(self, user_query: str, history: list) -> bool:
        """Determine if new documents should be fetched based on user query and history.

        Args:
            user_query (str): The user's query.
            history (list): The history of previous interactions.

        Returns:
            bool: True if new documents should be fetched, False otherwise.
        """
        if not history:
            self.logger.info("There is no content in history, fetching new documents!")
            return True
        response = self.rag_fetch_new_chain.invoke(user_query)
        response = self.extract_response_content(response)
        return re.sub(r'\W+ ', '', response).lower().startswith('yes')

    def create_interaction_thread(self, user_query: str, history: list, fetch_new_documents: bool) -> list:
        """Create the conversation thread based on user input and history.

        Args:
            user_query (str): The user's query.
            history (list): The history of previous interactions.
            fetch_new_documents (bool): Whether to fetch new documents.

        Returns:
            list: The constructed conversation thread.
        """
        thread = []
        if fetch_new_documents:
            thread.append(('system', os.getenv('fetch_documents_instruction')))
            thread.append(('human', user_query))
        else:
            thread.append(('human', user_query))
        return thread

    def create_rag_chain(self, retriever_chain: dict, llm_chain: str) -> str:
        """Create the RAG chain to invoke both the retriever and the language model.

        Args:
            retriever_chain (dict): The retriever chain configuration.
            llm_chain (str): The language model chain.

        Returns:
            str: The combined RAG chain.
        """
        return retriever_chain | llm_chain

    def track_provenance(self, reply: str, fetch_new_documents: bool, user_query: str) -> None:
        """Track the provenance of the response if applicable.

        Args:
            reply (str): The response from the LLM.
            fetch_new_documents (bool): Indicates if new documents were fetched.
            user_query (str): The original user query.
        """
        if os.getenv("provenance_method") == "similarity":
            self.logger.info("Tracking provenance using similarity method.")
            compute_llm_provenance_cloud(reply, user_query)

        if fetch_new_documents:
            self.logger.info("Tracking provenance for new documents.")
            compute_rerank_provenance(reply, user_query)

    def extract_response_content(self, response: dict) -> str:
        """Extract the content from the response dictionary.

        Args:
            response (dict): The response dictionary.

        Returns:
            str: The extracted content.
        """
        if isinstance(response, dict) and "content" in response:
            return response["content"]
        return ""

    def addDocument(self, filename: str) -> None:
        """Add a document to the existing data structure and process it.

        Args:
            filename (str): The path to the document to be added.

        Raises:
            ValueError: If the file type is unsupported or the document loading fails.
        """
        try:
            # Load documents based on file extension
            if filename.lower().endswith('pdf'):
                docs = PyPDFLoader(filename).load()
            elif filename.lower().endswith('json'):
                docs = JSONLoader(
                    file_path=filename,
                    jq_schema=os.getenv("json_schema"),
                    text_content=os.getenv("json_text_content") == "True",
                ).load()
            elif filename.lower().endswith('csv'):
                docs = CSVLoader(filename).load()
            elif filename.lower().endswith('docx'):
                docs = Docx2txtLoader(filename).load()
            elif filename.lower().endswith('xlsx'):
                docs = UnstructuredExcelLoader(filename).load()
            elif filename.lower().endswith('pptx'):
                docs = UnstructuredPowerPointLoader(filename).load()
            else:
                raise ValueError(f"Unsupported file type: {filename}")

            # Process the documents to extract skills and personality
            new_docs = []
            for doc in docs:
                # Get the skills and personality predictions
                try:
                    skills = self.parseCV(doc)
                    doc.metadata['skills'] = skills
                    doc.metadata['personality'] = self.personality_predictor.predict(doc)
                    new_docs.append(doc)
                except Exception as e:
                    self.logger.error(f"Failed to process document {doc}: {str(e)}")

            # Initialize the text splitter based on environment variable
            self.initialize_text_splitter()

            # Split the documents into chunks
            new_chunks = self.text_splitter.split_documents(new_docs)

            # Update chunked documents
            self.chunked_documents.extend(new_chunks)

            # Store the updated chunked documents in a sparse database
            with open(f"{os.getenv('vector_store_uri')}_sparse.pickle", 'wb') as f:
                pickle.dump(self.chunked_documents, f)

            # Add new chunks to the vector database
            self.db.add_documents(new_chunks)

            # Update BM25 retriever
            self.update_bm25_retriever()

            # Update ensemble retriever
            self.update_ensemble_retriever()

            # Configure re-ranking if enabled
            self.configure_reranking()

            self.logger.info(f"Successfully added and processed document: {filename}")

        except Exception as e:
            self.logger.error(f"Error adding document {filename}: {str(e)}")
            raise

    def initialize_text_splitter(self) -> None:
        """Initialize the text splitter based on the environment configuration."""
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
            breakpoint_threshold_amount = None
            number_of_chunks = None
            if os.getenv('breakpoint_threshold_amount') != 'None':
                breakpoint_threshold_amount = float(os.getenv('breakpoint_threshold_amount'))
            if os.getenv('number_of_chunks') != 'None':
                number_of_chunks = int(os.getenv('number_of_chunks'))
            self.text_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type=os.getenv('breakpoint_threshold_type'),
                breakpoint_threshold_amount=breakpoint_threshold_amount,
                number_of_chunks=number_of_chunks
            )

    def update_bm25_retriever(self) -> None:
        """Update the BM25 retriever with the current chunked documents."""
        bm25_retriever = BM25Retriever.from_texts(
            [x.page_content for x in self.chunked_documents],
            metadatas=[x.metadata for x in self.chunked_documents]
        )
        self.bm25_retriever = bm25_retriever

    def update_ensemble_retriever(self) -> None:
        """Update the ensemble retriever with the current database and BM25 retriever."""
        retriever = self.db.as_retriever(search_type="mmr", search_kwargs={'k': int(os.getenv('vector_store_k'))})
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, retriever],
            weights=[0.5, 0.5]
        )

    def configure_reranking(self) -> None:
        """Configure re-ranking for the retrieval process if enabled."""
        if os.getenv("rerank") == "True":
            if os.getenv("rerank_model") == "flashrank":
                self.compressor = FlashrankRerank(top_n=int(os.getenv("rerank_k")))
            else:
                self.compressor = ScoredCrossEncoderReranker(
                    model=HuggingFaceCrossEncoder(model_name=os.getenv("rerank_model")),
                    top_n=int(os.getenv("rerank_k"))
                )

            self.rerank_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=self.ensemble_retriever
            )
