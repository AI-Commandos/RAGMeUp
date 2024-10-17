import os
import re

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from provenance import (DocumentSimilarityAttribution,
                        compute_llm_provenance_cloud,
                        compute_rerank_provenance)
from RAGHelper import RAGHelper


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
        self.rewrite_chain = None
        self.rewrite_ask_chain = None
        self.attributor = None
        self.rag_fetch_new_chain = None
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
        if os.getenv("use_gemini") == "True":
            self.logger.info("Initializing Gemini conversation.")
            return ChatGoogleGenerativeAI(model=os.getenv("gemini_model_name"), convert_system_message_to_human=True)
        if os.getenv("use_azure") == "True":
            self.logger.info("Initializing Azure OpenAI conversation.")
            return AzureChatOpenAI(openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                                   azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"])
        if os.getenv("use_ollama") == "True":
            self.logger.info("Initializing Ollama conversation.")
            return OllamaLLM(model=os.getenv("ollama_model"))

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
        context_retriever = self.rerank_retriever if self.rerank else self.ensemble_retriever
        return {"context": context_retriever | RAGHelper.format_documents,
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
            self.logger.info(f"The response of the rewrite loop is - {response}")
            response = self.extract_response_content(response)

            if re.sub(r'\W+ ', '', response).lower().startswith('yes'):
                return self.extract_response_content(self.rewrite_chain.invoke(user_query))
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

        thread = self.create_interaction_thread(history, fetch_new_documents)
        # Create prompt from prompt template
        prompt = ChatPromptTemplate.from_messages(thread)

        # Create llm chain
        llm_chain = prompt | self.llm

        if fetch_new_documents:
            context_retriever = self.ensemble_retriever if self.rerank else self.rerank_retriever
            retriever_chain = {
                "docs": context_retriever,
                "context": context_retriever | RAGHelper.format_documents,
                "question": RunnablePassthrough()
            }
            llm_chain = prompt | self.llm | StrOutputParser()
            rag_chain = (
                retriever_chain
                | RunnablePassthrough.assign(
                    answer=lambda x: llm_chain.invoke(
                        {"docs": x["docs"], "context": x["context"], "question": x["question"]}
                    ))
                | combine_results
            )
        else:
            retriever_chain = {"question": RunnablePassthrough()}
            llm_chain = prompt | self.llm | StrOutputParser()
            rag_chain = (
                retriever_chain
                | RunnablePassthrough.assign(
                    answer=lambda x: llm_chain.invoke(
                        {"question": x["question"]}
                    ))
                | combine_results
            )

        user_query = self.handle_rewrite(user_query)
        # Check if we need to apply Re2 to mention the question twice
        if os.getenv("use_re2") == "True":
            user_query = f'{user_query}\n{os.getenv("re2_prompt")}{user_query}'

        # Invoke RAG pipeline
        reply = rag_chain.invoke(user_query)

        # Track provenance if needed
        if fetch_new_documents and os.getenv("provenance_method") in ['rerank', 'attention', 'similarity', 'llm']:
            self.track_provenance(reply, user_query)

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

    @staticmethod
    def create_interaction_thread(history: list, fetch_new_documents: bool) -> list:
        """Create the conversation thread based on user input and history.

        Args:
            user_query (str): The user's query.
            history (list): The history of previous interactions.
            fetch_new_documents (bool): Whether to fetch new documents.

        Returns:
            list: The constructed conversation thread.
        """
        # Create prompt template based on whether we have history or not
        thread = [(x["role"], x["content"].replace("{", "(").replace("}", ")")) for x in history]
        if fetch_new_documents:
            thread = [('system', os.getenv('rag_instruction')), ('human', os.getenv('rag_question_initial'))]
        else:
            thread.append(('human', os.getenv('rag_question_followup')))
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

    def track_provenance(self, reply: str, user_query: str) -> None:
        """Track the provenance of the response if applicable.

        Args:
            reply (str): The response from the LLM.
            user_query (str): The original user query.
        """
        # Add the user question and the answer to our thread for provenance computation
        # Retrieve answer and context
        answer = reply.get('answer')
        context = reply.get('docs')

        provenance_method = os.getenv("provenance_method")
        self.logger.info(f"Provenance method: {provenance_method}")

        # Use the reranker if the provenance method is 'rerank'
        if provenance_method == "rerank":
            self.logger.info("Using reranking for provenance attribution.")
            if not self.rerank:
                raise ValueError("Provenance attribution is set to rerank but reranking is not enabled. "
                                 "Please choose another method or enable reranking.")

            reranked_docs = compute_rerank_provenance(self.compressor, user_query, context, answer)
            self.logger.debug(f"Reranked documents computed: {len(reranked_docs)} docs reranked.")

            # Build provenance scores based on reranked docs
            provenance_scores = []
            for doc in context:
                reranked_score = next(
                    (d.metadata['relevance_score'] for d in reranked_docs if d.page_content == doc.page_content), None)
                if reranked_score is None:
                    self.logger.warning(f"Document not found in reranked docs: {doc.page_content}")
                provenance_scores.append(reranked_score)
            self.logger.debug("Provenance scores computed using reranked documents.")

        # Use similarity-based provenance if method is 'similarity'
        elif provenance_method == "similarity":
            self.logger.info("Using similarity-based provenance attribution.")
            provenance_scores = self.attributor.compute_similarity(user_query, context, answer)
            self.logger.debug("Provenance scores computed using similarity method.")

        # Use LLM-based provenance if method is 'llm'
        elif provenance_method == "llm":
            self.logger.info("Using LLM-based provenance attribution.")
            provenance_scores = compute_llm_provenance_cloud(self.llm, user_query, context, answer)
            self.logger.debug("Provenance scores computed using LLM-based method.")

        # Add provenance scores to documents
        for i, score in enumerate(provenance_scores):
            reply['docs'][i].metadata['provenance'] = score
            self.logger.debug(f"Provenance score added to doc {i}: {score}")

    @staticmethod
    def extract_response_content(response: dict) -> str:
        """Extract the content from the response dictionary.

        Args:
            response (dict): The response dictionary.

        Returns:
            str: The extracted content.
        """
        # return getattr(response, 'content', getattr(response, 'answer', response['answer']))
        if hasattr(response, 'content'):
            response = response.content
        elif hasattr(response, 'answer'):
            response = response.answer
        elif 'answer' in response:
            response = response["answer"]
        return response
