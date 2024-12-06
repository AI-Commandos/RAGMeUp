import os
import re
import torch
from provenance import (
    compute_attention,
    compute_rerank_provenance,
    compute_llm_provenance,
    DocumentSimilarityAttribution
)
from RAGHelper import RAGHelper
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from langchain.chains.llm import LLMChain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough


class RAGHelperLocal(RAGHelper):
    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger
        self.tokenizer, self.model = self._initialize_llm()
        self.llm = self._create_llm_pipeline()
        self.embeddings = self._initialize_embeddings()

        # Load the data
        self.load_data()

        # Create RAG chains
        self.rag_fetch_new_chain = self._create_rag_chain()
        self.rewrite_ask_chain, self.rewrite_chain = self._initialize_rewrite_chains()

        # Initialize provenance method
        self.attributor = DocumentSimilarityAttribution() if os.getenv("provenance_method") == "similarity" else None

    def _initialize_llm(self):
        """Initialize the LLM based on the available hardware and configurations."""
        llm_model = os.getenv('llm_model')
        trust_remote_code = os.getenv('trust_remote_code') == "True"

        if torch.backends.mps.is_available():
            self.logger.info("Initializing LLM on MPS.")
            tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,
                device_map="auto"
            ).to(torch.device("mps"))
        elif os.getenv('force_cpu') == "True":
            self.logger.info("LLM on CPU (slow!).")
            tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                trust_remote_code=trust_remote_code,
            ).to(torch.device("cpu"))
        else:
            self.logger.info("Initializing LLM on GPU.")
            bnb_config = self._get_bnb_config()
            tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=trust_remote_code)
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                quantization_config=bnb_config,
                trust_remote_code=trust_remote_code,
                device_map="auto"
            )

        return tokenizer, model

    @staticmethod
    def _get_bnb_config():
        """Get BitsAndBytes configuration for model quantization."""
        use_4bit = True
        bnb_4bit_compute_dtype = "float16"
        bnb_4bit_quant_type = "nf4"
        use_nested_quant = False

        return BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=use_nested_quant,
        )

    def _create_llm_pipeline(self):
        """Create and return the LLM pipeline for text generation."""
        text_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=float(os.getenv('temperature')),
            repetition_penalty=float(os.getenv('repetition_penalty')),
            return_full_text=True,
            max_new_tokens=int(os.getenv('max_new_tokens')),
            model_kwargs={
                'device_map': 'auto',
            }
        )
        return HuggingFacePipeline(pipeline=text_generation_pipeline)

    @staticmethod
    def _initialize_embeddings():
        """Initialize and return embeddings for vector storage."""
        model_kwargs = {
            'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if os.getenv(
                'force_cpu') != "True" else 'cpu'
        }
        return HuggingFaceEmbeddings(
            model_name=os.getenv('embedding_model'),
            model_kwargs=model_kwargs
        )

    def _create_rag_chain(self):
        """Create and return the RAG chain for fetching new documents."""
        rag_thread = [
            {'role': 'system', 'content': os.getenv('rag_fetch_new_instruction')},
            {'role': 'user', 'content': os.getenv('rag_fetch_new_question')}
        ]
        rag_prompt_template = self.tokenizer.apply_chat_template(rag_thread, tokenize=False)
        rag_prompt = PromptTemplate(
            input_variables=["question"],
            template=rag_prompt_template,
        )
        rag_llm_chain = LLMChain(llm=self.llm, prompt=rag_prompt)
        return {"question": RunnablePassthrough()} | rag_llm_chain

    def _initialize_rewrite_chains(self):
        """Initialize and return rewrite ask and rewrite chains if required."""
        rewrite_ask_chain = None
        rewrite_chain = None

        if os.getenv("use_rewrite_loop") == "True":
            rewrite_ask_chain = self._create_rewrite_ask_chain()
            rewrite_chain = self._create_rewrite_chain()

        return rewrite_ask_chain, rewrite_chain

    def _create_rewrite_ask_chain(self):
        """Create and return the chain to ask if rewriting is needed."""
        rewrite_ask_thread = [
            {'role': 'system', 'content': os.getenv('rewrite_query_instruction')},
            {'role': 'user', 'content': os.getenv('rewrite_query_question')}
        ]
        rewrite_ask_prompt_template = self.tokenizer.apply_chat_template(rewrite_ask_thread, tokenize=False)
        rewrite_ask_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=rewrite_ask_prompt_template,
        )
        rewrite_ask_llm_chain = LLMChain(llm=self.llm, prompt=rewrite_ask_prompt)

        context_retriever = self.rerank_retriever if self.rerank else self.ensemble_retriever
        return {"context": context_retriever | RAGHelper.format_documents,
                "question": RunnablePassthrough()} | rewrite_ask_llm_chain

    def _create_rewrite_chain(self):
        """Create and return the rewrite chain."""
        rewrite_thread = [{'role': 'user', 'content': os.getenv('rewrite_query_prompt')}]
        rewrite_prompt_template = self.tokenizer.apply_chat_template(rewrite_thread, tokenize=False)
        rewrite_prompt = PromptTemplate(
            input_variables=["question"],
            template=rewrite_prompt_template,
        )
        rewrite_llm_chain = LLMChain(llm=self.llm, prompt=rewrite_prompt)

        return {"question": RunnablePassthrough()} | rewrite_llm_chain

    def handle_rewrite(self, user_query: str) -> str:
        """Handle the rewriting of the user query if necessary."""
        if os.getenv("use_rewrite_loop") == "True":
            response = self.rewrite_ask_chain.invoke(user_query)
            end_string = os.getenv("llm_assistant_token")
            reply = response['text'][response['text'].rindex(end_string) + len(end_string):]
            reply = re.sub(r'\W+ ', '', reply)

            if reply.lower().startswith('no'):
                response = self.rewrite_chain.invoke(user_query)
                reply = response['text'][response['text'].rindex(end_string) + len(end_string):]
                return reply
            else:
                return user_query
        else:
            return user_query

    def handle_user_interaction(self, user_query, history):
        """Handle user interaction, fetching documents and managing query rewriting, document provenance, and LLM response.

        Args:
            user_query (str): The user's query.
            history (list): A list of previous conversation history.

        Returns:
            tuple: The conversation thread and the LLM response with potential provenance scores.
        """
        fetch_new_documents = self._should_fetch_new_documents(user_query, history)
        thread = self._prepare_conversation_thread(history, fetch_new_documents)
        input_variables = self._determine_input_variables(fetch_new_documents)
        prompt = self._create_prompt_template(thread, input_variables)

        llm_chain = self._create_llm_chain(fetch_new_documents, prompt)

        # Handle rewrite and re2
        user_query = self.handle_rewrite(user_query)
        if os.getenv("use_re2") == "True":
            user_query = f'{user_query}\n{os.getenv("re2_prompt")}{user_query}'
        
        reply = self._invoke_rag_chain(user_query, llm_chain)

        if fetch_new_documents:
            self._track_provenance(user_query, reply, thread)

        return thread, reply

    def _should_fetch_new_documents(self, user_query, history):
        """Determine whether to fetch new documents based on the user's query and conversation history."""
        if len(history) == 0:
            return True

        response = self.rag_fetch_new_chain.invoke(user_query)
        reply = self._extract_reply(response)
        return reply.lower().startswith('yes')

    @staticmethod
    def _prepare_conversation_thread(history, fetch_new_documents):
        """Prepare the conversation thread with formatted history and new instructions."""
        thread = [{"role": x["role"], "content": x["content"].replace("{", "(").replace("}", ")")} for x in history]
        if fetch_new_documents:
            thread = []
        if len(thread) == 0:
            thread.append({'role': 'system', 'content': os.getenv('rag_instruction')})
            thread.append({'role': 'user', 'content': os.getenv('rag_question_initial')})
        else:
            thread.append({'role': 'user', 'content': os.getenv('rag_question_followup')})
        return thread

    @staticmethod
    def _determine_input_variables(fetch_new_documents):
        """Determine the input variables for the prompt based on whether new documents are fetched."""
        return ["context", "question"] if fetch_new_documents else ["question"]

    def _create_prompt_template(self, thread, input_variables):
        """Create a prompt template using the tokenizer and the conversation thread."""
        prompt_template = self.tokenizer.apply_chat_template(thread, tokenize=False)
        return PromptTemplate(input_variables=input_variables, template=prompt_template)

    def _create_llm_chain(self, fetch_new_documents, prompt):
        """Create the LLM chain for invoking the RAG pipeline."""
        if fetch_new_documents:
            return {
                "docs": self.ensemble_retriever,
                "context": self.ensemble_retriever | RAGHelper.format_documents,
                "question": RunnablePassthrough()
            } | LLMChain(llm=self.llm, prompt=prompt)
        return {"question": RunnablePassthrough()} | LLMChain(llm=self.llm, prompt=prompt)

    @staticmethod
    def _invoke_rag_chain(user_query, llm_chain):
        """Invoke the RAG pipeline for the user's query."""
        return llm_chain.invoke(user_query)

    @staticmethod
    def _extract_reply(response):
        """Extract and clean the LLM reply from the response."""
        end_string = os.getenv("llm_assistant_token")
        reply = response['text'][response['text'].rindex(end_string) + len(end_string):]
        return re.sub(r'\W+ ', '', reply)

    def _track_provenance(self, user_query, reply, thread):
        """Track the provenance of the LLM response and annotate documents with provenance scores."""
        provenance_method = os.getenv("provenance_method")
        if provenance_method in ['rerank', 'attention', 'similarity', 'llm']:
            answer = self._extract_reply(reply)
            new_history = [{"role": msg["role"], "content": msg["content"].format_map(reply)} for msg in thread]
            new_history.append({"role": "assistant", "content": answer})
            context = RAGHelper.format_documents(reply['docs']).split("\n\n<NEWDOC>\n\n")

            provenance_scores = self._compute_provenance(provenance_method, user_query, reply, context, answer, new_history)
            for i, score in enumerate(provenance_scores):
                reply['docs'][i].metadata['provenance'] = score

    def _compute_provenance(self, provenance_method, user_query, reply, context, answer, new_history):
        """Compute provenance scores based on the selected method."""
        if provenance_method == "rerank":
            return self._compute_rerank_provenance(user_query, reply, answer)
        if provenance_method == "attention":
            return compute_attention(self.model, self.tokenizer,
                                     self.tokenizer.apply_chat_template(new_history, tokenize=False), user_query,
                                     context, answer)
        if provenance_method == "similarity":
            return self.attributor.compute_similarity(user_query, context, answer)
        if provenance_method == "llm":
            return compute_llm_provenance(self.tokenizer, self.model, user_query, context, answer)
        return []

    def _compute_rerank_provenance(self, user_query, reply, answer):
        """Compute rerank-based provenance for the documents."""
        if not os.getenv("rerank") == "True":
            raise ValueError(
                "Provenance attribution is set to rerank but reranking is not enabled. Please choose another provenance method or enable reranking.")

        reranked_docs = compute_rerank_provenance(self.compressor, user_query, reply['docs'], answer)
        return [d.metadata['relevance_score'] for d in reranked_docs if
                d.page_content in [doc.page_content for doc in reply['docs']]]
