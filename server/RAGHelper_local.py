import os
import torch

from provenance import (compute_attention, compute_rerank_provenance, compute_llm_provenance, DocumentSimilarityAttribution)
from ScoredCrossEncoderReranker import ScoredCrossEncoderReranker
from RAGHelper import RAGHelper
from RAGHelper import formatDocuments

from transformers import BitsAndBytesConfig
from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig,
  pipeline,
)

from langchain.chains.llm import LLMChain
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

import re
import pickle

class RAGHelperLocal(RAGHelper):
    def __init__(self, logger):
        llm_model = os.getenv('llm_model')
        trust_remote_code = os.getenv('trust_remote_code') == "True"
        
        # Quantization doesn't work on CPU
        if torch.backends.mps.is_available():
            # running on MacOS with Metal available
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=trust_remote_code)
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16,  # Use float16 for better performance
                device_map="auto"
            )
            self.model = self.model.to(torch.device("mps"))
        elif not(os.getenv('force_cpu') == "True"):
            # Set up the LLM
            use_4bit = True
            bnb_4bit_compute_dtype = "float16"
            bnb_4bit_quant_type = "nf4"

            use_nested_quant = False
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
            )

            if compute_dtype == torch.float16 and use_4bit:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    logger.debug("=" * 80)
                    logger.debug("Your GPU supports bfloat16: accelerate training with bf16=True")
                    logger.debug("=" * 80)
            
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=trust_remote_code)

            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                quantization_config=bnb_config,
                trust_remote_code=trust_remote_code
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=trust_remote_code)
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                trust_remote_code=trust_remote_code,
                device='cpu'
            )

        text_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=float(os.getenv('temperature')),
            repetition_penalty=float(os.getenv('repetition_penalty')),
            return_full_text=True,
            max_new_tokens=int(os.getenv('max_new_tokens')),
        )

        self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        # Set up embedding handling for vector store
        if torch.backends.mps.is_available():
            model_kwargs = {
                'device': 'mps'
            }
        elif os.getenv('force_cpu') == "True":
            model_kwargs = {
                'device': 'cpu'
            }
        else:
            model_kwargs = {
                'device': 'cuda'
            }
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv('embedding_model'),
            model_kwargs=model_kwargs
        )

        # Load the data
        self.loadData()

        # Create the RAG chain for determining if we need to fetch new documents
        rag_thread = [{
            'role': 'system', 'content': os.getenv('rag_fetch_new_instruction')
        }, {
            'role': 'user', 'content': os.getenv('rag_fetch_new_question')
        }]
        rag_prompt_template = self.tokenizer.apply_chat_template(rag_thread, tokenize=False)
        rag_prompt = PromptTemplate(
            input_variables=["question"],
            template=rag_prompt_template,
        )
        rag_llm_chain = LLMChain(llm=self.llm, prompt=rag_prompt)
        self.rag_fetch_new_chain = (
            {"question": RunnablePassthrough()} |
            rag_llm_chain
        )

        # For provenance
        if os.getenv("provenance_method") == "similarity":
            self.attributor = DocumentSimilarityAttribution()

        # Also create the rewrite loop LLM chain, if need be
        self.rewrite_ask_chain = None
        self.rewrite_chain = None
        if os.getenv("use_rewrite_loop") == "True":
            # First the chain to ask the LLM if a rewrite would be required
            rewrite_ask_thread = [{
                'role': 'system', 'content': os.getenv('rewrite_query_instruction')
            }, {
                'role': 'user', 'content': os.getenv('rewrite_query_question')
            }]
            rewrite_ask_prompt_template = self.tokenizer.apply_chat_template(rewrite_ask_thread, tokenize=False)
            rewrite_ask_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=rewrite_ask_prompt_template,
            )
            rewrite_ask_llm_chain = LLMChain(llm=self.llm, prompt=rewrite_ask_prompt)
            context_retriever = self.ensemble_retriever
            if os.getenv("rerank") == "True":
                context_retriever = self.rerank_retriever
            self.rewrite_ask_chain = (
                {"context": context_retriever | formatDocuments, "question": RunnablePassthrough()} |
                rewrite_ask_llm_chain
            )

            # Next the chain to ask the LLM for the actual rewrite(s)
            rewrite_thread = [{
                'role': 'user', 'content': os.getenv('rewrite_query_prompt')
            }]
            rewrite_prompt_template = self.tokenizer.apply_chat_template(rewrite_thread, tokenize=False)
            rewrite_prompt = PromptTemplate(
                input_variables=["question"],
                template=rewrite_prompt_template,
            )
            rewrite_llm_chain = LLMChain(llm=self.llm, prompt=rewrite_prompt)
            self.rewrite_chain = (
                {"question": RunnablePassthrough()} |
                rewrite_llm_chain
            )

    def handle_rewrite(self, user_query):
        # Check if we even need to rewrite or not
        if os.getenv("use_rewrite_loop") == "True":
            # Ask the LLM if we need to rewrite
            response = self.rewrite_ask_chain.invoke(user_query)
            end_string = os.getenv("llm_assistant_token")
            reply = response['text'][response['text'].rindex(end_string)+len(end_string):]
            reply = re.sub(r'\W+ ', '', reply)
            if reply.lower().startswith('no'):
                # Start the rewriting into different alternatives
                response = self.rewrite_chain.invoke(user_query)
                # Take out the alternatives
                reply = response['text'][response['text'].rindex(end_string)+len(end_string):]
                # Show be split by newlines
                return reply
            else:
                # We do not need to rewrite
                return user_query
        else:
            return user_query

    # Main function to handle user interaction
    def handle_user_interaction(self, user_query, history):
        if len(history) == 0:
            fetch_new_documents = True
        else:
            # Prompt for LLM
            response = self.rag_fetch_new_chain.invoke(user_query)
            end_string = os.getenv("llm_assistant_token")
            reply = response['text'][response['text'].rindex(end_string)+len(end_string):]
            reply = re.sub(r'\W+ ', '', reply)
            if reply.lower().startswith('yes'):
                fetch_new_documents = True
            else:
                fetch_new_documents = False

        # Create prompt template based on whether we have history or not
        thread = [{"role": x["role"], "content": x["content"].replace("{", "(").replace("}", ")")} for x in history]
        if fetch_new_documents:
            thread = []
        if len(thread) == 0:
            thread.append({
                'role': 'system', 'content': os.getenv('rag_instruction')})
            thread.append({
                'role': 'user', 'content': os.getenv('rag_question_initial')
            })
        else:
            thread.append({
                'role': 'user', 'content': os.getenv('rag_question_followup')
            })

        # Determine input variables
        if fetch_new_documents:
            input_variables = ["context", "question"]
        else:
            input_variables = ["question"]

        prompt_template = self.tokenizer.apply_chat_template(thread, tokenize=False)

        # Create prompt from prompt template
        prompt = PromptTemplate(
            input_variables=input_variables,
            template=prompt_template,
        )

        # Create llm chain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        if fetch_new_documents:
            # Rewrite the question if needed
            user_query = self.handle_rewrite(user_query)
            context_retriever = self.ensemble_retriever
            if os.getenv("rerank") == "True":
                context_retriever = self.rerank_retriever
            rag_chain = (
                {"docs": context_retriever, "context": context_retriever | formatDocuments, "question": RunnablePassthrough()} |
                llm_chain
            )
        else:
            rag_chain = (
                {"question": RunnablePassthrough()} |
                llm_chain
            )
        
        # Check if we need to apply Re2 to mention the question twice
        if os.getenv("use_re2") == "True":
            user_query = f'{user_query}\n{os.getenv("re2_prompt")}{user_query}'

        # Invoke RAG pipeline
        reply = rag_chain.invoke(user_query)

        # See if we need to track provenance
        if fetch_new_documents and os.getenv("provenance_method") in ['rerank', 'attention', 'similarity', 'llm']:
            # Add the user question and the answer to our thread for provenance computation
            end_string = os.getenv("llm_assistant_token")
            answer = reply['text'][reply['text'].rindex(end_string)+len(end_string):]
            new_history = [{"role": msg["role"], "content": msg["content"].format_map(reply)} for msg in thread]
            new_history.append({"role": "assistant", "content": answer})
            context = formatDocuments(reply['docs']).split("\n\n<NEWDOC>\n\n")

            # Use the reranker but now on the answer (and potentially query too)
            if os.getenv("provenance_method") == "rerank":
                if not(os.getenv("rerank") == "True"):
                    raise ValueError("Provenance attribution is set to rerank but reranking is not enabled. Please choose another provenance method or turn on reranking.")
                reranked_docs = compute_rerank_provenance(self.compressor, user_query, reply['docs'], answer)
                
                # This is a bit of a hassle because reranked_docs is now reordered and we have no definitive key to use because of hybrid search.
                # Note that we can't just return reranked_docs because the LLM may refer to "doc #1" in the order of the original scoring.
                provenance_scores = []
                for doc in reply['docs']:
                    # Find the document in reranked_docs
                    reranked_score = [d.metadata['relevance_score'] for d in reranked_docs if d.page_content == doc.page_content][0]
                    provenance_scores.append(reranked_score)
            # See if we need to do attention-based provenance
            elif os.getenv("provenance_method") == "attention":
                # Now compute the attention scores and add them to the docs
                provenance_scores = compute_attention(self.model, self.tokenizer, self.tokenizer.apply_chat_template(new_history, tokenize=False), user_query, context, answer)
            # See if we need to do similarity-base provenance
            elif os.getenv("provenance_method") == "similarity":
                provenance_scores = self.attributor.compute_similarity(user_query, context, answer)
            # See if we need to use LLM-based provenance
            elif os.getenv("provenance_method") == "llm":
                provenance_scores = compute_llm_provenance(self.tokenizer, self.model, user_query, context, answer)
            
            # Add the provenance scores
            for i, score in enumerate(provenance_scores):
                reply['docs'][i].metadata['provenance'] = score

        return (thread, reply)

    def addDocument(self, filename):
        if filename.lower().endswith('pdf'):
            docs = PyPDFLoader(filename).load()
        if filename.lower().endswith('json'):
            docs = JSONLoader(
                file_path = filename,
                jq_schema = os.getenv("json_schema"),
                text_content = os.getenv("json_text_content") == "True",
            ).load()
        if filename.lower().endswith('csv'):
            docs = CSVLoader(filename).load()
        if filename.lower().endswith('docx'):
            docs = Docx2txtLoader(filename).load()
        if filename.lower().endswith('xlsx'):
            docs = UnstructuredExcelLoader(filename).load()
        if filename.lower().endswith('pptx'):
            docs = UnstructuredPowerPointLoader(filename).load()

        # Skills and personality are global and don't work on chunks, so do them first
        new_docs = []
        for doc in docs:
            # Get the skills by using the LLM and attach to the doc
            skills = self.parseCV(doc)
            doc.metadata['skills'] = skills
            # Also get the personality
            doc.metadata['personality'] = self.personality_predictor.predict(doc)
            new_docs.append(doc)

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

        new_chunks = self.text_splitter.split_documents(new_docs)

        self.chunked_documents = self.chunked_documents + new_chunks
        # Store these too, for our sparse DB
        with open(f"{os.getenv('vector_store_uri')}_sparse.pickle", 'wb') as f:
            pickle.dump(self.chunked_documents, f)

        # Add to vector DB
        self.db.add_documents(new_chunks)

        # Add to BM25
        bm25_retriever = BM25Retriever.from_texts(
            [x.page_content for x in self.chunked_documents],
            metadatas=[x.metadata for x in self.chunked_documents]
        )

        # Update full retriever too
        retriever = self.db.as_retriever(search_type="mmr", search_kwargs = {'k': int(os.getenv('vector_store_k'))})
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )

        if os.getenv("rerank") == "True":
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