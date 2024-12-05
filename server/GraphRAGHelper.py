import os
import spacy
import torch

from RAGHelper import RAGHelper
from RAGHelper_local import RAGHelperLocal
from relik import Relik
from relik.inference.data.objects import RelikOutput
from neo4j import GraphDatabase
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

import re
from provenance import (
    compute_attention,
    compute_rerank_provenance,
    compute_llm_provenance,
    DocumentSimilarityAttribution
)
from langchain.chains.llm import LLMChain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

class GraphRAGHelper(RAGHelper):
    def __init__(self, logger):
        super().__init__(logger)
        
        self.tokenizer, self.model = self._initialize_llm()
        self.llm = self._create_llm_pipeline()

        self.relation_extractor = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-nyt-large")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))  
        self.nlp = spacy.load("en_core_web_sm")

        # Load the data
        self.load_data()

        # # Create RAG chains
        # self.rag_fetch_new_chain = self._create_rag_chain()
        self.rewrite_ask_chain, self.rewrite_chain = self._initialize_rewrite_chains()
    
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
        
        # Store knowledge graph in Neo4j
        self.construct_graph(self.chunked_documents)
        self.save_graph_to_neo4j()

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
    
    def _get_bnb_config(self):
        """Get the BitsAndBytes configuration for quantization."""
        return RAGHelperLocal._get_bnb_config()
    
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

        return {"context": self.retrieve_documents, "question": RunnablePassthrough()} | rewrite_ask_llm_chain

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
        
    @staticmethod 
    def _prepare_conversation_thread(history):
        """Prepare the conversation thread for the user interaction."""
        return RAGHelperLocal._prepare_conversation_thread(history, False)

    def _create_prompt_template(self, thread):
        """Create a prompt template using the tokenizer and the conversation thread."""
        prompt_template = self.tokenizer.apply_chat_template(thread, tokenize=False)
        return PromptTemplate(input_variables=["question"], template=prompt_template)
    
    def _create_llm_chain(self, prompt):
        """Create the LLM chain for invoking the GraphRAG pipeline."""
        return {"question": RunnablePassthrough()} | LLMChain(llm=self.llm, prompt=prompt)
    
    def construct_graph(self, documents):
        graph = {}
        for doc in documents:
            relik_out: RelikOutput = self.relation_extractor(doc.page_content)
            for relation in relik_out.relations:
                subject = relation.subject
                object = relation.object
                predicate = relation.predicate
                if subject not in graph:
                    graph[subject] = []
                graph[subject].append((predicate, object))
        self.graph = graph

    def save_graph_to_neo4j(self):
        with self.driver.session() as session:
            for subject, relations in self.graph.items():
                for predicate, object in relations:
                    session.write_transaction(self._create_relationship, subject, predicate, object)

    @staticmethod
    def _create_relationship(tx, subject, predicate, object):
        query = (
            "MERGE (a:Entity {name: $subject}) "
            "MERGE (b:Entity {name: $object}) "
            "MERGE (a)-[r:RELATION {type: $predicate}]->(b)"
        )
        tx.run(query, subject=subject, predicate=predicate, object=object)

    def retrieve_documents(self, query):
        related_docs = []
        with self.driver.session() as session:
            result = session.read_transaction(self._find_related_documents, query)
            for record in result:
                subject = record["subject"]
                predicate = record["predicate"]
                object = record["object"]
                related_docs.append((subject, predicate, object))
        return related_docs
    
    def _extract_entities(self, text):
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities
    
    def _find_related_documents(self, tx, query):
        entities = self._extract_entities(query)
        if not entities:
            entities = [query]  # Fallback to using the entire query if no entities are found

        cypher_query = (
            "MATCH (a:Entity)-[r:RELATION]->(b:Entity) "
            "WHERE " + " OR ".join([f"a.name CONTAINS '{entity}' OR b.name CONTAINS '{entity}'" for entity in entities]) + " "
            "RETURN a.name AS subject, r.type AS predicate, b.name AS object"
        )
        result = tx.run(cypher_query)
        return result
    
    def handle_user_interaction(self, user_query, history):
        """
        Handle user interaction by processing their query and maintaining conversation history.

        Args:
            user_query (str): The user's query.
            history (list): The history of previous interactions.

        Returns:
            tuple: A tuple containing the conversation thread and the reply.
        """
        thread = self._prepare_conversation_thread(history)
        prompt = self._create_prompt_template(thread)

        self.logger.info("Prepared conversation thread and created prompt template.")

        # Create llm chain
        # llm_chain = self._create_llm_chain(prompt)

        user_query = self.handle_rewrite(user_query)
        # Check if we need to apply Re2 to mention the question twice
        if os.getenv("use_re2") == "True":
            user_query = f'{user_query}\n{os.getenv("re2_prompt")}{user_query}'

        # Retrieve documents
        response = self.retrieve_documents(user_query)

        return (thread, response)

    def add_document(self, filename):
        """
        Add a document to the knowledge graph.

        Args:
            filename (str): The name of the file to be added.
        """
        new_docs = self._load_document(filename)
        self.logger.info("Chunking the documents.")
        new_chunks = self._split_documents(new_docs)
        self._update_chunked_documents(new_chunks)
        self._add_to_vector_database(new_chunks)
        self.construct_graph(new_chunks)
        self.save_graph_to_neo4j()

    def delete_document(self, filename):
        """
        Delete a document from the knowledge graph.

        Args:
            filename (str): The name of the file to be deleted.
        """
        file_path = os.path.join(self.data_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            self.logger.info(f"Deleted file: {filename}")
            # Reconstruct the knowledge graph
            all_docs = self._load_documents()
            self.construct_graph(all_docs)
            self.save_graph_to_neo4j()
        else:
            self.logger.warning(f"File not found: {filename}")