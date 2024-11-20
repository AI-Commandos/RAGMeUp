import os
import spacy

from RAGHelper_cloud import RAGHelperCloud
from relik import Relik
from relik.inference.data.objects import RelikOutput
from neo4j import GraphDatabase
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

def combine_results(inputs: dict) -> dict:
    """
    Combine the results from different retrievers and the language model.

    Args:
        inputs (dict): A dictionary containing the inputs from the retrievers and the language model.

    Returns:
        dict: A dictionary containing the combined results.
    """
    combined_docs = inputs.get("docs", [])
    combined_context = inputs.get("context", "")
    combined_question = inputs.get("question", "")
    combined_answer = inputs.get("answer", "")

    return {
        "docs": combined_docs,
        "context": combined_context,
        "question": combined_question,
        "answer": combined_answer
    }

class GraphRAGHelper(RAGHelperCloud):
    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger
        self.relation_extractor = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-nyt-large")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))  
        self.nlp = spacy.load("en_core_web_sm")

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
        # TODO: dynamically create cypher query (with LLM)
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
        fetch_new_documents = self.should_fetch_new_documents(user_query, history)

        thread = self.create_interaction_thread(history, fetch_new_documents)
        # Create prompt from prompt template
        prompt = ChatPromptTemplate.from_messages(thread)

        # Create llm chain
        llm_chain = prompt | self.llm

        if fetch_new_documents:
            # Use graph-based retrieval to fetch related documents
            related_docs = self.retrieve_documents(user_query)
            context_retriever = self.ensemble_retriever if self.rerank else self.rerank_retriever
            retriever_chain = {
                "docs": context_retriever,
                "context": context_retriever | RAGHelperCloud.format_documents,
                "question": RunnablePassthrough()
            }
            # Combine graph-based documents with retrieved documents
            rag_chain = (
                retriever_chain
                | RunnablePassthrough.assign(
                    answer=lambda x: llm_chain.invoke(
                        {"docs": related_docs + x["docs"], "context": x["context"], "question": x["question"]}
                    ))
                | combine_results
            )
        else:
            retriever_chain = {"question": RunnablePassthrough()}
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
        response = rag_chain.invoke(user_query)

        # Track provenance if needed
        if fetch_new_documents and os.getenv("provenance_method") in ['rerank', 'attention', 'similarity', 'llm']:
            self.track_provenance(response, user_query)

        return (thread, response)