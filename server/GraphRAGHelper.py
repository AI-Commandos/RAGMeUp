from RAGHelper import RAGHelper
from relik import Relik
from relik.inference.data.objects import RelikOutput
from neo4j import GraphDatabase

class GraphRAGHelper(RAGHelper):
    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger
        self.relation_extractor = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-nyt-large")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))

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

    @staticmethod
    def _find_related_documents(tx, query):
        cypher_query = (
            "MATCH (a:Entity)-[r:RELATION]->(b:Entity) "
            "WHERE a.name CONTAINS $query OR b.name CONTAINS $query "
            "RETURN a.name AS subject, r.type AS predicate, b.name AS object"
        )
        result = tx.run(cypher_query, query=query)
        return result