from RAGHelper import RAGHelper
from relik import Relik
from relik.inference.data.objects import RelikOutput

class GraphRAGHelper(RAGHelper):
    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger
        self.relation_extractor = Relik.from_pretrained("sapienzanlp/relik-relation-extraction-nyt-large")

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

    def retrieve_documents(self, query):
        related_docs = []
        for subject, relations in self.graph.items():
            if query in subject:
                for predicate, object in relations:
                    related_docs.append((subject, predicate, object))
        return related_docs