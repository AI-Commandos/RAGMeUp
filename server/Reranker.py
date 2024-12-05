from rank_bm25 import BM25Okapi
import sqlite3
import pandas as pd

class Reranker:
    def __init__(self, feedback_db='feedback.db'):
        self.feedback_db = feedback_db

    def get_feedback(self):
        """
        Fetch feedback for a specific document from the feedback database.
        """
        conn = sqlite3.connect(self.feedback_db)
        
        query = """
        SELECT query, answer, document_id, rating
        FROM Feedback 
        """
        
        feedback = pd.read_sql_query(query, conn)
        conn.close()
        return feedback
    

    def retrieve_with_bm25(self, query, documents):
        """
        Retrieve documents based on BM25 scores.
        """
        # Tokenize the documents
        tokenized_docs = [doc.split() for doc in documents]

        # Initialize BM25
        bm25 = BM25Okapi(tokenized_docs)

        # Tokenize the query
        tokenized_query = query.split()

        # Retrieve scores
        scores = bm25.get_scores(tokenized_query)

        # Combine documents with their scores
        retrieved_docs = [{"document": doc, "bm25_score": score} for doc, score in zip(documents, scores)]

        return retrieved_docs

    def compute_relevance_score(self, bm25_score, feedback_score, alpha=0.7, beta=0.3):
        """
        Compute the relevance score based on BM25 score and user feedback.
        """
        return alpha * bm25_score + beta * feedback_score

    def rerank_documents_with_feedback(self, query, documents):
        """
        Rerank documents based on BM25 scores and user feedback.
        """
        # Retrieve documents with BM25 scores
        retrieved_docs = self.retrieve_with_bm25(query, documents)

        # Fetch feedback scores for the documents
        for doc in retrieved_docs:
            feedback = self.get_feedback()
            feedback_score = sum([f[3] for f in feedback]) if feedback else 0
            doc["feedback_score"] = feedback_score

        # Compute relevance scores
        for doc in retrieved_docs:
            doc["relevance_score"] = self.compute_relevance_score(doc["bm25_score"], doc["feedback_score"])

        # Sort documents by relevance score
        reranked_docs = sorted(retrieved_docs, key=lambda x: x["relevance_score"], reverse=True)

        return reranked_docs
    
    def main_reranker(self):
        feedback_df = self.get_feedback()
        print(feedback_df)
        

    
def main():
    # Example usage
    try:
        reranker = Reranker()
        reranker.main_reranker(
            
        )
    
    except Exception as e:
        print(f"Reranking failed: {e}")
    

if __name__ == "__main__":
    main()
    

