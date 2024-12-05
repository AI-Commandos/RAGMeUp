class Reranker:
    def __init__(self, 
                 feedback_db='feedback.db'):
        self.feedback_db = feedback_db


    def get_feedback():
        """
        Fetch feedback for a specific document from the feedback database.
        """
        conn = sqlite3.connect(self.feedback.db)
        # cursor = conn.cursor()

        query = f"""
        SELECT query, answer, document_id, rating
        FROM Feedback 
        WHERE document_id = ?
        """	
        
        # cursor.execute(
        #     "SELECT query, answer, rating, timestamp FROM Feedback WHERE document_id = ?",
        #     (document_id,)
        # )
        # feedback = cursor.fetchall()

        feedback_df = pd.read_sql_query(query, conn)
        conn.close()

        # # Format feedback as a list of dictionaries
        # feedback_list = [
        #     {"query": row[0], "answer": row[1], "rating": row[2], "timestamp": row[3]} for row in feedback
        # ]
        print('feedback_df in Reranker.py:', feedback_df)
        return feedback_df


    def compute_relevance_score(bm25_score, feedback_score, alpha=0.7, beta=0.3):
        """
        Compute the relevance score based on BM25 score and user feedback.
        """
        return alpha * bm25_score + beta * feedback_score

    def rerank_documents_with_feedback(query, documents, feedback_db):
        """
        Rerank documents based on BM25 scores and user feedback.
        """
        # Retrieve documents with BM25 scores
        retrieved_docs = retrieve_with_bm25(query)

        # Fetch feedback scores for the documents
        for doc in retrieved_docs:
            feedback = get_feedback()  # Fetch cumulative thumbs feedback
            print('feedback in rerank_documents_with_feedback:', feedback)
            doc["feedback_score"] = feedback if feedback is not None else 0

        # Compute new relevance scores
        for doc in retrieved_docs:
            doc["relevance_score"] = compute_relevance_score(
                bm25_score=doc["bm25_score"], 
                feedback_score=doc["feedback_score"]
            )

        # Sort documents by relevance score
        reranked_docs = sorted(retrieved_docs, key=lambda x: x["relevance_score"], reverse=True)
        return reranked_docs
