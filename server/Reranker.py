from rank_bm25 import BM25Okapi
import sqlite3
import pandas as pd
import os
import json

class Reranker:
    def __init__(self, feedback_db='feedback.db', data_dir='data_directory'):
        self.feedback_db = feedback_db
        self.data_dir = data_dir
        print('data_directory in INIT:', self.data_dir)
        # data_directory = os.getenv('data_directory')
        # print('data_directory in INIT:', data_directory)
        # self.data_dir = data_directory

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
    
    def get_documents(self):
        """
        Retrieve a list of documents from the data directory.

        This endpoint checks the configured data directory and returns a list of files
        that match the specified file types.

        Returns:
            JSON response containing the list of files.
        # """   
        # data_dir = self.data_dir    
        # print('data_dir in get_documents:', data_dir)    
        
        data_directory = os.getenv('data_directory')
        print('data_directory in get_documents:', data_directory)
        file_types = os.getenv("file_types", "").split(",")
        print('file_types:', file_types)

        # Filter files based on specified types
        files = [f for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f)) and os.path.splitext(f)[1][1:] in file_types]
        
        print('files in get_documents:', files)
        print('files type in get_documents:', type(files))
        
        return jsonify(files)
    
    def combiner(self, feedback):
        print('feedback in combiner:', feedback)
        for doc in range(len(feedback['document_id'])):
            document_ids = feedback['document_id'][doc].replace('[', '').replace(']', '').replace("'", "").split(',')
            # print('document_ids:', document_ids)
            # print('document_ids length:', len(document_ids))
            
            rating = feedback['rating'][doc]
            # print('rating:', rating)
            
            for doc_id in range(len(document_ids)):
                print('document_id:', document_ids[doc_id])
                document_id = document_ids[doc_id]
                
                # Create a new dataframe with document_id and rating
                new_row = pd.DataFrame({'document_id': [document_id], 'rating': [rating]})
                

                if 'combined_df' not in locals():
                    combined_df = new_row
                    
                # Append the new row to the combined dataframe
                if document_id in combined_df['document_id'].values:
                    combined_df.loc[combined_df['document_id'] == document_id, 'rating'] += rating
                else:
                    combined_df = pd.concat([combined_df, new_row], ignore_index=True)

        print('combined_df:', combined_df)
        print('combined_df length:', len(combined_df))
        
        return combined_df
            

    # def retrieve_with_bm25(self, query, documents):
    #     """
    #     Retrieve documents based on BM25 scores.
    #     """
    #     # Tokenize the documents
    #     tokenized_docs = [doc.split() for doc in documents]

    #     # Initialize BM25
    #     bm25 = BM25Okapi(tokenized_docs)

    #     # Tokenize the query
    #     tokenized_query = query.split()

    #     # Retrieve scores
    #     scores = bm25.get_scores(tokenized_query)

    #     # Combine documents with their scores
    #     retrieved_docs = [{"document": doc, "bm25_score": score} for doc, score in zip(documents, scores)]

    #     return retrieved_docs

    # def compute_relevance_score(self, bm25_score, feedback_score, alpha=0.7, beta=0.3):
    #     """
    #     Compute the relevance score based on BM25 score and user feedback.
    #     """
    #     return alpha * bm25_score + beta * feedback_score

    # def rerank_documents_with_feedback(self, query, documents):
    #     """
    #     Rerank documents based on BM25 scores and user feedback.
    #     """
    #     # Retrieve documents with BM25 scores
    #     retrieved_docs = self.retrieve_with_bm25(query, documents)

    #     # Fetch feedback scores for the documents
    #     for doc in retrieved_docs:
    #         feedback = self.get_feedback()
    #         feedback_score = sum([f[3] for f in feedback]) if feedback else 0
    #         doc["feedback_score"] = feedback_score

    #     # Compute relevance scores
    #     for doc in retrieved_docs:
    #         doc["relevance_score"] = self.compute_relevance_score(doc["bm25_score"], doc["feedback_score"])

    #     # Sort documents by relevance score
    #     reranked_docs = sorted(retrieved_docs, key=lambda x: x["relevance_score"], reverse=True)

    #     return reranked_docs
    

                
        
        
        
    
    def main_reranker(self):
        feedback_df = self.get_feedback()
        combined_df = self.combiner(feedback_df)
        documents = self.get_documents()
        print(feedback_df)
        print('document_id colum:', feedback_df['document_id'])
        print('document_id type:', type(feedback_df['document_id']))
        print('document_id series:', feedback_df['document_id'])
        
        

    
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
    

