from rank_bm25 import BM25Okapi
import sqlite3
import pandas as pd
import os
# from flask import jsonify

class Reranker:
    def __init__(self, feedback_db='feedback.db', data_directory='data'):
        self.feedback_db = feedback_db
        self.data_dir = os.getenv(data_directory)


    def get_feedback_reranker(self):
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
    
    def get_documents_reranker(self):
        """
        Retrieve a list of documents from the data directory.

        This endpoint checks the configured data directory and returns a list of files
        that match the specified file types.

        Returns:
            JSON response containing the list of files.
        # """   
  
        os.environ['data_directory'] = 'data'
        os.environ["file_types"] = "txt,csv,pdf,json"

        data_directory = os.getenv('data_directory')
        print('data_directory in get_documents_reranker:', data_directory)
        file_types = os.getenv("file_types", "").split(",")
        print('file_types:', file_types)

        # Filter files based on specified types
        files = [f for f in os.listdir(data_directory)
                if os.path.isfile(os.path.join(data_directory, f)) and os.path.splitext(f)[1][1:] in file_types]
        
        print('files in get_documents_reranker:', files)
        print('files type in get_documents_reranker:', type(files))
        
        return files
        
    
            

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
   
    
def combiner(self, feedback, documents_lst):
        # print('feedback in combiner:', feedback)
        for doc in range(len(feedback['document_id'])):
            document_ids = feedback['document_id'][doc].replace('[', '').replace(']', '').replace("'", "").split(',')
            # print('document_ids:', document_ids)
            # print('document_ids length:', len(document_ids))
            
            rating = feedback['rating'][doc]
            # print('rating:', rating)
            
            for doc_id in range(len(document_ids)):
                # print('document_id:', document_ids[doc_id])
                document_id = document_ids[doc_id]
                
                # Create a new dataframe with document_id and rating
                new_row = pd.DataFrame({'document_id': [document_id], 'rating': [rating]})
                
                # Check if feedback_rating_df exist otherwise create it
                if 'feedback_rating_df' not in locals():
                    feedback_rating_df = new_row
                    
                # Append the new row to the combined dataframe
                if document_id in feedback_rating_df['document_id'].values:
                    feedback_rating_df.loc[feedback_rating_df['document_id'] == document_id, 'rating'] += rating
                else:
                    feedback_rating_df = pd.concat([feedback_rating_df, new_row], ignore_index=True)

        print('feedback_rating_df:', feedback_rating_df)
        print('combined_df length:', len(feedback_rating_df))
        
        # Create a dataframe from documents_lst
        documents_df = pd.DataFrame(documents_lst, columns=['document_id'])

        # Merge feedback_rating_df with documents_df on document_id
        combined_df = feedback_rating_df.merge(documents_df, on='document_id', how='inner')
        print('combined_df:', combined_df)
        return combined_df
                
       
        
    
    def main_reranker(self):
        feedback_df = self.get_feedback_reranker()
        documents_lst = self.get_documents_reranker()
        combined_df = self.combiner(feedback_df, documents_lst)

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
    

