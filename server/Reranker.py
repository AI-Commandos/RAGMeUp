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
        
    
    def combiner(self, feedback, documents_lst):
        # print('feedback in combiner:', feedback)
        for doc in range(len(feedback['document_id'])):
            document_ids = feedback['document_id'][doc].replace('[', '').replace(']', '').replace(' " ', '').split(',')
            # print('document_ids:', document_ids)
            # print('document_ids length:', len(document_ids))
            
            rating = feedback['rating'][doc]
            # print('rating:', rating)
            
            for doc_id in range(len(document_ids)):
                # print('document_id:', document_ids[doc_id])
                document_id = document_ids[doc_id]
                document_id = document_id.strip('"')
                # print('document_id:', document_id)
                
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

        # print('feedback_rating_df:', feedback_rating_df)
        # print('feedback_rating_df length:', len(feedback_rating_df))
        
        # Create a dataframe from documents_lst
        documents_df = pd.DataFrame(documents_lst, columns=['document_id'])
        # print('documents_df:', documents_df)
                    
        # Merge documents_df with feedback_rating_df on 'document_id'
        combined_df = pd.merge(documents_df, feedback_rating_df, on='document_id', how='left')

        # Fill NaN values in the 'rating' column with 0
        combined_df['rating'].fillna(0, inplace=True)
        # Ensure the 'rating' column is of integer type
        combined_df['rating'] = combined_df['rating'].astype(int)
            
        # print('combined_df:', combined_df)
        return combined_df
                
                            

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
        print('scores:', scores)

        # Combine documents with their scores
        retrieved_docs = [{"document": doc, "bm25_score": score} for doc, score in zip(documents, scores)]
        # Convert retrieved_docs to a DataFrame
        retrieved_docs_df = pd.DataFrame(retrieved_docs)
        print('retrieved_docs_df:', retrieved_docs_df)
        return retrieved_docs_df


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
    
    def compute_relevance_score(bm25_score_weight, feedback_weight, alpha=0.7, beta=0.3):
        """
        Compute the relevance score based on BM25 weight and user feedback weight.
        
        Args:
            bm25_score_weight (float): The BM25 relevance score for a document.
            feedback_weight (float): The user feedback weight for a document.
            alpha (float): Weight of BM25 in the relevance formula.
            beta (float): Weight of user feedback in the relevance formula.

        Returns:
            float: Combined relevance score.
        """
        print('bm25_score_weight:', bm25_score_weight)
        print('feedback_weight:', feedback_weight)  
        print('alpha:', alpha)
        print('beta:', beta)
        

        print('alpha type:', type(alpha))
        print('bm25_score_weight type:', type(bm25_score_weight))
        print('beta type:', type(beta))
        print('feedback_weight type:', type(feedback_weight))
        
        print('score:', alpha * bm25_score_weight + beta * feedback_weight)
        
        return alpha * bm25_score_weight + beta * feedback_weight
    
    def rerank_documents_with_feedback(self, query, documents, feedback, combined_df):
        """
        Rerank documents based on BM25 scores and user feedback.

        Args:
            query (str): User query.
            documents (list of str): List of document texts.
            feedback (DataFrame): User feedback DataFrame with ratings and helpfulness.

        Returns:
            list of dict: Reranked documents with relevance scores.
        """
        
        # queries = feedback['query'].unique()
        # print('queries:', queries)
        # for query in queries:
        #     print('query:', query)
        #     feedback_query = feedback.loc[feedback['query'] == query]
        #     print('feedback_query:', feedback_query)
        #     bm25 = self.retrieve_with_bm25(query, documents)
        #     bm25['query'] = query
        #     if 'bm25_df' not in locals():
        #         bm25_df = bm25
        #     else:
        #         bm25_df = pd.concat([bm25_df, bm25], ignore_index=True)
        #     print('bm25:', bm25)
        # print('bm25_df:', bm25_df)
            

        # # Tokenize documents and query
        # tokenized_docs = [doc.split() for doc in documents]
        # bm25 = BM25Okapi(tokenized_docs)
        # query_tokens = query.split()

        # # Get BM25 scores
        # bm25_scores = bm25.get_scores(query_tokens)
        
        bm25_df = self.retrieve_with_bm25(query, documents)
        print('bm25_df in rerank_documents etc.:', bm25_df)
        # Merge bm25_df with combined_df on 'document' and 'document_id'
        combined_df = pd.merge(bm25_df, combined_df, left_on='document', right_on='document_id', how='left')
        print('combined_df after merge with bm_25_df:', combined_df)

        relevance_scores = []
        for i in range(len(combined_df)):
            feedback_score = combined_df['rating'][i]
            print('feedback_score:', feedback_score)
            print('feedback_score type:', type(feedback_score))
            bm25_score_reranker = combined_df['bm25_score'][i]
            print('bm25_score:', bm25_score_reranker)
            print('bm25_score type:', type(bm25_score_reranker))
            relevance_score = self.compute_relevance_score(bm25_score_reranker, feedback_score)
            print('relevance_score:', relevance_score)
            relevance_scores.append(relevance_score)
        combined_df['relevance_score'] = relevance_scores
        print('combined_df after adding relevance_score:', combined_df)
        # # Compute relevance scores for each document
        # reranked_docs = []
        # for i, doc in enumerate(documents):
        #     # Fetch feedback weight for the document
        #     feedback_row = feedback.loc[feedback['document_id'] == i]
        #     feedback_weight = (
        #         feedback_row['rating'].values[0] * feedback_row['helpfulness'].values[0]
        #         if not feedback_row.empty else 0
        #     )

        #     # Compute relevance score
        #     relevance_score = compute_relevance_score(bm25_scores[i], feedback_weight)
            
        #     # Append to results
        #     reranked_docs.append({
        #         "document": doc,
        #         "bm25_score": bm25_scores[i],
        #         "feedback_score": feedback_weight,
        #         "relevance_score": relevance_score
        #     })

        # Sort documents by relevance score
        reranked_docs = sorted(combined_df, key=lambda x: x["relevance_score"], reverse=True)
        print('reranked_docs:', reranked_docs)
        return reranked_docs


          
    def main_reranker(self):
        feedback_df = self.get_feedback_reranker()
        documents_lst = self.get_documents_reranker()
        combined_df = self.combiner(feedback_df, documents_lst)
        query = 'What is Word2Vec?'
        rerank_df = self.rerank_documents_with_feedback(query, documents_lst, feedback_df, combined_df)

        print(feedback_df)
        print('combined_df', combined_df)
        print('rerank_df', rerank_df)
        
        

    
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
    

