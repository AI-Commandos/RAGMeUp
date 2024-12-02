from PostgresBM25Retriever import PostgresBM25Retriever  
from RerankerReorder import ScoredCrossEncoderReranker 

# Step 1: Initialize the retriever
connection_uri = 'data' #"postgresql://username:password@localhost:5432/your_database"
#table_name = "your_table_name"
retriever = PostgresBM25Retriever(connection_uri=connection_uri, k=10)

# Step 2: Fetch relevant documents for a given query
query = "Deep Learning"
retrieved_documents = retriever._get_relevant_documents(query=query, run_manager=None)

# Step 3: Instantiate the reranker
#reranker = ScoredCrossEncoderReranker(model=your_cross_encoder_model, top_n=5)

# Step 4: Evaluate the reranker
#original_output = reranker.compress_documents(retrieved_documents, query)
#print("Original Output (without reordering):")
#for doc in original_output:
#    print(doc.page_content[:100], doc.metadata)

# Step 5: Compare with modified reranker (with reordering)
# Assuming your modified reranker includes the long-context reordering
#modified_output = reranker.compress_documents(retrieved_documents, query)
#print("Modified Output (with reordering):")
#for doc in modified_output:
#    print(doc.page_content[:100], doc.metadata)

# Optional: Close the database connection
#retriever.close()