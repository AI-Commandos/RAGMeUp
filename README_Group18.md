# README- Group18
# Enhancing the RAG Me Up Framework with Text2SQL Component
This project extends the existing **RAG Me Up framework** by integrating a **Text2SQL component**, enabling the framework to process natural language queries and convert them into SQL queries. This addition improves the framework's capabilities in handling structured data stored in databases, allowing users to retrieve insights efficiently.

## Key Changes and Additions
### 1. New Component: Text2SQL

File: server/text2_sql.py
Description:
A Text2SQL class was added to the framework, which uses the Hugging Face model suriya7/t5-base-text-to-sql to convert natural language queries into SQL queries.
Features:
Tokenizer and Model Initialization: Leverages AutoTokenizer and AutoModelForSeq2SeqLM for translation tasks.
Database Integration: Allows connection to a PostgreSQL database through a configurable db_uri.
Error Handling: Includes robust logging for query translation and execution.
### 2. Environment Configuration Updates

File: server/.env.template
Modifications:
Added the following keys for configuring the Text2SQL component:
db_uri: Specifies the database connection URI.
llm_model: Specifies the Hugging Face Text2SQL model.
Updated other configuration keys to enable compatibility with the new component.
The modified section includes instructions on initializing and integrating this component.
### 3. Integration with RAG Framework

File: server/RAGHelper.py
Modifications:
Method _initialize_retrievers:
Added a logic branch for Text2SQL to allow SQL-based queries when vector_store is set to postgres.
New Method retrieve_from_sql:
Accepts a natural language query, translates it into SQL using the Text2SQL component, and executes it on the database.
Logs both the generated SQL query and its results for traceability.

## How to Use
1. Setup

Update the .env file using the template server/.env.template:
、、、
HF_TOKEN=your_huggingface_api_token
db_uri=your_postgresql_connection_uri
llm_model=suriya7/t5-base-text-to-sql
、、、




