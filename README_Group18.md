# README- Group18

Author： Hanyue Wang，Xuechun LYU and Hanieh Attarimoghadam

# Enhancing the RAG Me Up Framework with Text2SQL Component
This project extends the existing **RAG Me Up framework** by integrating a **Text2SQL component**, enabling the framework to process natural language queries and convert them into SQL queries. This addition improves the framework's capabilities in handling structured data stored in databases, allowing users to retrieve insights efficiently.

## Key Changes and Additions
### 1. New Component: Text2SQL
- File: server/text2_sql.py
- Description: A Text2SQL class was added to the framework, which uses the Hugging Face model suriya7/t5-base-text-to-sql to convert natural language queries into SQL queries.
- Features:
  - Tokenizer and Model Initialization: Leverages AutoTokenizer and AutoModelForSeq2SeqLM for translation tasks.
  - Database Integration: Allows connection to a PostgreSQL database through a configurable db_uri.
  - Error Handling: Includes robust logging for query translation and execution.
### 2. Environment Configuration Updates

- File: server/.env.template
- Modifications:
  - Added the following keys for configuring the Text2SQL component:
    - db_uri: Specifies the database connection URI.
    - llm_model: Specifies the Hugging Face Text2SQL model.
  - Updated other configuration keys to enable compatibility with the new component.
  - The modified section includes instructions on initializing and integrating this component.
    
### 3. Integration with RAG Framework

- File: server/RAGHelper.py
  - To integrate the Text2SQL component into the RAG Me Up framework, we made the following modifications in `server/RAGHelper.py`:
- Modifications:
  - 1. Method `_initialize_retrievers`:
    - Added a logic branch for Text2SQL to allow SQL-based queries when vector_store is set to postgres.
    - Purpose: This block initializes the Text2SQL component during the setup of the `RAGHelper` class.
    - Details:
      - Retrieves the model name (`text_to_sql_model`) from environment variables or defaults to `suriya7/t5-base-text-to-sql`.
      - Establishes a connection to the PostgreSQL database using `self.vector_store_sparse_uri` as the `db_uri`.
      - Logs the initialization process to ensure the component is set up correctly.
```python
# Initialize Text-to-SQL
text_to_sql_model = os.getenv("text_to_sql_model", "suriya7/t5-base-text-to-sql")
self.logger.info(f"Initializing Text2SQL with model: {text_to_sql_model}")
self.text_to_sql = TextToSQL(model_name=text_to_sql_model, db_uri=self.vector_store_sparse_uri)

self.logger.info("RAGHelper initialized successfully with Text-to-SQL support.")
```
      
  - 2. New Method `retrieve_from_sql`:
    - Purpose:
      - This method allows the RAG pipeline to handle natural language queries that require SQL-based retrieval.
    - Details:
      - Query Translation:
        - The method takes a natural language query (`user_query`) as input.
        - Uses the Text2SQL component's `translate` method to generate an SQL query.
        - Logs the translated SQL query for debugging purposes.
      - SQL Execution:
        - Executes the generated SQL query on the connected database using the Text2SQL component's `execute` method.
        - Logs the execution process and results.
      - Error Handling:
        - Catches exceptions during SQL execution and logs errors to ensure transparency.
        - Returns an empty result set if an error occurs.
      - Return Format:
        - The results are formatted as a list of dictionaries, with each dictionary containing the type (`sql_result`) and content (`result`) of the query output.

```python
def retrieve_from_sql(self, user_query):
    sql_query = self.text_to_sql.translate(user_query)
    self.logger.info(f"Received user query for SQL retrieval: {user_query}")
    try:
        sql_results = self.text_to_sql.execute(sql_query)
        self.logger.info(f"Generated SQL Query: {sql_query}")  
    except Exception as e:
        self.logger.error(f"Error executing SQL: {e}")
        sql_results = []
    self.logger.info(f"SQL Query Results: {sql_results}")
    return [{"type": "sql_result", "content": result} for result in sql_results]
```

## How to Use
### 1. Setup

Update the .env file using the template server/.env.template:

```env
HF_TOKEN=your_huggingface_api_token
db_uri=your_postgresql_connection_uri
llm_model=suriya7/t5-base-text-to-sql
```

Install required dependencies:

```bash
pip install psycopg2 transformers
```

### 2. Run the Server

Start the RAG Me Up server and ensure that the Text2SQL component is initialized.
### 3. Query Execution

Provide a natural language query through the RAG pipeline.
The Text2SQL component translates the query into SQL and retrieves data from the connected database.

## Example
Here’s an example flow using the new component:

User Input: "What is the birth date of Lincoln?"
```sql
SELECT birth_date FROM table_name_94 WHERE name = "Lincoln";
```
Output: The birth date of Lincoln from the database.

## Benefits
- Enhanced Query Flexibility: Supports complex natural language queries by converting them into SQL.
- Streamlined Data Access: Integrates structured data retrieval seamlessly into the RAG framework.
- Extensible Design: Can be adapted for additional databases and SQL variations in the future.


