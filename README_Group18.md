# README- Group18

Author： Hanyue Wang，Xuechun LYU and Hanieh Attarimoghadam

# Enhancing the RAG Me Up Framework with Text2SQL Component
This project extends the existing **RAG Me Up framework** by integrating a **Text2SQL component**, enabling the framework to process natural language queries and convert them into SQL queries. This addition improves the framework's capabilities in handling structured data stored in databases, allowing users to retrieve insights efficiently.

## Introduction to Text2SQL
**Text2SQL** is a specialized component that enables natural language queries to be seamlessly converted into SQL commands, bridging the gap between conversational interfaces and relational databases. By leveraging the suriya7/t5-base-text-to-sql model from Hugging Face, Text2SQL empowers users to interact with structured data without needing to know SQL syntax.

In this project, Text2SQL plays a pivotal role by:

- Enhancing Usability: Allowing users to query structured databases with natural language, making data retrieval more intuitive.
- Integrating with the RAG Framework: Acting as a critical retrieval mechanism for structured data sources, complementing the framework’s existing capabilities for unstructured data retrieval.
- Improving Accessibility: Making database-driven applications more user-friendly for non-technical users.

With Text2SQL, the **RAG Me Up framework** can now seamlessly handle mixed queries spanning both structured and unstructured data, broadening its application and utility.

## The RAG pipeline with Text2sql pipeline is visualized in the image below:
![RAG pipeline drawing](./RAG_with_Text2sql.drawio.svg)


## Key Changes and Additions
### 1. New Component: Text2SQL
- File: `server/text2_sql.py`
- Description: A Text2SQL class was added to the framework, which uses the Hugging Face model suriya7/t5-base-text-to-sql to convert natural language queries into SQL queries.
#### Features:
- Tokenizer and Model Initialization: Leverages `AutoTokenizer` and `AutoModelForSeq2SeqLM` for translation tasks.
- Database Integration: Allows connection to a PostgreSQL database through a configurable db_uri.

```python
    def __init__(self, model_name="suriya7/t5-base-text-to-sql", db_uri=None):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    self.db_uri = db_uri
    # Configure logger
    self.logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
```

- Query Translation (`translate` method): Converts a natural language query into an SQL query using the NLP model.
```python
def translate(self, user_query):
    self.logger.info(f"Translating user query: {user_query}")
    input_text = f"translate English to SQL: {user_query}"
    input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
    outputs = self.model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated SQL query: {sql_query}")
    return sql_query
```
- SQL Execution (`execute` method): Executes the translated SQL query on a PostgreSQL database and retrieves results.
- Error Handling: Includes robust logging for query translation and execution.

```python
def execute(self, sql_query):
    try:
        print(f"Executing SQL query: {sql_query}")
        connection = psycopg2.connect(self.db_uri)
        cursor = connection.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        connection.close()
        print(f"SQL query executed successfully, results: {results}")
        return results
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return {"error": str(e)}
```
### 2. Environment Configuration Updates

- File: `server/.env.template`
- Modifications: Added `sql_db`:
  - Specifies the database connection string for SQL-based retrieval.
```python
  sql_db='postgresql://langchain:langchain@localhost:6024/langchain'
```
  - This key enables the Text2SQL component to connect to a PostgreSQL database and execute SQL queries generated from natural language inputs.

    
### 3. Integration with RAG Framework

- File: `server/RAGHelper.py`
  - To integrate the Text2SQL component into the RAG Me Up framework, we made the following modifications in `server/RAGHelper.py`:
#### Modifications:
- Method `_initialize_retrievers`:
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
self.text_to_sql = TextToSQL(model_name=text_to_sql_model, db_uri=sql_db)

self.logger.info("RAGHelper initialized successfully with Text-to-SQL support.")
```

 - Method `retrieve_from_sql`:
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

 - Method `inference_pipeline`:
   - The inference_pipeline method integrates various retrieval mechanisms (dense, sparse, and SQL-based) and leverages a language model to generate a final answer for the user’s query. It acts as the central pipeline that orchestrates the retrieval and reasoning processes, making it a core component of the system.

   - How It Works:
     - 1. Retrieval:
       - Dense Retrieval: Finds relevant documents using vector embeddings (e.g., Milvus).
       - Sparse Retrieval: Matches documents using BM25 for keyword-based search.
       - SQL Retrieval: Translates natural language queries into SQL, executes them on a database, and retrieves results.
     - 2. History Handling:
       - Reuses previous results if available or fetches new documents as needed.
     - 3. Answer Generation:
       - Combines retrieved results into a context.
       - Uses an LLM to generate an answer based on the query and context.

    - Role in the Project:
      - 1. Unifies structured (SQL) and unstructured (document) data retrieval.
      - 2. Supports multi-turn queries by maintaining context with `history`.
      - 3. Enhances user experience by providing clear and accurate answers.




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

### Answer comparison
With the integration of the Text2SQL component, the RAG Me Up system provides more comprehensive and detailed answers. By supporting SQL queries, the system can extract relevant information from structured databases, complementing the dense and sparse retrieval mechanisms. This enhancement significantly improves the specificity and accuracy of the answers, allowing the system to combine insights from both unstructured text and structured data efficiently.

From the comparison below, as shown in Figure: Answer Without Text2SQL and Figure: Answer With Text2SQL, it is evident that the addition of Text2SQL enables the system to deliver more in-depth and contextually rich responses.

- **Figure: Answer Without Text2SQL：**
![RAG original answer](./original_answer.jpg)
- **Figure: Answer With Text2SQL**
![answer after add text2sql to RAG](./answer_after_text2sql.jpg)

## Project Summary
This project successfully integrates the Text2SQL component into the RAG Me Up framework, enhancing its ability to handle both unstructured and structured data. The system now combines dense, sparse, and SQL-based retrievals to deliver highly accurate, detailed, and contextually rich answers. This advancement bridges the gap between conversational AI and structured data, making the framework more versatile and capable of supporting diverse use cases in knowledge retrieval.

### Strengths
- Enhanced Retrieval Capabilities:
  - Supports unstructured document retrieval through dense and sparse methods.
  - Adds SQL-based structured data retrieval, enabling seamless interaction with databases.
- Improved Answer Quality:
  - Combines multiple retrieval sources to generate comprehensive and accurate answers.
  - Uses an LLM to integrate insights from all data sources into a unified response.
- Scalability:
  - Flexible design allows the integration of additional retrieval components or LLMs in the future.
- User-Friendly:
  - Simplifies complex database queries by allowing natural language inputs, making the system accessible to non-technical users.

### Future Improvements
- Optimization of Query Processing:
  - Implement caching mechanisms or query optimization techniques to reduce latency for repeated or similar queries.
- Advanced SQL Generation:
  - Enhance the Text2SQL model to better handle complex queries by training on more diverse datasets or fine-tuning with domain-specific data.
- Result Validation:
  - Introduce post-processing steps to validate the accuracy and relevance of SQL results before including them in the final answer.
- User Interaction:
  - Develop an interactive feedback mechanism where users can refine their queries or provide corrections to improve the system's accuracy.
- Broader Data Source Integration:
  - Extend support to other structured data formats (e.g., NoSQL databases) and multi-modal data (e.g., images, videos).
- Error Transparency:
  - Improve error reporting to clearly indicate issues in retrieval or query execution, helping users better understand the limitations of the response.

