# Text2SQL Component in a Retrieval-Augmented Generation (RAG) Framework

## Overview
This project introduces a **Text2SQL retrieval component** into an existing Retrieval-Augmented Generation (RAG) framework. The component allows the system to transform natural language queries into SQL queries, enabling data retrieval from structured databases instead of the sole retrieval of unstructured documents.

The implementation integrates a PostgreSQL database for Text2SQL processing and updates various components of the RAG framework to support SQL-based retrieval seamlessly.

---

## Key Changes and Features

### 1. **Environment Configuration**
- Added new configurations to the `.env.template` for Text2SQL support:
  - `TEXT2SQL_DB_URI`: Database URI for PostgreSQL.
    - In order for this to work you must host a separate SQL database somewhere that colab can connect to and place the URI in this environment variable.
  - `TEXT2SQL_DB_NAME`: Database name for PostgreSQL
  - `use_text2sql`: Boolean flag to enable Text2SQL functionality.
  - `text2sql_prompt`: LLM prompt for generating SQL queries from natural language.
- Default RAG parameters were adjusted to disable provenance tracking when Text2SQL is used.
    - This was done seeing that Text2SQL has some inherent provenance tracking by providing the user with the SQL query that was executed.
    - Of course in use cases where the SQL statements retrieve many rows of data provenance tracking would be required. However, for this assignment this was out of scope seeing that we had time and resource constraints.

---

### 2. **PostgreSQL Text2SQL Retriever**
A new class, `PostgresText2SQLRetriever`, is introduced. It:
- Uses a pretrained LLM to generate SQL queries from user prompts based on the database schema.
- Connects to a PostgreSQL database to execute the queries and fetch results.
- Includes methods to:
  - Load CSV data into database tables.
    - Currently only CSV files are supported for this task. You must make sure that these CSV files contain tabular/relational data such that it can be loaded into a relational database.
  - Automatically generate and validate the database schema.
    - The schema is dynamically generated based on the tables that reside in the database. This is used in order to provide the LLM with the schema such that it can generate valid SQL statements. 
  - Extract the SQL statement from the LLM response using regex
  - Validate the SQL statement using SQLParse library
  - Retrieve data and format it into the RAG framework's document format.

Within this file two other classes are declared but not used. Those include:
- `SQLQuery`, which inherits from Pydantic's base model. We wanted to implement function calling with the LLM such that we would provide this data model in the prompt and then validate the generated query with this class. However, we did not have enough time to implement this.
- `SQLGenerator`, a class that generates SQL queries using a T5 model. It initializes a model and tokenizer from huggingface, and provides a method to generate SQL queries from input prompts. However, the performance of this model was very poor, therefore we opted for a Text2SQL retrieval component that makes use of the LLM already built in the RAG framework. 


**File:** `server/PostgresText2SQLRetriever.py`

---

### 3. **RAGHelperSQL Class**
A new helper class, `RAGHelperSQL`, extends the RAG framework to incorporate SQL-based retrieval:
- Initializes the Text2SQL retriever described above and overwrites the ```load_data()``` method such that CSV files are loaded in from the specified data directory.
- Inherits all methods from the RAGHelperLocal class. However, only inherits the constructor of the RAGHelper base class.
- Creates a new HuggingFace pipeline containing the LLM specifically for the Text2SQL component which differs as `return_full_text` is set to `False`

**File:** `server/RAGHelper_sql.py`

---

### 4. **Server Integration**
- Updated the server initialization (`server.py`) to instantiate `RAGHelperSQL` if `use_text2sql` is enabled.
- Adjusted routes to process and format SQL query results for chat responses.

---

### 5. **Prompt and Query Generation**
- A prompt template is defined for generating SQL queries using an LLM, with key rules:
  - Use PostgreSQL dialect.
  - Output SQL only, enclosed in triple backticks (\`\`\`).
  - Rely strictly on the database schema provided.
  - Return only the SQL statement, with no explanation, comments, or extra text.
    - Strangely, it sometime still fails to do this properly and provides more text than necessary. 
- SQL validation ensures the generated queries are executable before retrieval.

---

### 6. **Dependencies**
Added new dependencies in `requirements.txt`:
- `sqlparse`: For SQL query parsing and validation.

---

## Usage

### Environment Setup
1. Configure `.env` with the following:
   ```env
   TEXT2SQL_DB_URI=postgresql://user:password@host:port/dbname
   use_text2sql=True
   text2sql_prompt="Your SQL generation prompt here."
2. Ensure the PostgreSQL database is running and accessible.

## Running the Server
1. Start the server with: ```python server.py```
2. Place the ngrok url in ```application.conf```'s environment variable ```server_url```
3. Navigate to ```localhost:9000``` in your browser.

## Example

The following image shows a screenshot of a question asked to the RAG framework. 

![Example](example_question.png)

In this image you can see that with our component the document the assistant refers to is a row retrieved from the database, which in this case is the count of rows in the games table. The provenance provided in this answer is the SQL statement that was provided by the LLM and subsequently executed. 

## Dataset

The data we used for testing this component was a kaggle dataset found [here](https://www.kaggle.com/datasets/technika148/football-database). It contains several several relational database tables that can be put into a database directly.

## Limitations & Improvements