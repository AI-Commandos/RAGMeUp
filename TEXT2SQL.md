# Text2SQL component in RAGMeUp

## Overview
This project introduces a **Text2SQL retrieval component** into an existing Retrieval-Augmented Generation (RAG) framework. The component allows the system to transform natural language queries into SQL queries, enabling data retrieval from structured databases instead of the retrieval of unstructured documents.

The implementation integrates a PostgreSQL database for Text2SQL processing and updates various components of the RAG framework to support SQL-based retrieval seamlessly.

## How our process differs:

### 1. **Retrieval type**
- **Original RAGMeUp**: 
  - Focuses on unstructured document retrieval using a hybrid method combining dense (vector-based) and sparse (BM25) retrieval mechanisms.
- **Text2SQL component**: 
  - Adds a new retrieval mode, enabling queries against structured databases using SQL. This changes the setting from document embeddings to relational data.

---

### 2. **Database integration**
- **Original RAGMeUp**:
  - Utilizes vector databases like Milvus or PostgreSQL with `pgvector` and `pg_search` for hybrid retrieval.
- **Text2SQL component**:
  - Incorporates PostgreSQL directly for executing SQL queries. This involves dynamically generating and validating database schemas and loading CSV-based relational data into PostgreSQL tables.

---

### 3. **Provenance attribution**
- **Original RAGMeUp**:
  - Implements several provenance methods for unstructured document sources.
- **Text2SQL component**:
  - Relies on inherent provenance tracking provided by the SQL query itself, which is returned alongside the result. While this is okay for situations in which a few rows are returned, queries that return a lot of records would still need to use provenance. However, currently the maximum number of records returned by the retriever is 50. Note that each record corresponds to a retrieved document. Therefore making provenance attribution possible on the records for future work. 

---

### 4. **LLM integration**
- **Original RAGMeUp**:
  - Uses LLMs for answering user's questions based on the query and the documents (context) retrieved by the hybrid retriever, incorporating both dense and sparse retrieval. 
- **Text2SQL component**:
  - Adapts the LLM in order to additionally generate SQL queries with the LLM based on the user's query and the database schema, replacing the hybrid retrieval mechanism.

---

### 5. **Pipeline changes**
- **Original RAGMeUp**:
  - Features a sequential pipeline for chunking, embedding generation, retrieval, and response generation specifically for document-based (unstructured) queries.
- **Text2SQL component**:
  - Adds a separate route for SQL query generation, validation, and execution. Afterwards the retrieved database records are injected into the prompt. Just like the documents in the original RAGMeUp

### 6. **Data handling**
- **Original RAGMeUp**:
  - Processes and chunks unstructured data into embeddings stored in a vector database.
- **Text2SQL component**:
  - Skips chunking and embedding; instead, loads relational data directly into a database for SQL-based retrieval. It is important to note that currently only CSV files are supported and they must be put in the folder that is specified in the `.env` in the `data_directory` environment variable.

---

## Key Changes and features added to the code base.

### 1. **Environment configuration**
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

### 2. **PostgreSQL Text2SQL retriever**
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

### 3. **RAGHelperSQL class**
A new helper class, `RAGHelperSQL`, extends the RAG framework to incorporate SQL-based retrieval:
- Initializes the Text2SQL retriever described above and overwrites the ```load_data()``` method such that CSV files are loaded in from the specified data directory.
- Inherits all methods from the RAGHelperLocal class. However, only inherits the constructor of the RAGHelper base class.
- Creates a new HuggingFace pipeline containing the LLM specifically for the Text2SQL component which differs as `return_full_text` is set to `False`

**File:** `server/RAGHelper_sql.py`

---

### 4. **Server**
- Updated the server initialization (`server.py`) to instantiate `RAGHelperSQL` if `use_text2sql` is enabled.
- Adjusted routes to process and format SQL query results for chat responses.

---

### 5. **Prompt and query generation**
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

### Environment setup
1. Configure `.env` with the following:
   ```env
   TEXT2SQL_DB_URI=postgresql://user:password@host:port/dbname
   use_text2sql=True
   text2sql_prompt="Your SQL generation prompt here."
2. Ensure the PostgreSQL database is running and accessible.
    - We hosted the postgres database as a docker container using this exact command:
        - ```sudo docker run -d --name postgres -e POSTGRES_USER=<user> -e POSTGRES_PASSWORD=<pass> -e POSTGRES_DB=text2sql -p 5432:5432 postgres```
    - Then we hosted it for free on ngrok with:
        - ```ngrok tcp 5432```

## Running the server
1. Start the server with: ```python server.py```
2. Place the ngrok url in ```application.conf```'s environment variable ```server_url```
3. Navigate to ```localhost:9000``` in your browser.

## Example

The following image shows a screenshot of a question asked to the RAG framework. 

![Example](example_question.png)

In this image you can see that with our component the document the assistant refers to is a row retrieved from the database, which in this case is the count of rows in the games table. The provenance provided in this answer is the SQL statement that was provided by the LLM and subsequently executed. As you can see the answer could have been formatted differently as the SQL statement is now referred to as document 0, however we simply passed the SQL statement and answer through the same template as the original RAGMeUp framework does.

## Dataset

The data we used for testing this component was a kaggle dataset found [here](https://www.kaggle.com/datasets/technika148/football-database). It contains several several relational database tables related to football that can be put into a database directly. Users can ask some interesting questions on football statistics which can be answered by the dataset. So in order for us to have run this RAGMeUp framework with this component we had uploaded the csv files of this dataset to our Google Drive, subsequently mounting the drive to colab and setting the ```data_directory``` environment variable to the correct path.

## Findings

Please note that due to the inherent nature of the Text2SQL retrieval component, there are no serialized database chunks to upload. Our implementation just loads in relational data into a postgres database in order to subsequently execute LLM generated SQL statements on. Therefore we also did not use milvus but resorted to a simple PostgreSQL database. 

Before we started using the already available LLM for the Text2SQL functionality, we used a T5 model on HuggingFace specifically fine-tuned for Text2SQL, however, the performance of this model was very poor. Even when providing the model with the correct database schema it would still hallucinate and create SQL statements with non-existent tables and fields. Therefore we opted for the LLM in the current implementation.

Additionally, we wanted to create a prompt in which we use a technique called function calling. Some LLM's support of explicit function calling where the model can choose a function, pass arguments, and interpret structured responses. However, the LLM we used LLama does not support this natively. Therefore it can be adapted with custom prompt engineering. We wanted to try and do this by passing a string representation of a Pydantic base model of what a valid SQL query would look like, hoping to elicit a response structured like this Pydantic data model. Afterwards we would also be able to then validate the response of the LLM with Pydantic. Due to time constraints we did not implement this fully. However, with specific prompt engineering tailored to our task in combination with an SQLParser from the sqlparse library, the returned SQL statements are pretty solid. 

Due to a mistake in the ```RAGHelperLocal._create_llm_chain()``` the retriever is called twice. In the original hybrid retrieval method this is not really a problem as it is fast and deterministic, however in our case the retriever component uses an LLM, which is relatively slow, and inherently non-determenistic. Also, in some cases things might go wrong when executing SQL statements, seeing that there will be two concurrent calls using the same class and the same database connection and cursor. In order to fix this the declaration of the LLM chain must be changed such that our retriever is invoked before passing it to the langchain, after this the returned docs/rows should be passed to the chain. However, to make this work was not such a straightforward task. Therefore we were not able to fully implement it.