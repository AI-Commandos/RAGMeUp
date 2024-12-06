# GraphRAG: Extending RAG Me Up with Neo4j Graph Integration

This project introduces **GraphRAG**, an extension to the [RAG Me Up framework](https://github.com/AI-Commandos/RAGMeUp), by integrating Neo4j as a graph database into the RAG pipeline. GraphRAG leverages graph-based data storage and querying capabilities, enabling improved data relationships and enhanced retrieval.

GraphRAG is created for an assignment of the Natural Language Processing course of JADS by **Tjielke Nabuurs, Jeroen Dortmans and Luuk Jacobs** Our main goal of the assignment was to modify the components of the RAG pipeline that is used by default by the RAG Me Up repository.

# Motivation

Our academic interest lies in graph-based retrieval systems given how graph structures can represent complex relationships and interconnections in a way that mirrors human logic and knowledge. Graph databases like Neo4j are not only highly intuitive for modeling complex systems but also incredibly powerful for performing traceable, structured queries. By using graphs, we can represent the inherent relationships in data and make use of their attributes, enriching the data upon retrieval.

Especially in production environments, reliability and interpretability are paramount. GraphRAG's graph-based approach offers:

1. **Enrichment of Retrieved Data**: GraphRAG not only retrieves the target information but also leverages the graph structure to enrich it with additional relevant attributes.
2. **Traceability**: Each retrieval result is directly linked to its origin in the graph, making it easy to verify and understand why specific information was selected.
3. **Logical Consistency**: The graph structure enforces relationships and dependencies, ensuring that results adhere to the logical connections defined within the data. Which ensure more reliable responses and reduce hallucinations.

# Overview of Modifications

To additionally store graphs in the RAG pipeline, we use Neo4j for database management given it is the current industry standard. To offer a loose coupling between the existing server and the neo4j database, we create a seperate server that connects to the database, which is accessible to the server using a REST API. The implementation only uses the RagHelperCloud configuration for enabling the retrieval of relevant documents and combining them for LLM processing. The implementation is tested using the GEMINI API as its LLM.
We modified the RAG Me Up framework to include a Neo4j-based graph integration in the following ways:

1. **Seperate REST API accessible server for database connection** This architecture introduces modularity and scalability by decoupling the database logic from the main application. Which also provides ease of integration for future use-cases.
2. **Added a graph-based pdf loader**, This uses the LLM to extract triplets from the data, which are inputted into the database. The schema can be defined beforehand in the .env or dynamically retrieved from the database
3. **Implemented a graph-based csv loaders** to import csv's into the database, a seperate csv parser and .. .
4. **Implemented Graph based retrieval** where we create cypher queries by combining schema information and the user question. Furthermore, using few shot prompting technique, the LLM will return "None" when the question cannot be relevantly answered using a Cypher query.

We have created an end-to-end functioning graphdb implementation of RAG, which can upload both csv and pdf files.

### Implementation Details: Graph-Based Retrieval

In this project, we extended the default RAG pipeline by integrating Neo4j for graph-based retrieval. A new retriever function, `graph_retriever`, was developed to enable querying a Neo4j database using schema-aware Cypher queries. This integration enhances the retrieval process by leveraging graph-based relationships and attributes. Below are the key aspects of this implementation:

1. **Neo4j Schema Integration**

   - Dynamically retrieves the database schema using the `/schema` endpoint.
   - Formats the schema into a prompt-friendly text representation for the LLM.

2. **LLM-Driven Query Generation**

   - Combines schema information and user queries to create schema-aware prompts.
   - Employs a few-shot learning approach to guide the LLM in generating Cypher queries.

3. **Query Execution and Data Retrieval**

   - Executes the LLM-generated Cypher query against the Neo4j database using the `/query` endpoint.
   - Converts query results into LangChain `Document` object, enriched with metadata for downstream processing.

4. **Fallback Mechanism**

   - When the LLM determines a query cannot be answered using the schema, it returns `None` to avoid unnecessary computations or invalid results.

5. **Integration with Other Retrievers**

   - Prioritizes graph-based results from the `graph_retriever` as the most enriched and reliable source of information, setting it as document 0.
   - Dynamically calculates available space using 'chunk_size' after processing graph documents, filling the remaining capacity with documents from the other retrievers.

### Implementation details: Graph-based document uploading

GraphRAG introduces two primary functions to handle document uploading into the Neo4j graph database:

#### 1. `add_csv_to_graphdb`

This function enables the uploading of CSV data into the Neo4j graph database. Here's how it works:

1. **Reads the CSV File:**

   - Opens the CSV file from the specified path and parses its content using Python's `csv.DictReader`.

2. **Defines the Graph Schema:**

   - The function uses a predefined schema to convert each row of the CSV into Cypher queries. For instance:

```cypher
MERGE (q:Quote {text: $quoteText})
MERGE (t:Topic {name: $topicName})
MERGE (q)-[:IS_PART_OF]->(t)
```

3. **Sends Data to Neo4j:**

   - Each row is transformed into a JSON payload containing the Cypher query and its parameters. These payloads are sent to the Neo4j server via the `/add_instances` REST API endpoint.

4. **Logs Status:**

   - Logs all significant steps, including the number of records successfully uploaded or any server responses.
  
#### Reasoning for specific use-case of topics and quotes

The use-case of GraphRAG is developed on a dataset that consists of customer feedback data, collected through an NPS (Net Promoter Score) survey. It aims to use detected topics in the customer feedback to pinpoint recurring issues. It includes the following elements:

1. *Quote of the Feedback*:  
   Customers’ responses to the follow-up question:  
   "Please tell us more about why you wouldn’t recommend [Organization]."  
   These responses provide qualitative insights into customer satisfaction and dissatisfaction.

2. *Topics Detected in the Quote*:  
   Automatically identified themes or topics in the feedback, such as 'Relieving of concerns', or 'Relationship with your point of contact'. Topics are derived using natural language processing (NLP) and help cluster feedback for actionable analysis.

#### 2. `add_document_to_graphdb`

This function processes non-CSV documents (e.g., PDFs) and uploads their extracted data to Neo4j:

1. **Metadata Identification:**

   - Determines the file type and metadata (e.g., file source).

2. **Schema Configuration:**

- If `dynamic_neo4j_schema` is set to `True`, the schema is dynamically fetched from the Neo4j database using the `/schema` REST API endpoint. The fetched schema is then formatted into a prompt-friendly text structure.
- Otherwise, a predefined schema in the `.env` file is used.

4. **Triplet Extraction Using LLM:**

- Combines the schema, the document content, and predefined instructions to generate a Cypher query prompt using a language model (e.g., GEMINI).
- The prompt includes instructions and few-shot examples to guide the LLM in creating schema-aware Cypher queries.
- **Query Execution:**

  - Executes the Cypher queries generated by the LLM after escaping special characters (like curly braces) to ensure compatibility.
  - The queries are parsed into JSON payloads, sent to Neo4j via the `/add_instances` endpoint, and logged for success or failure.

# Setup & Configuration:

We have developed GraphRAG with a local Neo4j database that is hosted through Neo4j desktop.

step 1. Therefore, to setup such Neo4j database connection one must download Neo4j desktop and run it to host Neo4j locally.

step 2. set your environment variables for running the neo4j server.py. You have to:
   - define the ngrok authentication token in the variable `ngrok_token`.
   - define the location of the neo4j desktop uri, if you use neo4j desktop, this is: bolt://localhost:7687
   - your Neo4j username
   - your neo4j password

step 3. run the neo4j server.py file and save the ngrok tunnel address. 

step 4. Set your environment variables for the RAGMeUp server
   - Your public ngrok url should be stored in the variable `neo4j_location`.
   - Define `use_gemini` as True such that the `RAGHelperCloud` class can use Gemini as LLM.
   - Adapt `GOOGLE_API_KEY` for your Gemini athentication.

step 5. Run server.py, we run this in the Ubuntu WSL. 

step 6. OPTIONAL: Possible changes you can make to the environment variables to improve your results:
   - max_document_limit, this is the maximum amount of document chunks that will be outputted by the retrieval
   - rag_retrieval_instruction, retrieval_few_shot, rag_retrieval_question to improve the llm prompt 
   - neo4j_insert_instruction, neo4j_insert_schema, neo4j_insert_data_only, neo4j_insert_few_shot to change the document upload llm prompt
   - dynamic_neo4j_schema, if set to True it will fetch the schema from the neo4j server, if set to False, it will use the llm instruction of neo4j_insert_data_only. When you define a schema here, the llm will use the schema. If you leave schema open, the llm will dynamically create new nodes and relationships. 
