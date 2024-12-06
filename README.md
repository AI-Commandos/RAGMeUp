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

5. **Integration with other retrievers**
   - as the graph_retriever offers the most enriched data source, when it creates results, it seen as most important and based on the chunk_size (kijk hiervoor naar de code om beter uit te leggen)

### Implementation details: Graph-based document uploading

two new function: add_csv_to_graphdb, add_document_to_graphdb,

first add_csv_to_graphdb uitleggen

add_document_to_graphdb uitleggen, leg hierbij ook uit dat je met een environment variable kan instellen of je de schema wil definiëren in de .env file of dynamisch wil krijgen uit de database.

# Setup & Configuration:

We have developed GraphRAG with a local Neo4j database that is hosted through Neo4j desktop.

Therefore to initialize Neo4j database connection

Download Neo4j desktop and run to host Neo4j locally

run neo4j server.py locally. `Graph_whisperer` class thereafter initilizes a Neo4j database connection. in this file is also the ngrok authentication token defined. Ngrok is used to expose the neo4j server. the public url is stored in the `.env` file in the variable `neo4j_location`

set gemini = true in .env to make use of Gemini as LLM, API key needed

By our preference we run RAG Me Up server.py in wsl.
