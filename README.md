# GraphRAG: Extending RAG Me Up with Neo4j Graph Integration

- Neo4j integration with RAG Me Up to store and retrieve data via graph queries.
- The RagHelperCloud configuration orchestrates retrieval and LLM processing.
- Tested using Neo4j Desktop & GEMINI.

Key additions:

1. Separate Neo4j server with REST endpoints
2. Graph-based CSV and PDF loaders
3. Graph-based retrieval

# How GraphRAG Works

### Graph-Based Retrieval

The `graph_retriever` function queries a Neo4j database with **schema-aware Cypher**. Key points:

1. **Neo4j Schema Integration**

   - Dynamically retrieves the schema via `/schema`.
   - Formats it into a prompt-friendly representation.

2. **LLM-Driven Query Generation**

   - Combines schema info + user queries to form schema-aware prompts.
   - Uses few-shot learning to guide Cypher generation.

3. **Query Execution & Data Retrieval**

   - Executes LLM-generated queries via `/query`.
   - Converts results into LangChain `Document` objects (with metadata).

4. **Fallback Mechanism**

   - If no valid query applies, returns `None` to skip redundant computation.

5. **Integration with Other Retrievers**
   - Graph-based documents are prioritized as “document 0.”
   - Remaining “slots” (based on `chunk_size`) are filled by other retrievers.

### Graph-Based Document Uploading

GraphRAG provides two main functions for adding data to Neo4j:

#### 1. `add_csv_to_graphdb`

1. **Reads the CSV**

   - Uses Python’s `csv.DictReader` for parsing.

2. **Defines the Graph Schema**

   - Converts rows into Cypher queries, e.g.:
     ```cypher
     MERGE (q:Quote {text: $quoteText})
     MERGE (t:Topic {name: $topicName})
     MERGE (q)-[:IS_PART_OF]->(t)
     ```

3. **Sends Data to Neo4j**

   - Packages Cypher + parameters into JSON, sent via `/add_instances`.

4. **Logs Status**
   - Tracks uploaded records and server responses.

**Use Case Example (NPS Feedback)**

- _Quotes_: Customer responses to “Please tell us more...”
- _Topics_: Automatically derived themes (e.g., “Relationship with contact”).

#### 2. `add_document_to_graphdb`

1. **Metadata Identification**

   - Determines file type (PDF, etc.).

2. **Schema Configuration**

   - Fetches schema dynamically (`dynamic_neo4j_schema = True`) or uses a predefined `.env` schema.

3. **Triplet Extraction with LLM**

   - Combines content + schema to form Cypher prompts.
   - Few-shot examples guide the LLM’s query creation.

4. **Query Execution**
   - Escapes special chars, then sends queries as JSON via `/add_instances`.
   - Logs success/failure.

# Setup & Configuration:

1. install Neo4j Desktop & run to host a local Neo4j database

2. set environment variables:

   - ngrok authentication token in `ngrok_token`.
   - location of the neo4j desktop uri in `neo4j_location`, if you use neo4j desktop, this is: bolt://localhost:7687
   - your Neo4j username in `neo4j_user`
   - your neo4j password `neo4j_password`

3. Confugure RAGMeUp server

   - run the `neo4j server.py` file and save public ngrok url in `neo4j_location`.
   - Set `use_gemini` as True such that `RAGHelperCloud` can use Gemini as LLM.
   - `GOOGLE_API_KEY` for Gemini athentication.
   - launch `server.py`

4. OPTIONAL improvements:

- `max_document_limit` to set the maximum amount of document chunks that will be outputted by the retrieval
- `rag_retrieval_instruction`, `retrieval_few_shot`, `rag_retrieval_question` to improve the LLM prompt
- `neo4j_insert_instruction`, `neo4j_insert_schema`, `neo4j_insert_data_only`, `neo4j_insert_few_shot` to change the document upload LLM prompt
- `dynamic_neo4j_schema`, if set to True it will fetch the schema from the neo4j server, if set to False, it will use the LLM instruction of `neo4j_insert_data_only`. When you define a schema here, the LLM will use the schema. If you leave schema open, the LLM will dynamically create new nodes and relationships.
