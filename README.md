# RAG Me Up
RAG Me Up is a generic framework (server + UIs) that enables you do to RAG on your own dataset **easily**. Its essence is a small and lightweight server and a couple of ways to run UIs to communicate with the server (or write your own).

RAG Me Up can run on CPU but is best run on any GPU with at least 16GB of vRAM when using the default instruct model.

Combine the power of RAG with the power of fine-tuning - check out our [LLaMa2Lang repository](https://github.com/UnderstandLingBV/LLaMa2Lang) on fine-tuning LLMs which can then be used in RAG Me Up.

# Updates
- **2024-06-26** Updated readme, added more file types, robust self-inflection
- **2024-06-05** Upgraded to Langchain v0.2

# Installation
## Server
```
git clone https://github.com/UnderstandLingBV/RAGMeUp.git
cd server
pip install -r requirements.txt
```
Then run the server using `python server.py` from the server subfolder.

## Scala UI
Make sure you have JDK 17+. Download and install [SBT](https://www.scala-sbt.org/) and run `sbt run` from the `server/scala` directory or alternatively download the [compiled binary](https://github.com/UnderstandLingBV/RAGMeUp/releases/tag/scala-ui) and run `bin/ragemup(.bat)`

# How does RAG Me Up work?
RAG Me Up aims to provide a robust RAG pipeline that is configurable without necessarily writing any code. To achieve this, a couple of strategies are used to make sure that the user query can be accurately answered through the documents provided.

The RAG pipeline is visualized in the image below:
![RAG pipeline drawing](./ragmeup.drawio.svg)

The following steps are executed. Take note that some steps are optional and can be turned off through configuring the `.env` file.

__Top part - Indexing__
1. You collect and make your documents available to RAG Me Up.
2. Using different file type loaders, RAG Me Up will read the contents of your documents. Note that for some document types like JSON and XML, you need to specify additional configuration to tell RAG Me Up what to extract.
3. Your documents get chunked using a recursive splitter.
4. The chunks get converted into document (chunk) embeddings using an embedding model. Note that this model is usually a different one than the LLM you intend to use for chat.
5. RAG Me Up uses a hybrid search strategy, combining dense vectors in the vector database with sparse vectors using BM25. By default, RAG Me Up uses a local [Milvus database](https://milvus.io/).

__Bottom part - Inference__
1. Inference starts with a user asking a query. This query can either be an initial query or a follow-up query with an associated history and documents retrieved before. Note that both (chat history, documents) need to be passed on by a UI to properly handle follow-up querying.
2. A check is done if new documents need to be fetched, this can be due to one of two cases:
    - There is no history given in which case we always need to fetch documents
    - **[OPTIONAL]** The LLM itself will judge whether or not the question - in isolation - is phrased in such a way that new documents are fetched or whether it is a follow-up question on existing documents. A flag called `fetch_new_documents` is set to indicate whether or not new documents need to be fetched.
3. Documents are fetched from both the vector database (dense) and the BM25 index (sparse). Only executed if `fetch_new_documents` is set.
4. **[OPTIONAL]** Reranking is applied to extract the most relevant documents returned by the previous step. Only executed if `fetch_new_documents` is set.
5. **[OPTIONAL]** The LLM is asked to judge whether or not the documents retrieved contain an accurate answer to the user's query. Only executed if `fetch_new_documents` is set.
    - If this is not the case, the LLM is used to rewrite the query with the instruction to optimize for distance based similarity search. This is then fed back into step 3. **but only once** to avoid lengthy or infinite loops.
6. The documents are injected into the prompt with the user query. The documents can come from:
    - The retrieval and reranking of the document databases, if `fetch_new_documents` is set.
    - The history passed on with the initial user query, if `fetch_new_documents` is **not** set.
7. The LLM is asked to answer the query with the given chat history and documents.
8. The answer, chat history and documents are returned.

# Configuration
RAG Me Up uses a `.env` file for configuration, see `.env.template`. The following fields can be configured:

## LLM configuration
- `llm_model` This is the main LLM (instruct or chat) model to use that you will converse with. Default is LLaMa3-8B
- `llm_eos_token` Not all (finetuned/QLoRA) LLMs always use the same EOS token as their parent (for example with LLaMa3 finetunes). Set this variable to be the proper EOS token or set to `None` to use the model tokenizer's.
- `embedding_model` The model used to convert your documents' chunks into vectors that will be stored in the vector store
- `trust_remote_code` Set this to true if your LLM needs to execute remote code
- `force_cpu` When set to True, forces RAG Me Up to run fully on CPU (not recommended)

### Use OpenAI
If you want to use OpenAI as LLM backend, make sure to set `use_openai` to True and make sure you (externally) set the environment variable `OPENAI_API_KEY` to be your OpenAI API Key.

## Data configuration
- `data_directory` The directory that contains your (initial) documents to load into the vector store
- `file_types` Comma-separated list of file types to load. Supported file types: `PDF, JSON, DOCX, XSLX, PPTX, CSV, XML`
- `json_schema` If you are loading JSON, this should be the schema (using `jq_schema`). For example, use `.` for the root of a JSON object if your data contains JSON objects only and `.[0]` for the first element in each JSON array if your data contains JSON arrays with one JSON object in them
- `json_text_content` Whether or not the JSON data should be loaded as textual content or as structured content (in case of a JSON object)
- `xml_xpath` If you are loading XML, this should be the XPath of the documents to load (the tags that contain your text)

## Retrieval configuration
- `vector_store_path` RAG Me Up caches your vector store on disk if possible to make loading a next time faster. This is the location where the vector store is stored. Remove this file to force a reload of all your documents
- `vector_store_k` The number of documents to retrieve from the vector store
- `rerank` Set to either True or False to enable reranking
- `rerank_k` The number of documents to keep after reranking. Note that if you use reranking, this should be your final target for `k` and `vector_store_k` should be set (significantly) higher. For example, set `vector_store_k` to 10 and `rerank_k` to 3

## LLM parameters
- `temperature` The chat LLM's temperature. Increase this to create more diverse answers
- `repetition_penalty` The penalty for repeating outputs in the chat answers. Some models are very sensitive to this parameter and need a value bigger than 1.0 (penalty) while others benefit from inversing it (lower than 1.0)
- `max_new_tokens` This caps how much tokens the LLM can generate in its answer. More tokens means slower throughput and more memory usage

## Prompt configuration
- `rag_instruction` An instruction message for the LLM to let it know what to do. Should include a mentioning of it performing RAG and that documents will be given as input context to generate the answer from.
- `rag_question_initial` The initial question prompt that will be given to the LLM only for the first question a user asks, that is, without chat history
- `rag_question_followup` This is a follow-up question the user is asking. While the context resulting from the prompt will be populated by RAG from the vector store, if chat history is present, this prompt will be used instead of `rag_question_initial`
- `rag_fetch_new_instruction` RAG Me Up automatically determines whether or not new documents should be fetched from the vector store or whether the user is asking a follow-up question on the already fetched documents by leveraging the same LLM that is used for chat. This environment variable determines the prompt to use to make this decision. Be very sure to instruct your LLM to answer with yes or no only and make sure your LLM is capable enough to follow this instruction
- `rag_fetch_new_question` The question prompt used in conjunction with `rag_fetch_new_instruction` to decide if new documents should be fetched or not
- `user_rewrite_loop` Set to either True or False to enable the rewriting of the initial query. Note that a rewrite will always occur at most once
- `rewrite_query_instruction` This is the instruction of the prompt that is used to ask the LLM to judge whether a rewrite is necessary or not. Make sure you force the LLM to answer with yes or no only
- `rewrite_query_question` This is the actual query part of the prompt that isued to ask the LLM to judge a rewrite
- `rewrite_query_prompt` If the rewrite loop is on and the LLM judges a rewrite is required, this is the instruction with question asked to the LLM to rewrite the user query into a phrasing more optimized for RAG. Make sure to instruct your model adequately.

## Document splitting configuration
- `splitter` The Langchain document splitter to use. For now, only `RecursiveCharacterTextSplitter` is supported
- `chunk_size` The chunk size to use when splitting up documents
- `chunk_overlap` The chunk overlap

# Funding
We are actively looking for funding to democratize AI and advance its applications. Contact us at info@commandos.ai if you want to invest.
