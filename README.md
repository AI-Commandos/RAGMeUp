# RAG Me Up
RAG Me Up is a generic framework (server + UIs) that enables you do to RAG on your own dataset **easily**. Its essence is a small and lightweight server and a couple of ways to run UIs to communicate with the server (or write your own).

RAG Me Up can run on CPU but is best run on any GPU with at least 16GB of vRAM when using the default instruct model.

Combine the power of RAG with the power of fine-tuning - check out our [LLaMa2Lang repository](https://github.com/UnderstandLingBV/LLaMa2Lang) on fine-tuning LLMs which can then be used in RAG Me Up.

# Updates
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

# Configuration
RAG Me Up uses a `.env` file for configuration, see `.env.template`. The following fields can be configured:

## LLM configuration
- `llm_model` This is the main LLM (instruct or chat) model to use that you will converse with. Default is LLaMa3-8B
- `embedding_model` The model used to convert your documents' chunks into vectors that will be stored in the vector store
- `trust_remote_code` Set this to true if your LLM needs to execute remote code
- `force_cpu` When set to True, forces RAG Me Up to run fully on CPU (not recommended)

## Data configuration
- `data_directory` The directory that contains your (initial) documents to load into the vector store. Supported files: `PDF, JSON, DOCX, XSLX, PPTX`
- `file_types` Comma-separated list of file types to load
- `json_schema` If you are loading JSON, this should be the schema (using `jq_schema`). For example, use `.` for the root of a JSON object if your data contains JSON objects only and `.[0]` for the first element in each JSON array if your data contains JSON arrays with one JSON object in them
- `json_text_content` Whether or not the JSON data should be loaded as textual content or as structured content (in case of a JSON object)
- `vector_store_path` RAG Me Up caches your vector store on disk if possible to make loading a next time faster. This is the location where the vector store is stored. Remove this file to force a reload of all your documents
- `vector_store_k` The number of documents to retrieve from the vector store

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

## Document splitting configuration
- `splitter` The Langchain document splitter to use. For now, only `RecursiveCharacterTextSplitter` is supported
- `chunk_size` The chunk size to use when splitting up documents
- `chunk_overlap` The chunk overlap

# Funding
We are actively looking for funding to democratize AI and advance its applications. Contact us at info@commandos.ai if you want to invest.
