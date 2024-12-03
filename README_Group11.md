# Group 11 changes to RAGMeUp

## Idea
There were a lot of possibilities for changing or expanding the RAGMeUp framework for this assignment, eventually we decided to implement ColBERT as reranker. Implementing ColBERT offers the chance to work with a state-of-the-art retrieval framework that uses deep learning to enhance semantic understanding. By leveraging BERT’s contextual embeddings, ColBERT allows the system to retrieve passages not only based on exact keyword matches but on deeper semantic relevance, which is particularly useful when dealing with ambiguous or context-sensitive queries. This aligns well with projects like chatbot applications or question-answering systems where precision and context matter.

Another compelling reason for implementing ColBERT is the efficiency in managing large datasets. Unlike traditional dense retrieval models, ColBERT uses late interaction techniques that balance computational costs and accuracy, making it scalable for real-world applications. This means we are able to explore how such a model can bridge the gap between research and practical deployment, addressing challenges like latency and user experience in information-heavy applications.

Finally, the project provides an opportunity for hands-on learning with advanced concepts in machine learning and information retrieval, including transformer-based models, indexing, and clustering. Working with ColBERT deepens our understanding of how neural networks can be optimized for retrieval tasks, offering valuable insights into AI techniques. This means we can explore how these systems can be applied to real-world problems, creating a more engaging, powerful, and responsive chatbot, while also building technical expertise that’s highly relevant nowadays

## Implementation

In order to implement ColBERT as reranker, a couple of functions and files had to be added/changed.

However, first we managed to connect a data folder containing all pdf's used in the course. This data folder is located in our own google drive and navigated to within the google collab file. Locally, in the 'env.template' file we set the following value: "data_directory='data/drive/MyDrive/NLP'". This results in these pdf's being used in the RAG framework so that information from these pdf's can be retrieved and returned when a user asks for it in the chat environment. We got this to work in the base model and now the implementation of ColBERT could start.

These were the steps that were taken.

1. Within the ".env.template" file the rerank model 'flashrank' was replaced with the 'colbert' model with the following deletion and addition of lines:

   **Deletion:**
   ```ruby
   rerank_k=3
   rerank_model=flashrank
   ```
   
   **Addition:**
   ```ruby
   rerank_model=colbert
   colbert_model=colbert-ir/colbertv2.0  # or path to your custom model
   colbert_nbits=2  # quantization bits
   colbert_doc_maxlen=180  # max document length
   colbert_query_maxlen=32  # max query length
   rerank_k=5  # number of documents to return after reranking
   ```
3. The file "ColBERTReranker.py" is added. This file contains the entire ColBERTReranker. It leverages ColBERT for reranking documents based on their relevance to a given query. The ColBERTReranker is a custom class that inherits from the BaseDocumentCompressor, aligning it with the LangChain document processing framework. 

    ```ruby
    class ColBERTReranker(BaseDocumentCompressor):
        """
        A document compressor that uses ColBERT for reranking documents.
        """
    ```
The core component where the reranking of documents occurs in the "compress_documents" method. It handles empty documents, prepares the documents for reranking, initializes the ColBERT model, and returns the "top_n" documents. It loads a pre-trained "colbertv2.0" which has a checkpoint trained on the MS MARCO Passage Ranking task.

    ```ruby
    colbert = RAGPretrainedModel.from_pretrained(self.model)
    colbert.index(
        index_name="/content/RAGMeUp/server/indexed",
        collection=doc_texts,
        document_ids=doc_ids,
        document_metadatas=[doc.metadata for doc in documents],
    )
    results = colbert.search(query)

    ```

After initializing the model, indexing is required to organize the documents into a structure optimized for fast retrieval. Without indexing, the system would need to perform a linear search across all documents, which is computationally expensive and impractical for large datasets. Note that they are stored in the denoted directory. Afterward, the searcher is used on the indexed structure to quickly find and rank documents relevant to a given query.


4. The file "RAGHelper.py" contains useful changes, but we have to note that a lot of changes are given since all single apostrophes are changed to doubles and code that was in one line is put under each other. This is done in the automatic formatting of vscode. These changes do not add value but are indicated by Git Hub, be aware of these. The meaningful change resides in the 'initialize reranker' function. Here the added code is indicated in the comments:
   
   ```ruby
   def _initialize_reranker(self):
        """Initialize the reranking model based on environment settings."""
        if self.rerank_model == "flashrank":
            self.logger.info("Setting up the FlashrankRerank.")
            self.compressor = FlashrankRerank(top_n=self.rerank_k)

        # This code adds the colbert rerank if it is available
        elif self.rerank_model == "colbert":
            self.logger.info("Setting up the ColBERT reranker.")
            self.compressor = ColBERTReranker(
                model_name=self.colbert_model,
                top_n=self.rerank_k,
                nbits=self.colbert_nbits,
                doc_maxlen=self.colbert_doc_maxlen,
                query_maxlen=self.colbert_query_maxlen,
            )
        # Until here.
   
        else:
            self.logger.info("Setting up the ScoredCrossEncoderReranker.")
            self.compressor = ScoredCrossEncoderReranker(
                model=HuggingFaceCrossEncoder(model_name=self.rerank_model),
                top_n=self.rerank_k
                top_n=self.rerank_k,
            )
   ```

5. In "RAGHelper_local.py" the following code is added:
   ```ruby
        if os.getenv('rerank_model') == 'flashrank':
            return [d.metadata['relevance_score'] for d in reranked_docs if
                d.page_content in [doc.page_content for doc in reply['docs']]]

        else:
            return [d['metadata']['relevance_score'] for d in reranked_docs if
                d['page_content'] in [doc.page_content for doc in reply['docs']]]
   ```

6. The "requirements.txt" should also be updated since there are new libraries required while using ColBERT:

   ```ruby
   colbert-ir==0.2.14  # Added for ColBERT reranking
   fsspec==2024.9.0
   ragatouille
   ```
**Limitations** \
- Although the ColBERTReranker should succesfully obtain more insightful responses in a faster time, due to it support for reranking with late interaction; it's current performance and running times do not reflect this behavior.
- The setting "use_rewrite_loop" to True results in better responses, but there seems to be a randomness issue where occasionally the response cannot be visualized in the user interface. Even though the response can be generated in the Google Colab "server.py" output, the team has attempted to fix this issue, but is not proficient with Scala and struggled to find a solution. 


These were the most prevalent steps for replacing FlashRank with ColBERT.

For implementing further changes take a look at: https://github.com/AI-Commandos/RAGMeUp/compare/main...LvR33:RAGMeUp:ElanoIter
