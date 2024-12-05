# Group 11 changes to RAGMeUp

## Idea
There were a lot of possibilities for changing or expanding the RAGMeUp framework for this assignment, eventually we decided to implement ColBERT as a reranker. Implementing ColBERT offers the chance to work with a state-of-the-art retrieval framework that uses deep learning to enhance semantic understanding. By leveraging BERT’s contextual embeddings, ColBERT allows the system to retrieve passages not only based on exact keyword matches but on deeper semantic relevance, which is particularly useful when dealing with ambiguous or context-sensitive queries. This aligns well with projects like chatbot applications or question-answering systems where precision and context matter.

Another compelling reason for implementing ColBERT is the efficiency in managing large datasets. Unlike traditional dense retrieval models, ColBERT uses late interaction techniques that balance computational costs and accuracy, among other benefits, making it scalable for real-world applications. This means we are able to explore how such a model can bridge the gap between research and practical deployment, addressing challenges like latency and user experience in information-heavy applications.

Finally, the project provides an opportunity for hands-on learning with advanced concepts in machine learning and information retrieval, including transformer-based models, indexing, and clustering. Working with ColBERT deepens our understanding of how neural networks can be optimized for retrieval tasks, offering valuable insights into AI techniques. This means we can explore how these systems can be applied to real-world problems, creating a more engaging, powerful, and responsive chatbot, while also building technical expertise that’s highly relevant nowadays

Therefore, including this reranker model in the RAG Framework should not only support more efficient reranking, which can lead to more documents which are able to be added to each answer given the higher efficiency. Furthermore, in the inclusion of this model, the answers of the model should, as described, be a better fit for those queries which are ambiguous or need contextual information.

## Implementation

In order to implement ColBERT as a reranker, a couple of functions and files had to be added/changed.

However, first we managed to connect a data folder containing all pdf's used in the course. This data folder is located in our own Google Drive and navigated to within the Google Collab file. Locally, in the 'env.template' file we set the following value: "data_directory='data/drive/MyDrive/NLP'". To be able to set this up, the homepage of the Google Drive should incorporate this NLP folder and the drive should be mounted on the following directory: `/content/RAGMeUp/server/data/drive`. This results in these pdf's being used in the RAG framework so that information from these pdf's can be retrieved and returned when a user asks for it in the chat environment. We got this to work in the base model and now the implementation of ColBERT could start.

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

    These variables were added to be able to set some specifics in the training of the model. The variables which can be seen are `rerank_model` which was changed to select the colbert model, `rerank_k` was added to increase the number of documents used in each answer, `colbert_model` is a variable containing the path of the relevant model on huggingface. For our case, the colbert model is found [here](https://huggingface.co/colbert-ir/colbertv2.0). The `colbert_nbits` variable is used to specify the the number of bits used to save the weights of the model. In this case, the baseline was chosen and the weights are stored in 2-bit. Finally, to make sure that the model does train on parts which are not relevant, the maximum token lengths for the documents and the queries were set. For this case, the maximum length of any possible query was set to 32 tokens and the maximum length of the documents is set to 180 tokens.
   
2. The file "ColBERTReranker.py" is added. This file contains the entire ColBERTReranker. It leverages ColBERT for reranking documents based on their relevance to a given query. The ColBERTReranker is a custom class that inherits from the BaseDocumentCompressor, aligning it with the LangChain document processing framework. 

    ```ruby
    class ColBERTReranker(BaseDocumentCompressor):
        """
        A document compressor that uses ColBERT for reranking documents.
        """
    ```
    The core component where the reranking of documents occurs is in the "compress_documents" method. It handles empty documents, prepares the documents for reranking, initializes the ColBERT model, and returns the "top_n" documents. It loads a pre-trained "colbertv2.0" which has a checkpoint trained on the MS MARCO Passage Ranking task. A pretrained model was used due to the limited size of the new training data. If there was more data, it would likely have been better if the model would be finetuned to fit the given task with the corresponding jargon in the field of NLP. However, this was not feasible due to the rlative low amounts of data, leading to a worse overall performance of the model.

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


3. The file "RAGHelper.py" contains useful changes, but most of the changes in this file was performed by the automatic formatting of vscode. These changes do not add value but are indicated by Git Hub, be aware of these. The meaningful change resides in the 'initialize reranker' function. Here the added code is indicated in the comments:
   
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

   This check was set up which reranking model should be selected. It performs this based on the rerank_model variable defined in the .env file. Other changes in this file were based on the change in the formatting of the model outputs, where ColBERT returns information in dictionary classes and the original flashrank returns Document classes:
   
   **Addition:**
   ```ruby
       def format_documents(docs):
        """
        Formats the documents for better readability.

        Args:
            docs (list): List of Document objects.

        Returns:
            str: Formatted string representation of documents.
        """
        doc_strings = []
        for i, doc in enumerate(docs):
            if type(doc) == dict:
              metadata_string = ", ".join(
                  [f"{md}: {doc['metadata'][md]}" for md in doc['metadata'].keys()]
              )
              doc_strings.append(
                  f"Document {i} content: {doc['page_content']}\nDocument {i} metadata: {metadata_string}"
              )

            else:
              metadata_string = ", ".join(
                  [f"{md}: {doc.metadata[md]}" for md in doc.metadata.keys()]
              )
              doc_strings.append(
                  f"Document {i} content: {doc.page_content}\nDocument {i} metadata: {metadata_string}"
              )
        return "\n\n<NEWDOC>\n\n".join(doc_strings)
   ```


5. In "RAGHelper_local.py" the following code is added, to resolve the same reformatting problems as above:
   ```ruby
        if os.getenv('rerank_model') == 'flashrank':
            return [d.metadata['relevance_score'] for d in reranked_docs if
                d.page_content in [doc.page_content for doc in reply['docs']]]

        else:
            return [d['metadata']['relevance_score'] for d in reranked_docs if
                d['page_content'] in [doc.page_content for doc in reply['docs']]]
   ```

6. The "requirements.txt" should also be updated since there are new libraries required while using ColBERT. These libraries are:

   ```ruby
   colbert-ir==0.2.14  # Added for ColBERT reranking
   fsspec==2024.9.0
   ragatouille
   ```

7. The "HomeController.scala" was also changed to resolve some problems when starting up. When starting up the application, it is very likely that there still are some things running in the background when trying to interact with the UI. Whilst this is not a problem for the chatbot, as it will respond with try again later, it would throw up an error looking at the added documents. Therefore, an error handler needed to be added in such a way that the retrieval of all documents would still give a valid response when the response code was not equal to 200 (given when a request is accepted). It was done in the following way:

   ```ruby
   def add() = Action.async { implicit request: Request[AnyContent] =>

    ws
      .url(s"${config.get[String]("server_url")}/get_documents")
      .withRequestTimeout(5 minutes)
      .get()
      .map(files => {
      if (files.status == 200) {
        Ok(views.html.add(files.json.as[Seq[String]]))
        } else {
          // Handle error cases
          Status(files.status)(s"Error: ${files.statusText} \n Please try refreshing the page.")
        }
      })}
   ```


## Limitations 
- Although the ColBERTReranker should successfully obtain more insightful responses in a faster time, due to its support for reranking with late interaction; its current performance and running times do not reflect this behavior. This is partially due to the fact that the indexing is currently performed at initialisation of the application, rather than at initialisation of the server. However, given the limited size of the data, this is currently still feasible and does not take that long.
- The setting "use_rewrite_loop" to True results in better responses, but there seems to be a randomness issue where occasionally the response cannot be visualized in the user interface. Even though the response can be generated in the Google Colab "server.py" output, the team has attempted to fix this issue, but is not proficient with Scala and struggled to find a solution. This can be a logical consequence from the fact that the rewriting of the model is often in the format {query} Read the question again: {query}. As said in the previous section, the maximum length of a query is equal to 32, which is the recommended maximum by multiple integrations ([Jina AI, 2024](https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/)). However, if the original query is more than 14 tokens, this limit will be breached and the model might not generate the response in the correct way.
- Lastly, the current implementation fixates the number of returned documents, rather than a dynamic number of documents which are returned.


These were the most prevalent steps for replacing FlashRank with a late-interaction model like ColBERT.

For implementing further changes take a look at: https://github.com/AI-Commandos/RAGMeUp/compare/main...LvR33:RAGMeUp:ElanoIter

Sources:
- Jina AI. (2024, August 30). Jina ColBERT v2: Multilingual Late Interaction Retriever for Embedding and Reranking. https://jina.ai/news/jina-colbert-v2-multilingual-late-interaction-retriever-for-embedding-and-reranking/
