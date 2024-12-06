# Report
The steps taken in the process of this assignment are as follows:
- The Original RAGMeUp Repository was cloned to our group’s GitHub account.
- The documentation was read thoroughly to gain an understanding of the pipeline.
- The course data (Lecture slides, papers, book chapters) were collected and renamed for interpretation clarity. This renamed data was uploaded into the bot’s data storage, allowing the RAG model to retrieve information from this material.

## Actual adjustments and additions to the RAG components 
1. Additional evaluation metrics (BLEU and ROUGE scores) were added to the RAG model, ensuring relevant output.
2. The reranker component of the RAG model was changed from FlashRank to ColBERT.

### 1. Additional Evaluation Metrics (BLEU and ROUGE Scores)

**BLEU and ROUGE scores**
- **BLEU (Bilingual Evaluation Understudy):**
BLEU is a precision-based metric that calculates how much of the generated text overlaps with the reference text (ground truth). It focuses on n-gram overlap (phrases of size 1 to 4) and penalizes overly short outputs. A higher BLEU score indicates that the generated response closely matches the ground truth in terms of word choice and structure.

- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
ROUGE evaluates recall by measuring how much of the reference text is captured in the generated response.

**Latency Measurement**
Using Python’s _time_ module, we measured:
- **Retrieval Latency:** Time taken to retrieve relevant documents.
-**Answer Generation Latency:** Time taken to generate the final response from the model.
This combination of quality (BLEU and ROUGE) and speed (latency) metrics allowed us to holistically evaluate the model's performance.

### 2. ColBERT (Contextualized Late Interaction over BERT)
RAG models are complex models, and to optimize their performance, many components need to be optimized. The model used for retrieval is an important component of the RAG model, as it is what sets RAG apart from other models. 

In a RAG model, the reranker selects the set of documents retrieved by the retrieval component to ensure the most relevant information is passed to the generator. It reorders the initial set of retrieved documents to ensure the retrieved documents are relevant for answering the query, essentially giving them a ‘second look’ before sending them to the model. This improves the quality of the generated output by providing the generator with relevant information.
RAG technology is slow as LLM calls introduce latency. Calling the reranking model introduces additional latency, and that is where ColBERT comes in, as it is one of the fastest available models. ColBERT is a reranking model that uses BERT embeddings to understand the context of the query (Khattab & Zaharia, 2020). ColBERT uses BERT to independently encode queries and documents, creating detailed embeddings. Instead of matching everything upfront, it compares these embeddings later using a lightweight scoring method. This approach keeps the model powerful while allowing pre-computed document embeddings, making it much faster. 

Various means of changing the reranker component have been attempted, however, the final implementation was performed through RAGatouille. RAGatouille’s focus lies entirely in simplifying the implementation of state-of-the-art methods to optimize RAG models, offering pre-trained ColBERT alongside methods to easily fine-tune them. 

### Implications and Limitations

#### Implications
1. Improved Answer Relevance:
With the inclusion of ColBERT, our RAG pipeline retrieves more relevant documents and does this faster than FlashRank.

#### Limitations
1. NLG Metrics are Imperfect:
Additional metrics like BERTScore or human evaluation could provide a more nuanced assessment.

Small Dataset:
Our evaluation was conducted on a limited dataset of course materials. A larger and more diverse dataset would provide better insights into the model's generalizability.


### Future Directions
Our primary focus was to improve the speed of the RAG model and improve the relevance of the retrieved documents per query. This was achieved through the implementation of ColBERT and additional evaluation metrics. However, there are many more aspects of the RAG model that could be optimized. One of those aspects is the type of data the model can take for information retrieval. By adding additional data types the model becomes more flexible in which data it can use, simplifying the process of expanding its database. 
Furthermore, we believe that the display of sources after submitting a query to the model could be more detailed. The model currently shows the document alongside a provenance score to display the reliability of the retrieved document. However, we believe it would be beneficial to have the system highlight the lines of text in the document from which the information was retrieved, making it easier for the user to find the context and learn more about the topic of their query. Alongside this, a good feature to add would be a separate window that displays the sources cited in the retrieved paper based on the relevance of the source to the user query.

Additionally, we believe that it could be beneficial to the user to see the metadata of the retrieved files, such as filesize, and number of words. This gives a clear indication of the file quickly and concisely, allowing the user to effortlessly select or download the file they wish to delve deeper into without first having to open it. Ease of use should be a priority when working with LLMs and anything that can speed up the process or make the process easier for the user should be considered for implementation.




### Sources used:
ColBERT Information: https://www.pondhouse-data.com/blog/advanced-rag-colbert-reranker

RAGatouille: https://github.com/AnswerDotAI/RAGatouille 

RAG From Scratch: https://www.youtube.com/watch?v=cN6S0Ehm7_8 
![image](https://github.com/user-attachments/assets/3f0df668-1013-4e91-8720-d93b9f3cbf5d)

Paper introducing ColBERT: Khattab, O., & Zaharia, M. (2020, April 27). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. arXiv.org. https://arxiv.org/abs/2004.12832

