# Made changes to the code 
The documents that are retrieved from the vector store are found by using a similarity search of the original query. In this PR we implemented a new approach using Hyde. With hyde, you create a hypothetical document from the query, which you in turn use to find documents in the vector store through a similarity search. To implement this functionality, some changes had to be made to the current way of how similar documents are retrieved, previously the RAG chain was given the full handling of retrieving documents and answering the user query. However, the hypothetical document is created from the  user's query, and it must only be used for retrieving documents and not for answering the user's question. This was not possible with Langchains complex way of using pipes for us. In the image below the flowchart of how hyde affects the process is shown.

<img width="896" alt="image" src="https://github.com/user-attachments/assets/d1fe76e1-c65b-467f-a984-a62ee4730ed7">

### Where in the code the changes are made
- `RAGHelper_local`
   - Changed function `handle_user_interaction`, to make function work more similar to the same function in RAGHelper_cloud
   - If hyde=True (in env)
      
- `RAGHelper_cloud`
- `RAGHelper`
   - Functions are added in `RAGHelper` to create a hypothetical document from query:
      - `apply_hyde_if_enabled`
      - `embed_query_with_hyde`
      - `_initialize_hyde_embeddings`
      - `self.hyde_embeddings` -> `CustomHyDE.py`
         ->    ```
               CustomHypotheticalDocumentEmbedder(
               llm_chain=llm_chain,
               base_embeddings=base_embeddings
               )
               ```
     - `self.hyde_embeddings.embed_query(query, return_text=True)`
- `requirements.txt` -> for making the system work in Paperspace

### How we were able to run server.py in paperspace
1. Create virtual environment with python version 3.10.12
```pip install virtualenv
virtualenv venv --python=python3.10.12
source venv/bin/activate
```
2. Clone the repo
```
git clone https://github.com/AI-Commandos/RAGMeUp.git
```
3. Install our refined requirements.txt (without pytorch)
```
pip install -r RAGMeUp/server/requirements.txt
```
5. Install pyngrok, and add auth token
```
pip install pyngrok
ngrok authtoken [INSERT NGROK TOKEN]
```
6. Setup hugginface with git
```
git config --global credential.helper store
huggingface-cli login
```
7. Install specfic version of pytorch
```
pip install 'torch @ https://download.pytorch.org/whl/cu121_full/torch-2.5.1%2Bcu121-cp310-cp310-linux_x86_64.whl'
```
9. Run the server
```
cd RAGMeUp/server
python server.py
```
9. Copy the printed ngrok tunnel URL, and use it to setup the scala UI in the project locally.

# Report
# RAG Pipeline Enhancement: HyDE Integration

## Objective

The goal of this project was to enhance the Retrieval-Augmented Generation (RAG) pipeline by integrating Hypothetical Document Embedding (HyDE). This method generates a hypothetical document as an intermediate step, which is then used for retrieval alongside the original query. The report details the thought process, implementation, configurations, and the advantages and disadvantages of this approach.

---

## Implementation Overview

### Approach

1. **Enhancing Query Representation with HyDE:**
   - HyDE generates a hypothetical document for the input query.
   - Both the original query and the generated document are used for similarity search.

2. **Dynamic Prompt Template System:**
   - Generalized templates for flexibility.
   - Custom templates for specific scenarios.

3. **Environmental Variable Configuration:**
   - All parameters for HyDE templates and behaviors are configurable via `.env` to ensure reproducibility and customization.

4. **End-to-End Testing:**
   - Ensured functionality through Flask endpoints to validate HyDE embeddings.

---

## Configurations

### Generalized Template
```plaintext
hyde_general_template="Write a {context_type} passage to {action}. Include {additional_context}.
Your response must be direct and avoid any unnecessary phrases, personal remarks, or repetitive text."
```

- **Explanation:**
  - This is the generalized template used for creating HyDE prompts. It includes placeholders (`{context_type}`, `{action}`, `{additional_context}`) that are dynamically replaced based on the query's context.

- **Example Usage:**
  - **`hyde_default_context_type`**: Specifies the type of passage (e.g., `scientific`, `financial`, `technical`).
  - **`hyde_default_action`**: Defines the purpose of the passage (e.g., `answer the following question`, `summarize the input`).
  - **`hyde_default_additional_context`**: Adds further instructions (e.g., `Ensure accuracy and provide references.`).

---

### Custom Template Example
```plaintext
hyde_custom_template="Write a passage in Korean to answer the
question in detail."
```

- **Explanation:**
  - This variable allows users to define their own custom templates for specific use cases. If provided, this template overrides the generalized template.
  - **Example**: A template that generates responses in a specific language or format.

---

### Other Configurations
```plaintext
hyde_enabled=True
hyde_multi_generations=1
```

- **`hyde_enabled`**: Enables or disables the HyDE functionality. Set to `True` to activate HyDE in the pipeline.
- **`hyde_multi_generations`**: Specifies the number of hypothetical document generations for HyDE. Increase this value for more comprehensive retrieval, but it will greatly increase the time it takes for retrieval.

---

### Environmental Variable Template
```plaintext
hyde_enabled=True  # Activates the HyDE integration.
hyde_multi_generations=1  # Number of hypothetical documents generated per query.
hyde_general_template="Write a {context_type} passage to {action}. Include {additional_context}.
Your response must be direct and avoid any unnecessary phrases, personal remarks, or repetitive text."  # Template for dynamic prompt generation.
hyde_custom_template=""  # Custom template for specific use cases. Overrides the general template if provided.
hyde_default_context_type="scientific"  # Default context type for the HyDE template (e.g., scientific, financial, technical).
hyde_default_action="answer the following question"  # Default action to define the purpose of the passage.
hyde_default_additional_context="Please provide references and ensure an academic tone."  # Additional instructions for generating high-quality responses.
```
---

# Results and Reflections on HyDE Performance

The performance of the HyDE-enhanced RAG pipeline was evaluated by comparing retrieval outcomes with and without HyDE for a variety of queries. The database was created using the slides of Natural Language Processing course. The questions are purposefully simplistic to highlight HyDE's strength with somewhat ambiguous queries. All the results can be found in ```hyde_table_markdown.md```.  The following table summarizes the key findings:

| **Query**                                           | **Documents Found (With HyDE)**                                                                                               | **Provenance Score (With HyDE)** | **Documents Found (Without HyDE)**                                                                                            | **Provenance Score (Without HyDE)** | **Time Taken (With HyDE)** | **Time Taken (Without HyDE)** |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------|----------------------------|-------------------------------|
| **What are the main considerations of the LDA paper?** | Detailed provenance from scientific documents related to LDA with scores ~0.99                                              | 0.993                            | Fewer references; lacks contextual links relevant to LDA                                                                      | 0.957                              | 46.72 seconds               | 23.69 seconds                |
| **What are the main challenges of prompt engineering?** | Identified nuanced issues, including sensitivity to scale and limitations of few-shot prompts                               | 0.993                            | Less contextual detail, with partial focus on challenges like privacy and optimization complexities                           | 0.942                              | 37.87 seconds               | 16.26 seconds                |
| **What are the different types of topic modeling?** | Broad insights into LDA, FLSA, and correlated/dynamic topic models with a high provenance score of ~0.988                   | 0.988                            | Focused on LDA with minimal additional context on related models                                                              | 0.976                              | 55.43 seconds               | 16.12 seconds                |
| **What are the limitations of BERTopic?**            | Highlights weaknesses like reliance on single-topic assumption per document and limited contextual relationships in topics | 0.997                            | Partial identification of BERTopic's limitations but misses finer details like interpretability challenges                     | 0.980                              | 70.46 seconds               | 13.46 seconds                |

---

## Observations

1. **Enhanced Document Retrieval:**
   - With HyDE, the pipeline consistently identified more relevant and contextually aligned documents, as evidenced by higher provenance scores.
   - Example: For the LDA-related query, HyDE retrieved documents with a provenance score of 0.993 versus 0.957 without HyDE.

2. **Processing Time Trade-Off:**
   - The time required for HyDE-enabled queries was approximately **2x longer**, reflecting the computational overhead of generating hypothetical documents.

3. **Improved Contextual Depth:**
   - HyDE provided broader and more insightful document retrieval, especially for abstract or complex queries like those involving prompt engineering and topic modeling.

---

## Recommendations for Future Optimization

- **Optimize for Latency:** Leverage smaller, faster models for hypothetical document generation to reduce processing time.
- **Implement Caching:** Cache results for recurring queries to enhance response speed without sacrificing accuracy.
- **Hybrid Strategies:** Experiment with combining dense and sparse retrieval methods for efficiency.

---


---

## Reflection: Advantages and disadvantages of HyDE in RAG

### Advantages
1. **Enhanced Retrieval Accuracy:**
   - Generating a hypothetical document often aligns better with the retrieval task than using the raw query alone.

2. **Increased Flexibility:**
   - Environmental variables allow for easy reconfiguration without code changes.

3. **Interoperability:**
   - HyDE embeds seamlessly into existing RAG pipelines, leveraging the power of LangChain.

4. **Reduction in Embedding Dependencies:**
   - Avoids the need for custom embedding algorithms by utilizing the LLM for hypothetical document generation.
5.  **Domain-Specific Relevance**: 
     - Effective for specialized applications like healthcare and legal domains.
3.  **Modular and Configurable**: 
     - Allows for experimentation with retrieval, reranking, and summarization modules.
   

### Disadvantages
1. **LLM Dependency:**
   - The success of HyDE is heavily reliant on the quality of the LLM used for hypothetical document generation.

2. **Risk of Hallucination:**
   - Hypothetical documents might misrepresent the user's intent, leading to retrieval of irrelevant results.

3. **Performance Overhead:**
   - Generating a hypothetical document introduces latency, especially in multi-generation scenarios.

4. **Domain Adaptation:**
   - Requires careful tuning of templates and configurations to align with specific domains.

---

## Incorporating Insights from Recent Research
Our approach incorporates findings from the 2024 study, "Searching for Best Practices in Retrieval-Augmented Generation." Key takeaways include:
1. **Hybrid Retrieval + HyDE**: Combining sparse (BM25) and dense embeddings, along with pseudo-documents generated by HyDE, significantly improves retrieval accuracy.
2. **Efficiency Trade-offs**: While the best-performance pipeline prioritizes accuracy, a balanced-efficiency pipeline can achieve comparable results with reduced latency.


---

## Future Directions

1. **Mitigating Hallucinations:**
   - Explore prompt engineering and domain-specific LLM fine-tuning.

2. **Integrating Late Interaction Models:**
   - Combining HyDE with reranking mechanisms like ColBERT for finer retrieval granularity.

3. **Evaluating RAG Performance:**
   - Incorporate automated RAG evaluation methods to measure faithfulness and reduce hallucination risks.

4. **Expanding Modalities:**
   - Adapt HyDE to support multi-modal input (e.g., text and vision).

---

## References

1. Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Precise zero-shot dense retrieval without relevance labels. arXiv preprint arXiv:2212.10496.
2. Wang, X., Wang, Z., Gao, X., Zhang, F., Wu, Y., Xu, Z., ... & Huang, X. J. (2024, November). Searching for best practices in retrieval-augmented generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (pp. 17716-17736).
3. LangChain Documentation - [LangChain](https://langchain.com)
4. RAG Me Up Repository - [GitHub](https://github.com/GeorgeAntono/RAGMeUp)

---
