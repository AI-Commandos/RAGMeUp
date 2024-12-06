
# RAG Evaluation

This repository builds on the RAGMeUp framework. This README is specific to the added RAG Evaluation framework. The framework was ran from Google Colab. It is advised to run the scripts from [`Colab_RAG_Eval.ipynb`](./Colab_RAG_Eval.ipynb) in the Colab environment. The file uses a .env template for evaluation. This template is loaded as .env file, and later described variables can be changed by writing to this environment. Lastly, Ensure you have a HuggingFace Token to insert. 

Run the [`eval_create_testset.py`](./eval_create_testset.py) file to create a testset. This testset is a dataset of QA-pairs. It is saved as a .csv file in the folder testsets. If this directory does not exist in the server directory, it is created. Within this folder, a new folder is created to save the `testset.csv` file and a `rag_chunk.pickle` file, which stores the chunks that are parsed from the documents.\
The following variables can be adjusted before creating the testset:
- `chunk_size` set the size of chunks the script uses to generate questions from.
- `rerank_k` set to define how many chunks the LLM uses to generate a question from. (It is advised keep `rerank` to True for RAG Evaluation).
- `eval_qa_pairs` set the number of Question-Answer pairs that should be generated. 
- `eval_sample_size` set the number of chunks to sample from for generating QA-pairs.
- `eval_question_query` set the prompt for generating questions.
- `eval_catch_irrelevant_chunks` Set if a prompt should be added to the question query to allow the LLM not to create a question based on irrelevant chunks (True/False).
- `eval_catch_irrelevant_chunks_prompt` Set the prompt to use if the previous variable is True.
- `eval_check_sample_relevance` set if the LLM should first judge a chunk if it is relevant to generate a question from (True/False). 
- `eval_check_sample_relevance_instruction` set the instruction prompt if check_sample_relevance is True.
- `eval_check_sample_relevance_query` set the query prompt if check_sample_relevance is True.
- `eval_retrieve_samples` set if the same samples as a previously generated testset should be used (True/False).
- `eval_retrieve_samples_folder` set the folder from which the testset should be retrieved if the previous variable is True.
- `eval_use_example_questions` Set if a prompt should be added to the question query to provide example questions to the LLM.
- `eval_example_questions` Set the example questions if the previous variable is True. Provide them as a string of a list.
- `eval_example_questions_prompt` set the prompt to instruct the LLM what to do with the example questions if use_example_questions is True.

Run the [`eval_evaluate_RAG.py`](./eval_evaluate_RAG.py) file to evaluate a RAG instance with a specified testset. The RAG's retrieved chunks and generated answers are added to the testset, and Recall and Recall-top-k are computed and printed. The resulting evalset in the same way as the testset, and as a excel file for inspection. The following variables can be adjusted before evaluating the RAG:
- `eval_testset_directory` set the directory in which the testset to use can be found.
- `eval_RAG_instance_name` set the name of the RAG instance, such that instances can be compared from their column names.
- `eval_ragas` set if the Ragas library should be used to compute evaluation metrics. Note that this is expected to give a timeout or Out Of Memory error when running in Colab.

