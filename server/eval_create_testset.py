import random
import logging
from dotenv import load_dotenv
import os
from random import sample
import pandas as pd
import shutil

from RAGHelper_cloud import RAGHelperCloud
from RAGHelper_local import RAGHelperLocal
from RAGHelper import RAGHelper

from datasets import Dataset

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough



# Load RAG
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

use_cloud = False
if os.getenv("use_openai") == "True" or os.getenv("use_gemini") == "True" or os.getenv("use_azure") == "True":
    raghelper = RAGHelperCloud(logger)
    use_cloud = True
else:
    raghelper = RAGHelperLocal(logger)

# Set variables from environment
ragas_use_n_documents = int(os.getenv("vector_store_k"))
if os.getenv("rerank") == "True":
    ragas_use_n_documents = int(os.getenv("rerank_k"))
end_string = os.getenv("llm_assistant_token")
ragas_sample_size = int(os.getenv("ragas_sample_size"))
ragas_qa_pairs = int(os.getenv("ragas_qa_pairs"))

# Set up the documents and get a sample
documents = raghelper.chunked_documents

print(f'Number of documents: {len(documents)}')
print(f'Number of documents to sample: {ragas_sample_size}')
document_sample = sample(documents, ragas_sample_size)

# Prepare template for generating questions
if use_cloud:
    thread = [
        ('system', os.getenv('ragas_question_instruction')),
        ('human', os.getenv('ragas_question_query'))
    ]
    prompt = ChatPromptTemplate.from_messages(thread)
else:
    thread = [
        {'role': 'system', 'content': os.getenv("ragas_question_instruction")},
        {'role': 'user', 'content': os.getenv("ragas_question_query")}
    ]
    prompt_template = raghelper.tokenizer.apply_chat_template(thread, tokenize=False)
    prompt = PromptTemplate(
        input_variables=["context"],
        template=prompt_template,
    )

rag_question = prompt | raghelper.llm

# Prepare template for generating answers with our questions
if use_cloud:
    thread = [
        ('system', os.getenv('ragas_answer_instruction')),
        ('human', os.getenv('ragas_answer_query'))
    ]
    prompt = ChatPromptTemplate.from_messages(thread)
else:
    thread = [
        {'role': 'system', 'content': os.getenv("ragas_answer_instruction")},
        {'role': 'user', 'content': os.getenv("ragas_answer_query")}
    ]
    prompt_template = raghelper.tokenizer.apply_chat_template(thread, tokenize=False)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

rag_answer = prompt | raghelper.llm

# Create test set
qa_pairs = []
for i in range(0, ragas_qa_pairs):
    selected_docs = random.sample(document_sample, min(ragas_use_n_documents, len(document_sample)))
    formatted_docs = RAGHelper.format_documents(selected_docs)

    question_chain = ({"context": RunnablePassthrough()} | rag_question)
    response = question_chain.invoke(formatted_docs)
    if use_cloud:
        if hasattr(response, 'content'):
            question = response.content
        elif hasattr(response, 'answer'):
            question = response.answer
        elif 'answer' in response:
            question = response["answer"]
    else:
        question = response[response.rindex(end_string)+len(end_string):]

    answer_chain = ({"context": RunnablePassthrough(), "question": RunnablePassthrough()} | rag_answer)
    response = answer_chain.invoke({"context": formatted_docs, "question": question})
    if use_cloud:
        if hasattr(response, 'content'):
            answer = response.content
        elif hasattr(response, 'answer'):
            answer = response.answer
        elif 'answer' in response:
            answer = response["answer"]
    else:
        answer = response[response.rindex(end_string)+len(end_string):]

    id_set = set([d.metadata.get("id", "no id found") for d in selected_docs])

    qa_pairs.append({"question": question, "ground_truth": answer, "true_doc_ids": id_set})


# convert type dataset to pandas dataframe
df_testset = pd.DataFrame(qa_pairs)

# Variables
rag_chunks_path = "rag_chunks.pickle"
testsets_folder = "testsets"

# Ensure the base testsets folder exists
os.makedirs(testsets_folder, exist_ok=True)

# Determine the next available folder name
folder_number = 1
while True:
    target_folder = os.path.join(testsets_folder, str(folder_number))
    if not os.path.exists(target_folder):
        break
    folder_number += 1

# Create the new folder
os.makedirs(target_folder)

# Save the DataFrame as CSV in the new folder
testset_csv_path = os.path.join(target_folder, "testset.csv")
df_testset.to_csv(testset_csv_path, index=False)

# Copy the rag_chunks file to the new folder
rag_chunks_copy_path = os.path.join(target_folder, "rag_chunks.pickle")
shutil.copy(rag_chunks_path, rag_chunks_copy_path)

print(f"Data saved in folder: {target_folder}")
