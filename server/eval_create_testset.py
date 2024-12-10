import random
import logging
from dotenv import load_dotenv
import os
from random import sample
import pandas as pd
import shutil
import pickle
import ast

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
use_n_documents = int(os.getenv("vector_store_k"))
if os.getenv("rerank") == "True":
    use_n_documents = int(os.getenv("rerank_k"))
end_string = os.getenv("llm_assistant_token")
sample_size = int(os.getenv("eval_sample_size"))
n_qa_pairs = int(os.getenv("eval_qa_pairs"))

use_example_questions = os.getenv("eval_use_example_questions").lower() == "true"
catch_irrelevant_chunks = os.getenv("eval_catch_irrelevant_chunks").lower() == "true"
check_sample_relevance = os.getenv("eval_check_sample_relevance").lower() == "true"
retrieve_samples = os.getenv("eval_retrieve_samples").lower() == "true"


# configure example questions
if use_example_questions:
  example_questions = ast.literal_eval(os.getenv("eval_example_questions"))
  current_q_query = os.getenv('eval_question_query')
  prompt_text = os.getenv('eval_example_questions_prompt')
  for i, question in enumerate(example_questions):
      prompt_text += f'{i+1}. {question}\n'
  new_q_query = current_q_query + prompt_text
  os.environ['eval_question_query'] = new_q_query

# configure catching irrelevant chunks (denying question generation)
if catch_irrelevant_chunks:
  current_q_query = os.getenv('eval_question_query')
  prompt_text = os.getenv('eval_catch_irrelevant_chunks_prompt')
  new_q_query = f'{current_q_query} {prompt_text}'
  os.environ['eval_question_query'] = new_q_query

# configure using the same samples as defined testset
if retrieve_samples:
  testset_path = f'./testsets/{os.getenv("eval_retrieve_samples_folder")}/'

  # load data
  testset = pd.read_csv(testset_path + 'testset.csv')
  testset['true_doc_ids'] = testset['true_doc_ids'].apply(ast.literal_eval)
  n_qa_pairs = len(testset)

  # Path to chunks
  testset_chunks_path = testset_path + 'rag_chunks.pickle'

  # Load chunks
  with open(testset_chunks_path, "rb") as file:
      rag_chunks = pickle.load(file)

  sampled_docs = testset['true_doc_ids'].apply(
    lambda ids: [doc for doc in rag_chunks if doc.metadata['id'] in ids]
    ).to_list()



# Set up the documents and get a sample
documents = raghelper.chunked_documents
document_sample = sample(documents, sample_size)

# Prepare template for checking sample relevance
if check_sample_relevance:
    if use_cloud:
        thread = [
            ('system', os.getenv('eval_sample_relevance_instruction')),
            ('human', os.getenv('eval_sample_relevance_query'))
        ]
        prompt = ChatPromptTemplate.from_messages(thread)
    else:
        thread = [
            {'role': 'system', 'content': os.getenv("eval_sample_relevance_instruction")},
            {'role': 'user', 'content': os.getenv("eval_sample_relevance_query")}
        ]
        prompt_template = raghelper.tokenizer.apply_chat_template(thread, tokenize=False)
        prompt = PromptTemplate(
            input_variables=["context"],
            template=prompt_template,
        )

    rag_sample_relevance = prompt | raghelper.llm

# Prepare template for generating questions
if use_cloud:
    thread = [
        ('system', os.getenv('eval_question_instruction')),
        ('human', os.getenv('eval_question_query'))
    ]
    prompt = ChatPromptTemplate.from_messages(thread)
else:
    thread = [
        {'role': 'system', 'content': os.getenv("eval_question_instruction")},
        {'role': 'user', 'content': os.getenv("eval_question_query")}
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
        ('system', os.getenv('eval_answer_instruction')),
        ('human', os.getenv('eval_answer_query'))
    ]
    prompt = ChatPromptTemplate.from_messages(thread)
else:
    thread = [
        {'role': 'system', 'content': os.getenv("eval_answer_instruction")},
        {'role': 'user', 'content': os.getenv("eval_answer_query")}
    ]
    prompt_template = raghelper.tokenizer.apply_chat_template(thread, tokenize=False)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

rag_answer = prompt | raghelper.llm

# Create test set
qa_pairs = []
qa_generated = 0
rejected_samples = []

while qa_generated < n_qa_pairs:
    if retrieve_samples:
        selected_docs = sampled_docs[qa_generated]
    else:
        selected_docs = random.sample(document_sample, min(use_n_documents, len(document_sample)))
    formatted_docs = RAGHelper.format_documents(selected_docs)

    if check_sample_relevance:
        sample_relevance_chain = ({"context": RunnablePassthrough()} | rag_sample_relevance)
        response = sample_relevance_chain.invoke(formatted_docs)
        if use_cloud:
            if hasattr(response, 'content'):
                sample_relevance = response.content
            elif hasattr(response, 'answer'):
                sample_relevance = response.answer
            elif 'answer' in response:
                sample_relevance = response["answer"]
        else:
            sample_relevance = response[response.rindex(end_string)+len(end_string):]

        if 'false' in sample_relevance.lower():
            rejected_samples.append(selected_docs)
            continue

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

    qa_generated += 1

# convert type dataset to pandas dataframe
df_testset = pd.DataFrame(qa_pairs)

# Save testset in folder
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

print(f"Testset saved in folder: {target_folder}")