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

from ragas import EvaluationDataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.run_config import RunConfig



# Load RAG
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

use_cloud = False
if os.getenv("use_openai") == "True" or os.getenv("use_gemini") == "True" or os.getenv("use_azure") == "True":
    raghelper = RAGHelperCloud(logger)
    use_cloud = True
else:
    raghelper = RAGHelperLocal(logger)

# set variables
end_string = os.getenv("llm_assistant_token")
k = int(os.getenv("rerank_k"))
eval_ragas = os.getenv("eval_ragas").lower() == "true"
testset_path = os.getenv("eval_testset_directory")
modelname = os.getenv("eval_RAG_instance_name")

# load testset
testset = pd.read_csv(testset_path + 'testset.csv')
testset['true_doc_ids'] = testset['true_doc_ids'].apply(ast.literal_eval)

# run RAG on each question in the testset
evalset = testset.copy()

RAG_answers = []
RAG_doc_ids = []
for question in testset['question']:
    (nh, response) = raghelper.handle_user_interaction(question, [])
    docs = response['docs']
    if use_cloud:
        if hasattr(response, 'content'):
            answer = response.content
        elif hasattr(response, 'answer'):
            answer = response.answer
        elif 'answer' in response:
            answer = response["answer"]
    else:
        answer = response['text'][response['text'].rindex(end_string)+len(end_string):]

    RAG_answers.append(answer)
    RAG_doc_ids.append([d.metadata.get("id", "no id found") for d in docs])

evalset[f'{modelname}_answer'] = RAG_answers
evalset[f'{modelname}_doc_ids'] = RAG_doc_ids


# Compute evaluation metrics

# Ensure `true_doc_ids` is parsed from string to list
if isinstance(evalset["true_doc_ids"].iloc[0], str):
  evalset["true_doc_ids"] = evalset["true_doc_ids"].apply(ast.literal_eval)

# Compute the count of intersecting IDs
n_docs_colname = f'{modelname}_n_docs_identified'
evalset[n_docs_colname] = evalset.apply(
    lambda row: len(set(row["true_doc_ids"]) & set(row[f"{modelname}_doc_ids"])),
    axis=1,
)

n_docs_top_k_colname = f'{modelname}_n_docs_identified_top_{k}'
evalset[n_docs_top_k_colname] = evalset.apply(
    lambda row: len(set(row["true_doc_ids"]) & set(row[f"{modelname}_doc_ids"][:k])),
    axis=1,
)

recall_colname = f'{modelname}_recall'
recall_top_k_colname = f'{modelname}_recall_top_{k}'
evalset[recall_colname] = evalset[n_docs_colname] / evalset['true_doc_ids'].apply(len)
evalset[recall_top_k_colname] = evalset[n_docs_top_k_colname] / evalset['true_doc_ids'].apply(len)

mean_recall = evalset[recall_colname].mean()
mean_recall_top_k = evalset[recall_top_k_colname].mean()
print(f'Model {modelname} has a recall of {mean_recall}')
print(f'Model {modelname} has a recall-top-{k} of {mean_recall_top_k}')

# Add true doc contexts column
with open(testset_path + 'rag_chunks.pickle', "rb") as file:
    rag_chunks = pickle.load(file)

evalset['true_doc_contexts'] = evalset['true_doc_ids'].apply(
    lambda ids: [doc.page_content for doc in rag_chunks if doc.metadata['id'] in ids]
    )

# Ensure the base evalsets folder exists
evalsets_folder = "evalsets"
os.makedirs(evalsets_folder, exist_ok=True)

# Determine the next available folder name
folder_number = 1
while True:
    target_folder = os.path.join(evalsets_folder, str(folder_number))
    if not os.path.exists(target_folder):
        break
    folder_number += 1

# Create the new folder
os.makedirs(target_folder)

evalset.to_csv(os.path.join(target_folder, "evalset.csv"), index=False)
evalset.to_excel(os.path.join(target_folder, "evalset.xlsx"), index=False)
shutil.copy(testset_path + 'rag_chunks.pickle', os.path.join(target_folder, "rag_chunks.pickle"))



# Create ragas formatted data
if eval_ragas:
    ragas_data = evalset.copy()
    ragas_cols = ['question', 'ground_truth', 'answer', 'contexts']
    
    # get cols of first specified model
    answer_col = [col for col in ragas_data.columns if col.endswith('answer')][0]
    context_col = [col for col in ragas_data.columns if col.endswith('doc_ids')][1]

    # rename answer_col to 'answer'
    ragas_data = ragas_data.rename(columns={answer_col: 'answer'})

    # Path to chunks
    file_path = "rag_chunks.pickle"

    # Load chunks
    with open(file_path, "rb") as file:
        rag_chunks = pickle.load(file)

    ragas_data['contexts'] = ragas_data[context_col].apply(
        lambda ids: [doc.page_content for doc in rag_chunks if doc.metadata['id'] in ids]
        )

    # delete columns that are not in ragas_cols
    ragas_data = ragas_data[ragas_cols]

    # Convert to new format in newer ragas version
    if True:
      new_ragas_cols = {"question": "user_input", "ground_truth": "reference", "answer": "response", "contexts": "retrieved_contexts"}
      ragas_data = ragas_data.rename(new_ragas_cols, axis=1)
    
    # create ragas dataset
    eval_dataset = EvaluationDataset.from_pandas(ragas_data)

    # Evaluate
    results = evaluate(
        eval_dataset,
        llm=raghelper.llm,
        embeddings=raghelper.embeddings,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall
            ],
        run_config=RunConfig(max_workers=1, timeout=600.0)
        )
    print("Evaluation Results:")
    print(results)