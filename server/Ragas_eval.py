import random
import logging
from dotenv import load_dotenv
import os
from random import sample
import json

from RAGHelper_cloud import RAGHelperCloud
from RAGHelper import RAGHelper

from datasets import Dataset

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.run_config import RunConfig
import time  # For latency calculation
from sacrebleu.metrics import BLEU  # For BLEU evaluation
from rouge_score import rouge_scorer  # For ROUGE evaluation

load_dotenv()
os.environ["use_rewrite_loop"] = "False"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

use_cloud = False
if os.getenv("use_openai") == "True" or os.getenv("use_gemini") == "True" or os.getenv("use_azure") == "True":
    raghelper = RAGHelperCloud(logger)
    use_cloud = True
else:
    raghelper = RAGHelper(logger)

ragas_use_n_documents = int(os.getenv("vector_store_k"))
if os.getenv("rerank") == "True":
    ragas_use_n_documents = int(os.getenv("rerank_k"))
end_string = os.getenv("llm_assistant_token")
ragas_sample_size = int(os.getenv("ragas_sample_size"))
ragas_qa_pairs = int(os.getenv("ragas_qa_pairs"))

# Set up the documents and get a sample
documents = raghelper.chunked_documents
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

    start_time = time.time()
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
    question_end_time = time.time()  # End latency timer for question

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
    answer_end_time = time.time()  # End latency timer for answer

    # --- Append QA Pair ---
    qa_pairs.append({
        "question": question,
        "ground_truth": answer,
        "retrieval_time": question_end_time - start_time,
        "answer_time": answer_end_time - question_end_time,
        "user_feedback": random.choice(["positive", "negative"]),  # Simulated feedback
    })

# Now we invoke the actual RAG pipeline
new_qa_pairs = []
for qa_pair in qa_pairs:
    (nh, response) = raghelper.handle_user_interaction(qa_pair['question'], [])
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

    result_dict = qa_pair
    result_dict['answer'] = answer
    result_dict['context'] = [doc.page_content for doc in docs]
    new_qa_pairs.append(result_dict)

# Convert qa_pairs to a format Ragas expects
ragas_data = [{
    "question": pair["question"],
    "answer": pair["answer"],
    "contexts": pair["context"],
    "ground_truth": pair["ground_truth"]
} for pair in new_qa_pairs]

# Create a Hugging Face Dataset
dataset = Dataset.from_list(ragas_data)
dataset.save_to_disk(os.getenv("ragas_dataset"))

# Evaluate

def evaluate_pipeline(ragas_data):
    """Run evaluation metrics and log results."""
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
    from ragas.run_config import RunConfig

    # --- RAGAS Metrics ---
    run_config = RunConfig(
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy]
    )
    # ragas_scores = evaluate(ragas_data, run_config) # not anything relevant

    # --- Latency Metrics ---
    retrieval_times = [item['retrieval_time'] for item in ragas_data]
    answer_times = [item['answer_time'] for item in ragas_data]

    avg_retrieval_latency = sum(retrieval_times) / len(retrieval_times)
    avg_answer_latency = sum(answer_times) / len(answer_times)

    # --- NLG Metrics (Readability) ---
    questions = [item['question'] for item in ragas_data]
    answers = [item['answer'] for item in ragas_data]
    ground_truths = [item['ground_truth'] for item in ragas_data]
    # Initialize BLEU and ROUGE scorer
    bleu = BLEU()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # Compute BLEU score
    bleu_score = bleu.corpus_score(answers, [ground_truths]).score
    # Compute ROUGE scores for each answer-ground_truth pair and average them
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for answer, ground_truth in zip(answers, ground_truths):
        rouge_result = scorer.score(answer, ground_truth)
        for key in rouge_scores:
            rouge_scores[key].append(rouge_result[key].fmeasure)
    # Average ROUGE scores
    avg_rouge_scores = {key: sum(values) / len(values) for key, values in rouge_scores.items()}
    # Combine scores
    readability_scores = {
        "BLEU": bleu_score,
        "ROUGE": avg_rouge_scores
    }

    # --- Log Results ---
    #logger.info(f"RAGAS Scores: {ragas_scores}")
    logger.info(f"Avg Retrieval Latency: {avg_retrieval_latency:.2f}s")
    logger.info(f"Avg Answer Latency: {avg_answer_latency:.2f}s")
    logger.info(f"Readability Metrics: {readability_scores}")

    # --- Save Results to File
    with open("evaluation_results.json", "w") as f:
        json.dump({
           # "ragas_scores": ragas_scores, Did not mean anything ultimately 
            "latency": {
                "retrieval": avg_retrieval_latency,
                "answer": avg_answer_latency,
            },
            "readability": readability_scores
        }, f, indent=4)
# Evaluate the RAG Pipeline
evaluation_result = evaluate_pipeline(ragas_data)

# Print results
print(evaluation_result)
