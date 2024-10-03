import random
import logging
from dotenv import load_dotenv
import os
from random import sample

from RAGHelper_cloud import RAGHelperCloud
from RAGHelper import RAGHelper
from RAGHelper import formatDocuments

from datasets import Dataset

from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from ragas.run_config import RunConfig

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
    formatted_docs = formatDocuments(selected_docs)

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

    qa_pairs.append({"question": question, "ground_truth": answer})

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
