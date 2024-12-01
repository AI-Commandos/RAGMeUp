import random
import logging
from dotenv import load_dotenv
import os
from random import sample

from RAGHelper_cloud import RAGHelperCloud
from RAGHelper import RAGHelper

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, HallucinationMetric, ToolCorrectnessMetric
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage


# Load environment variables
load_dotenv()
os.environ["use_rewrite_loop"] = "False"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use cloud or local RAGHelper
use_cloud = os.getenv("use_openai", "False") == "True"
raghelper = RAGHelperCloud(logger) if use_cloud else RAGHelper(logger)

# Load thresholds and model dynamically from .env
answer_relevancy_threshold = float(os.getenv("deepeval_answer_relevancy_threshold", 0.7))
faithfulness_threshold = float(os.getenv("deepeval_faithfulness_threshold", 0.7))
contextual_precision_threshold = float(os.getenv("deepeval_contextual_precision_threshold", 0.7))
contextual_recall_threshold = float(os.getenv("deepeval_contextual_recall_threshold", 0.7))
contextual_relevancy_threshold = float(os.getenv("deepeval_contextual_relevancy_threshold", 0.7))
hallucination_threshold = float(os.getenv("deepeval_hallucination_threshold", 0.5))
correctness_threshold = float(os.getenv("deepeval_correctness_threshold", 0.5))

# Set the LLM model
llm_model = os.getenv("llm_model")

# Define metrics dynamically
metrics = [
    AnswerRelevancyMetric(threshold=answer_relevancy_threshold, model=llm_model),
    FaithfulnessMetric(threshold=faithfulness_threshold, model=llm_model),
    ContextualPrecisionMetric(threshold=contextual_precision_threshold, model=llm_model),
    ContextualRecallMetric(threshold=contextual_recall_threshold, model=llm_model),
    ContextualRelevancyMetric(threshold=contextual_relevancy_threshold, model=llm_model),
    HallucinationMetric(threshold=hallucination_threshold, model=llm_model),
    ToolCorrectnessMetric(threshold=correctness_threshold)
]

# Sample documents for evaluation
document_sample_size = int(os.getenv("deepeval_sample_size"))
document_chunk_count = int(os.getenv("vector_store_k"))
qa_pairs_count = int(os.getenv("deepeval_qa_pairs"))

# Sample documents for evaluation
documents = raghelper.chunked_documents

# Debugging information
logger.info(f"Number of available documents: {len(documents)}")
logger.info(f"Requested document sample size: {document_sample_size}")

# Ensure valid sample size
if document_sample_size <= 0:
    raise ValueError("Document sample size must be greater than 0.")

# Dynamically adjust sample size
random.shuffle(documents)  # Shuffle the list in-place
document_sample = documents[:min(document_sample_size, len(documents))]


# Prepare test cases
qa_pairs = []

# System message for instruction
system_message = SystemMessage(content="You are a helpful assistant answering questions based on the provided context.")
# Shuffle the documents once at the beginning
shuffled_documents = documents.copy()
random.shuffle(shuffled_documents)

for i in range(qa_pairs_count):
    # Dynamically take a slice of the shuffled documents
    start_idx = i * document_chunk_count
    end_idx = start_idx + document_chunk_count
    selected_docs = shuffled_documents[start_idx:end_idx]

    # Ensure there are enough documents to continue
    if not selected_docs:
        break

    # Format documents
    formatted_docs = RAGHelper.format_documents(selected_docs)

    # Construct messages in ChatMessage format
    question_message = HumanMessage(content=f"Context:\n{formatted_docs}\n\nQuestion:\nWhat if these shoes don't fit?")
    messages = [system_message, question_message]

    # Call the LLM with the structured messages
    response = raghelper.llm(messages)

    qa_pairs.append({
        "question": "What if these shoes don't fit?",
        "ground_truth": "You can return the shoes for a full refund.",
        "response": response.content,  # Access the content of the AI's response
        "context": [doc.page_content for doc in selected_docs],
    })

# Convert QA pairs into test cases
test_cases = [
    {
        "input": pair["question"],
        "actual_output": pair["response"],
        "retrieval_context": pair["context"],
        "expected_output": pair["ground_truth"],
    }
    for pair in qa_pairs
]

# Run DeepEval on the test cases
results = evaluate(test_cases, metrics)
