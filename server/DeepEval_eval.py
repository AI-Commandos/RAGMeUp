import random
import logging
from dotenv import load_dotenv
import os
from random import sample

from RAGHelper_cloud import RAGHelperCloud
from RAGHelper import RAGHelper

from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    HallucinationMetric,
    RagasMetric,
    ToolCorrectnessMetric,
)

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
ragas_threshold = float(os.getenv("deepeval_ragas_threshold", 0.5))

# Set the LLM model
llm_model = os.getenv("llm_model")

# Define metrics dynamically
metrics = [
    AnswerRelevancyMetric(threshold=answer_relevancy_threshold, model=llm_model),
    FaithfulnessMetric(threshold=faithfulness_threshold, model=llm_model),
    ContextualPrecisionMetric(threshold=contextual_precision_threshold, model=llm_model),
    ContextualRecallMetric(threshold=contextual_recall_threshold, model=llm_model),
    ContextualRelevancyMetric(threshold=contextual_relevancy_threshold, model=llm_model),
    HallucinationMetric(threshold=hallucination_threshold),
    ToolCorrectnessMetric(),
    RagasMetric(threshold=ragas_threshold, model=llm_model),
]

# Sample documents for evaluation
document_sample_size = int(os.getenv("ragas_sample_size"))
document_chunk_count = int(os.getenv("vector_store_k"))
qa_pairs_count = int(os.getenv("ragas_qa_pairs"))

documents = raghelper.chunked_documents
document_sample = sample(documents, document_sample_size)

# Prepare test cases
qa_pairs = []
for _ in range(qa_pairs_count):
    selected_docs = random.sample(document_sample, document_chunk_count)
    formatted_docs = RAGHelper.format_documents(selected_docs)

    # Generate a sample question and response
    question = "What if these shoes don't fit?"  # Replace with dynamic question generation if needed
    ground_truth = "You can return the shoes for a full refund."  # Replace with expected answer
    response = raghelper.llm({"context": formatted_docs, "question": question})["text"]

    qa_pairs.append({
        "question": question,
        "ground_truth": ground_truth,
        "response": response,
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

# Print results
for result in results:
    print(f"Metric: {result['metric']}")
    print(f"Score: {result['score']}")
    print(f"Reason: {result['reason']}")
