import random
import logging
from dotenv import load_dotenv
import os
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
    ToolCorrectnessMetric
)
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from deepeval.test_case import LLMTestCase

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
llm_model = os.getenv("llm_model")

# Define default metrics
metrics = [
    AnswerRelevancyMetric(threshold=answer_relevancy_threshold, model=llm_model),
    FaithfulnessMetric(threshold=faithfulness_threshold, model=llm_model),
    ContextualPrecisionMetric(threshold=contextual_precision_threshold, model=llm_model),
    ContextualRecallMetric(threshold=contextual_recall_threshold, model=llm_model),
    ContextualRelevancyMetric(threshold=contextual_relevancy_threshold, model=llm_model),
    HallucinationMetric(threshold=hallucination_threshold, model=llm_model),
    ToolCorrectnessMetric(threshold=correctness_threshold)
]

# Define custom metrics
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class CounterfactualErrorHandling(BaseMetric):
    """Custom metric to evaluate counterfactual error handling in LLMs."""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.name = "CounterfactualHandling Metrics"
        self.evaluation_model = "gpt-3.5-turbo"
        self.include_reason = True
        self.strict_mode = True
        self.async_mode = False

    @property
    def __name__(self):
        return self.name

    def measure(self, test_case: LLMTestCase):
        question = test_case.input.lower()
        if "if" in question and "not" in question:
            score = 1.0 if "error" not in test_case.actual_output.lower() else 0.0
        else:
            score = 0.0
        self.score = score
        self.passed = score >= self.threshold
        if self.include_reason:
            self.reason = "Handled counterfactual correctly." if self.passed else "Failed to handle counterfactual."

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.passed


class LongDistanceInformationExtraction(BaseMetric):
    """Custom metric to evaluate long-distance information extraction."""

    def __init__(self, threshold: float = 0.5, distance_threshold: int = 50):
        super().__init__()
        self.threshold = threshold
        self.distance_threshold = distance_threshold
        self.name = "LongDistance Information Extraction Metrics"
        self.evaluation_model = "gpt-3.5-turbo"
        self.include_reason = True
        self.strict_mode = True
        self.async_mode = False

    @property
    def __name__(self):
        return self.name

    def measure(self, test_case: LLMTestCase):
        context = test_case.retrieval_context
        input_text = test_case.input.lower()
        if len(context) > self.distance_threshold:
            relevant_content = any(input_text in doc.lower() for doc in context)
            score = 1.0 if relevant_content else 0.0
        else:
            score = 0.0
        self.score = score
        self.passed = score >= self.threshold
        if self.include_reason:
            self.reason = "Successfully extracted long-distance information." if self.passed else "Failed to extract long-distance information."

    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.passed


# Add custom metrics to the list
metrics.append(CounterfactualErrorHandling(threshold=0.7))
metrics.append(LongDistanceInformationExtraction(threshold=0.7, distance_threshold=50))

# Parameters from environment variables
deepeval_use_n_documents = int(os.getenv("vector_store_k"))
if os.getenv("rerank") == "True":
    deepeval_use_n_documents = int(os.getenv("rerank_k"))
end_string = os.getenv("llm_assistant_token")
deepeval_sample_size = int(os.getenv("deepeval_sample_size", 10))
deepeval_qa_pairs = int(os.getenv("deepeval_qa_pairs", 5))

# Load and sample documents
documents = raghelper.chunked_documents
random.shuffle(documents)
document_sample = documents[:min(deepeval_sample_size, len(documents))]

# Prepare question generation template
if use_cloud:
    question_thread = [
        ('system', os.getenv('deepeval_question_instruction')),
        ('human', os.getenv('deepeval_question_query'))
    ]
    question_prompt = ChatPromptTemplate.from_messages(question_thread)
else:
    question_thread = [
        {'role': 'system', 'content': os.getenv("deepeval_question_instruction")},
        {'role': 'user', 'content': os.getenv("deepeval_question_query")}
    ]
    question_prompt = ChatPromptTemplate.from_messages(question_thread)

rag_question = question_prompt | raghelper.llm

# Prepare answer generation template
if use_cloud:
    answer_thread = [
        ('system', os.getenv('deepeval_answer_instruction')),
        ('human', os.getenv('deepeval_answer_query'))
    ]
    answer_prompt = ChatPromptTemplate.from_messages(answer_thread)
else:
    answer_thread = [
        {'role': 'system', 'content': os.getenv("deepeval_answer_instruction")},
        {'role': 'user', 'content': os.getenv("deepeval_answer_query")}
    ]
    answer_prompt = ChatPromptTemplate.from_messages(answer_thread)

rag_answer = answer_prompt | raghelper.llm

# Generate QA pairs
qa_pairs = []
for i in range(deepeval_qa_pairs):
    random.shuffle(document_sample)
    selected_docs = document_sample[:min(deepeval_use_n_documents, len(document_sample))]
    formatted_docs = RAGHelper.format_documents(selected_docs)

    # Generate question
    question_chain = {"context": RunnablePassthrough()} | rag_question
    question_response = question_chain.invoke(formatted_docs)

    if use_cloud:
        question = (
                getattr(question_response, "content", None)
                or question_response.get("answer")
                or question_response.get("content")
        )
    else:
        question = question_response.split(end_string)[-1]

    # Generate answer
    answer_chain = {"context": RunnablePassthrough(), "question": RunnablePassthrough()} | rag_answer
    answer_response = answer_chain.invoke({"context": formatted_docs, "question": question})

    if use_cloud:
        answer = (
                getattr(answer_response, "content", None)
                or answer_response.get("answer")
                or answer_response.get("content")
        )
    else:
        answer = answer_response.split(end_string)[-1]

    qa_pairs.append({
        "question": question,
        "ground_truth": answer,
        "context": [doc.page_content for doc in selected_docs],
    })

logger.info(f"Generated QA Pairs: {qa_pairs}")
for i, pair in enumerate(qa_pairs):
    if not pair["question"] or not pair["ground_truth"] or not pair["context"]:
        logger.error(f"Incomplete QA pair at index {i}: {pair}")

# Evaluate with DeepEval metrics
test_cases = [
    LLMTestCase(
        input=pair["question"],
        actual_output=pair["ground_truth"],
        expected_output=pair["ground_truth"],
        retrieval_context=pair["context"],
        context=pair["context"],
        tools_called=[],
        expected_tools=["WebSearch"],
    )
    for pair in qa_pairs
]

logger.info(f"Prepared Test Cases: {test_cases}")
results = evaluate(test_cases, metrics)
logger.info(f"Evaluation Results: {results}")

