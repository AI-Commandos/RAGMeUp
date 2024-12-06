import json
import operator
import pprint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from langchain import hub
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from dotenv import load_dotenv
import logging
import torch
from RAGHelper import RAGHelper
import os
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.schema import Document
from langgraph.types import Command


def _initialize_embeddings():
    """Initialize and return embeddings for vector storage."""
    model_kwargs = {
        'device': 'mps' if torch.backends.mps.is_available() else 'cuda' if os.getenv(
            'force_cpu') != "True" else 'cpu'
    }
    return HuggingFaceEmbeddings(
        model_name=os.getenv('embedding_model'),
        model_kwargs=model_kwargs
    )
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv(dotenv_path=".env")
rag_helper = RAGHelper(logger=logger)
rag_helper.embeddings = _initialize_embeddings()
rag_helper.load_data()

local_llm = "llama3.2"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

db = SQLDatabase.from_uri("sqlite:///data/chinook.db")
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
system_message = prompt_template.format(dialect="SQLite", top_k=10)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Milvus.from_documents(
    documents=doc_splits,
    collection_name="rag_chroma",
    embedding=_initialize_embeddings(),
    connection_args={"uri": 'data2.db'},
)

retriever = vectorstore.as_retriever(k= 3)
retirever = rag_helper.rerank_retriever

# Prompt
router_instructions = """You are an expert at routing a user question to a vectorstore or sql database.

The vectorstore contains documents related to Natural language processing.

Use the vectorstore for questions on these topics. For all else, use sql.

Return JSON with single key, datasource, that is 'sql' or 'vectorstore' depending on the question."""


### Retrieval Grader

# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""


### Generate

# Prompt
rag_prompt = """You are an assistant for question-answering tasks. 

Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review the user question:

{question}

Provide an answer to this questions using only the above context. 

Use three sentences maximum and keep the answer concise.

Answer:"""


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


### Answer Grader

# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

### Question Re-writer



# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
     Return JSON with single key, improved_question, with the improved question."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
question_rewriter = re_write_prompt | llm_json_mode | StrOutputParser()

# @tool
# def web_search(query:str) -> int:
#     """Web search tool that returns web results based on the query."""
#     documents = [
#         Document(page_content="Dogs are domesticated mammals, not natural wild animals.", metadata={"source": "web1"}),
#         Document(page_content="The Labrador Retriever is one of the most popular dog breeds in the world.", metadata={"source": "web2"}),
#         Document(page_content="Dogs communicate through body language, barking, and tail wagging.", metadata={"source": "web3"}),
#         Document(page_content="Dogs are known as loyal and friendly companions to humans.", metadata={"source": "web4"}),
#     ]
#     return [doc for doc in documents if query.lower() in doc.page_content.lower()]

# web_search_tool = web_search


class GraphState(MessagesState):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """

    question: str  # User question
    generation: str  # LLM generation
    sql: bool  # Binary decision to run sql_search
    max_retries: int  # Max number of retries for answer generation
    answers: int  # Number of answers generated
    loop_step: Annotated[int, operator.add]
    documents: list[str] # List of retrieved documents


### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Write retrieved documents to documents key in state
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "sql": False}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return {"generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to trandsform the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    # loop_step = state.get("loop_step", 0)


    # Score each doc
    filtered_docs = []
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions)]
            + [HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)["binary_score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "generation": state.get("generation"),
        }

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question, "max_retries": 1}

# def web_search(state):
#     """
#     Web search based based on the question

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Appended web results to documents
#     """

#     print("---WEB SEARCH---")
#     question = state["question"]
#     documents = state.get("documents", [])

#     # Web search
#     docs = web_search_tool.invoke({"query": question})
#     web_results = "\n".join([d["page_content"] for d in docs])
#     web_results = Document(page_content=web_results)
#     documents.append(web_results)
#     return {"documents": documents}

def sql_search(state):
    """
    Query SQL database based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended SQL results
    """

    chat_executor = create_react_agent(
        llm, tools = tools, state_modifier=system_message
    )
    question = state["question"]
    response = chat_executor.invoke({"messages": [{"role": "user", "content": question}]})
    # response = chat_executor.invoke({"messages": [("user", state["question"])]})
    return {"question": state["question"], "generation": response, "sql": True}


### Edges


def route_question(state):
    """
    Route question to sql database or RAG

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    route_question = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=state["question"])]
    )
    source = json.loads(route_question.content)["datasource"]
    if source == "sql":
        print("---ROUTE QUESTION TO SQL SEARCH---")
        return "sql_search"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or transform the query

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if state["max_retries"]>=1:
        print("---DECISION: MAX RETRIES---")
        return "max_retries"
    elif state["sql"] == True:
        return "generate"

    elif not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def should_retrieve_initial_documents(state):
    """
    Determine whether to fetch new documents

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    documents = state["documents"]
    if not documents:
        print("---DECISION: RETRIEVE---")
        return "retrieve"
    else:
        print("---DECISION: GRADE---")
        return "grade_documents"

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("sql_search", sql_search)  # sql_search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("transform_query", transform_query)  # transform query
workflow.add_node("generate", generate)  # generate


def rag_start(state):
    """
    Start the RAG agent

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---START RAG---")
    return state

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "sql_search": "sql_search",
        "vectorstore": "rag_start",
    },
)
workflow.add_node("rag_start", rag_start)
workflow.add_edge("sql_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("transform_query", "retrieve")

workflow.add_conditional_edges(
    "rag_start",
    should_retrieve_initial_documents,
    {
        "retrieve": "retrieve",
        "grade_documents": "grade_documents",
    },
)
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "max_retries": END,
    },
)

# Compile
graph = workflow.compile()
png_graph = graph.get_graph().draw_mermaid_png()
with open("adaptive_rag.png", "wb") as f:
    f.write(png_graph)
    
if __name__ == "__main__":
    # Run
    inputs = {
        # "question": "How many artist in artists table?",
        # "question": "What is soccer?",
        "question": "Explain natural language processing?",
        # "question": "Is a labrador a dog?",
        "documents": [],
        "max_retries": 0,
    }
    for output in graph.stream(inputs):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    pprint.pprint(value["generation"])