import os
from tqdm import tqdm

from langchain_core.documents.base import Document
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_milvus.vectorstores import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

from lxml import etree

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

import re
import pickle

# Make documents look a bit better than default
def formatDocuments(docs):
    doc_strings = []
    for doc in docs:
        metadata_string = ", ".join([f"{md}: {doc.metadata[md]}" for md in doc.metadata.keys()])
        doc_strings.append(f"Content: {doc.page_content}\nMetadata: {metadata_string}")
    return "\n\n".join(doc_strings)

def getFilenames(docs):
    return [{'s': doc.metadata['source'], 'c': doc.page_content} for doc in docs if 'source' in doc.metadata]

def combine_results(inputs):
    if "context" in inputs.keys() and "docs" in inputs.keys():
        return {
            "answer": inputs["answer"],
            "docs": inputs["docs"],
            "context": inputs["context"],
            "question": inputs["question"]
        }
    else:
        return {
            "answer": inputs["answer"],
            "question": inputs["question"]
        }

# Capture the context of the retriever
class CaptureContext(RunnablePassthrough):
    def __init__(self):
        self.captured_context = None
    
    def run(self, input_data):
        self.captured_context = input_data['context']
        return input_data

class RAGHelperCloud:
    def __init__(self, logger):
        if os.getenv("use_openai") == "True":
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        elif os.getenv("use_gemini") == "True":
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
        elif os.getenv("use_azure") == "True":
            self.llm = AzureChatOpenAI(
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            )

        # Set up embedding handling for vector store
        if os.getenv('force_cpu') == "True":
            model_kwargs = {
                'device': 'cpu'
            }
        else:
            model_kwargs = {
                'device': 'cuda'
            }
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv('embedding_model'),
            model_kwargs=model_kwargs
        )

        # Load the data
        self.loadData()

        # Create the RAG chain for determining if we need to fetch new documents
        rag_thread = [
            ('system', os.getenv('rag_fetch_new_instruction')),
            ('human', os.getenv('rag_fetch_new_question'))
        ]
        rag_prompt = ChatPromptTemplate.from_messages(rag_thread)
        rag_llm_chain = rag_prompt | self.llm
        self.rag_fetch_new_chain = (
            {"question": RunnablePassthrough()} |
            rag_llm_chain
        )

        # Also create the rewrite loop LLM chain, if need be
        self.rewrite_ask_chain = None
        self.rewrite_chain = None
        if os.getenv("use_rewrite_loop") == "True":
            # First the chain to ask the LLM if a rewrite would be required
            rewrite_ask_thread = [
                ('system', os.getenv('rewrite_query_instruction')),
                ('human', os.getenv('rewrite_query_question'))
            ]
            rewrite_ask_prompt = ChatPromptTemplate.from_messages(rewrite_ask_thread)
            rewrite_ask_llm_chain = rewrite_ask_prompt | self.llm
            context_retriever = self.ensemble_retriever
            if os.getenv("rerank") == "True":
                context_retriever = self.rerank_retriever
            self.rewrite_ask_chain = (
                {"context": context_retriever | formatDocuments, "question": RunnablePassthrough()} |
                rewrite_ask_llm_chain
            )

            # Next the chain to ask the LLM for the actual rewrite(s)
            rewrite_thread = [
                ('human', os.getenv('rewrite_query_prompt'))
            ]
            rewrite_prompt = ChatPromptTemplate.from_messages(rewrite_thread)
            rewrite_llm_chain = rewrite_prompt | self.llm
            self.rewrite_chain = (
                {"question": RunnablePassthrough()} |
                rewrite_llm_chain
            )

    # Loads the data and chunks it into an ensemble retriever
    def loadData(self):
        # Check if we have our files chunked already
        sparse_db_path = f"{os.getenv('vector_store_path')}_sparse.pickle"
        if os.path.exists(sparse_db_path):
            with open(sparse_db_path, 'rb') as f:
                self.chunked_documents = pickle.load(f)
        else:
            docs = []
            data_dir = os.getenv('data_directory')
            file_types = os.getenv("file_types").split(",")
            if "pdf" in file_types:
                loader = PyPDFDirectoryLoader(data_dir)
                docs = docs + loader.load()
            # Load JSON
            if "json" in file_types:
                text_content = True
                if os.getenv("json_text_content") == 'False':
                    text_content = False
                loader_kwargs = {
                    'jq_schema': os.getenv("json_schema"),
                    'text_content': text_content
                }
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.json",
                    loader_cls=JSONLoader,
                    loader_kwargs=loader_kwargs,
                )
                docs = docs + loader.load()
            # Load CSV
            if "csv" in file_types:
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.csv",
                    loader_cls=CSVLoader,
                )
                docs = docs + loader.load()
            # Load MS Word
            if "docx" in file_types:
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.docx",
                    loader_cls=Docx2txtLoader,
                )
                docs = docs + loader.load()
            # Load MS Excel
            if "xlsx" in file_types:
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.xlsx",
                    loader_cls=UnstructuredExcelLoader,
                )
                docs = docs + loader.load()
            # Load MS PPT
            if "pptx" in file_types:
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.pptx",
                    loader_cls=UnstructuredPowerPointLoader,
                )
                docs = docs + loader.load()
            # Load XML, which is nasty
            if "xml" in file_types:
                loader = DirectoryLoader(
                    path=data_dir,
                    glob="*.xml",
                    loader_cls=TextLoader,
                )
                xmldocs = loader.load()
                newdocs = []
                for index, doc in enumerate(xmldocs):
                    xmltree = etree.fromstring(doc.page_content.encode('utf-8'))
                    elements = xmltree.xpath(os.getenv("xml_xpath"))
                    elements = [etree.tostring(element, pretty_print=True).decode() for element in elements]
                    metadata = doc.metadata
                    metadata['index'] = index
                    newdocs = newdocs + [Document(page_content=doc, metadata=metadata) for doc in elements]
                docs = docs + newdocs

            newdocs = []
            for index, doc in enumerate(xmldocs):
                xmltree = etree.fromstring(doc.page_content.encode('utf-8'))
                elements = xmltree.xpath(os.getenv("xml_xpath"))
                elements = [etree.tostring(element, pretty_print=True).decode() for element in elements]
                metadata = doc.metadata
                metadata['index'] = index
                newdocs = newdocs + [Document(page_content=doc, metadata=metadata) for doc in elements]
            docs = docs + newdocs

            #if os.getenv('splitter') == 'RecursiveCharacterTextSplitter':
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(os.getenv('chunk_size')),
                chunk_overlap=int(os.getenv('chunk_overlap')),
                length_function=len,
                keep_separator=True,
                separators=[
                    "\n \n",
                    "\n\n",
                    "\n",
                    ".",
                    "!",
                    "?",
                    " ",
                    ",",
                    "\u200b",  # Zero-width space
                    "\uff0c",  # Fullwidth comma
                    "\u3001",  # Ideographic comma
                    "\uff0e",  # Fullwidth full stop
                    "\u3002",  # Ideographic full stop
                    "",
                ],
            )

            self.chunked_documents = self.text_splitter.split_documents(docs)
            # Store these too, for our sparse DB
            with open(f"{os.getenv('vector_store_path')}_sparse.pickle", 'wb') as f:
                pickle.dump(self.chunked_documents, f)

        vector_store_path = os.getenv('vector_store_path')
        if os.path.exists(vector_store_path):
            self.db = Milvus.from_documents(
                [], self.embeddings,
                connection_args={"uri": vector_store_path},
            )
        else:
            # Load chunked documents into the Milvus index
            self.db = Milvus.from_documents(
                [], self.embeddings,
                drop_old=True,
                connection_args={"uri": vector_store_path},
            )
            # Add the documents 1 by 1 so we can track progress
            with tqdm(total=len(self.chunked_documents), desc="Vectorizing documents") as pbar:
                for d in self.chunked_documents:
                    self.db.add_documents([d])
                    pbar.update(1)

        # Now the BM25 retriever
        bm25_retriever = BM25Retriever.from_texts(
            [x.page_content for x in self.chunked_documents],
            metadatas=[x.metadata for x in self.chunked_documents]
        )
        # Set up the vector retriever
        retriever=self.db.as_retriever(
            search_type="mmr", search_kwargs = {'k': int(os.getenv("vector_store_k"))}
        )

        # Now combine them to do hybrid retrieval
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )
        # Set up the reranker
        self.rerank_retriever = None
        if os.getenv("rerank") == "True":
            compressor = FlashrankRerank(top_n=int(os.getenv("rerank_k")))
            self.rerank_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=self.ensemble_retriever
            )

    def handle_rewrite(self, user_query):
        # Check if we even need to rewrite or not
        if os.getenv("use_rewrite_loop") == "True":
            # Ask the LLM if we need to rewrite
            response = self.rewrite_ask_chain.invoke(user_query)
            if hasattr(response, 'content'):
                response = response.content
            elif hasattr(response, 'answer'):
                response = response.answer
            elif 'answer' in response:
                response = response["answer"]
            response = re.sub(r'\W+ ', '', response)
            if response.lower().endswith('yes') or response.lower().endswith('ja'):
                # Start the rewriting into different alternatives
                response = self.rewrite_chain.invoke(user_query)

                if hasattr(response, 'content'):
                    response = response.content
                elif hasattr(response, 'answer'):
                    response = response.answer
                elif 'answer' in response:
                    response = response["answer"]

                # Show be split by newlines
                return response
            else:
                # We do not need to rewrite
                return user_query
        else:
            return user_query

    # Main function to handle user interaction
    def handle_user_interaction(self, user_query, history):
        if len(history) == 0:
            fetch_new_documents = True
        else:
            # Prompt for LLM
            response = self.rag_fetch_new_chain.invoke(user_query)
            if hasattr(response, 'content'):
                response = response.content
            elif hasattr(response, 'answer'):
                response = response.answer
            elif 'answer' in response:
                response = response["answer"]
            response = re.sub(r'\W+ ', '', response)
            if response.lower().endswith('yes') or response.lower().endswith('ja'):
                fetch_new_documents = True
            else:
                fetch_new_documents = False

        # Create prompt template based on whether we have history or not
        thread = [(x["role"], x["content"].replace("{", "(").replace("}", ")")) for x in history]
        if len(thread) == 0:
            thread.append(('system', os.getenv('rag_instruction')))
            thread.append(('human', os.getenv('rag_question_initial')))
        else:
            thread.append(('human', os.getenv('rag_question_followup')))

        # Create prompt from prompt template
        prompt = ChatPromptTemplate.from_messages(thread)

        # Create llm chain
        llm_chain = prompt | self.llm
        if fetch_new_documents:
            # Rewrite the question if needed
            user_query = self.handle_rewrite(user_query)
            context_retriever = self.ensemble_retriever
            if os.getenv("rerank") == "True":
                context_retriever = self.rerank_retriever
            
            retriever_chain = {
                "docs": context_retriever | getFilenames,
                "context": context_retriever | formatDocuments,
                "question": RunnablePassthrough()
            }
            llm_chain = prompt | self.llm | StrOutputParser()
            rag_chain = (
                retriever_chain
                | RunnablePassthrough.assign(
                    answer=lambda x: llm_chain.invoke(
                        {"docs": x["docs"], "context": x["context"], "question": x["question"]}
                    ))
                | combine_results
            )
        else:
            retriever_chain = {
                "question": RunnablePassthrough()
            }
            llm_chain = prompt | self.llm | StrOutputParser()
            rag_chain = (
                retriever_chain
                | RunnablePassthrough.assign(
                    answer=lambda x: llm_chain.invoke(
                        {"question": x["question"]}
                    ))
                | combine_results
            )

        # Invoke RAG pipeline
        reply = rag_chain.invoke(user_query)
        return (thread, reply)

    def addDocument(self, filename):
        if filename.lower().endswith('pdf'):
            docs = PyPDFLoader(filename).load()
        if filename.lower().endswith('json'):
            docs = JSONLoader(
                file_path = filename,
                jq_schema = os.getenv("json_schema"),
                text_content = os.getenv("json_text_content") == "True",
            ).load()
        if filename.lower().endswith('csv'):
            docs = CSVLoader(filename).load()
        if filename.lower().endswith('docx'):
            docs = Docx2txtLoader(filename).load()
        if filename.lower().endswith('xlsx'):
            docs = UnstructuredExcelLoader(filename).load()
        if filename.lower().endswith('pptx'):
            docs = UnstructuredPowerPointLoader(filename).load()

        # Skills and personality are global and don't work on chunks, so do them first
        new_docs = []
        for doc in docs:
            # Get the skills by using the LLM and attach to the doc
            skills = self.parseCV(doc)
            doc.metadata['skills'] = skills
            # Also get the personality
            doc.metadata['personality'] = self.personality_predictor.predict(doc)
            new_docs.append(doc)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv('chunk_size')),
            chunk_overlap=int(os.getenv('chunk_overlap')),
            length_function=len,
            keep_separator=True,
            separators=[
                "\n \n",
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                " ",
                ",",
                "\u200b",  # Zero-width space
                "\uff0c",  # Fullwidth comma
                "\u3001",  # Ideographic comma
                "\uff0e",  # Fullwidth full stop
                "\u3002",  # Ideographic full stop
                "",
            ],
        )
        new_chunks = self.text_splitter.split_documents(new_docs)

        self.chunked_documents = self.chunked_documents + new_chunks
        # Store these too, for our sparse DB
        with open(f"{os.getenv('vector_store_path')}_sparse.pickle", 'wb') as f:
            pickle.dump(self.chunked_documents, f)

        # Add to vector DB
        self.db.add_documents(new_chunks)

        # Add to BM25
        bm25_retriever = BM25Retriever.from_texts(
            [x.page_content for x in self.chunked_documents],
            metadatas=[x.metadata for x in self.chunked_documents]
        )

        # Update full retriever too
        retriever = self.db.as_retriever(search_type="mmr", search_kwargs = {'k': int(os.getenv('vector_store_k'))})
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )

        if os.getenv("rerank") == "True":
            compressor = FlashrankRerank(top_n=int(os.getenv("rerank_k")))
            self.rerank_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=self.ensemble_retriever
            )