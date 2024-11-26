import os

from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from PostgresText2SQLRetriever import PostgresText2SQLRetriever
from provenance import DocumentSimilarityAttribution
from RAGHelper import RAGHelper
from RAGHelper_local import RAGHelperLocal


class RAGHelperSQL(RAGHelperLocal):
    def __init__(self, logger):
        super().__init__(logger)
        self.logger = logger
        self.tokenizer, self.model = self._initialize_llm()
        self.llm = self._create_llm_pipeline()
        self.text2sql_uri = os.getenv("TEXT2SQL_DB_URI")
        self.ensemble_retriever = PostgresText2SQLRetriever(connection_uri=self.text2sql_uri)

        # Load the data
        self.load_data()

        # Create RAG chains
        self.rag_fetch_new_chain = self._create_rag_chain()
        self.rewrite_ask_chain, self.rewrite_chain = self._initialize_rewrite_chains()

        # Initialize provenance method
        self.attributor = DocumentSimilarityAttribution() if os.getenv("provenance_method") == "similarity" else None

    def load_data(self):
        csv_file_paths = os.listdir(self.data_dir)
        for csv_file_path in csv_file_paths:
            self.ensemble_retriever.setup_table(csv_file_path)
