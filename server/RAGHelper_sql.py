import os

from langchain_huggingface.llms import HuggingFacePipeline
from PostgresText2SQLRetriever import PostgresText2SQLRetriever
from RAGHelper import RAGHelper
from RAGHelper_local import RAGHelperLocal
from transformers import pipeline
from langchain.schema.runnable import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain.chains.llm import LLMChain
from operator import itemgetter


class RAGHelperSQL(RAGHelperLocal):
    """
    RAGHelperSQL is a class that extends RAGHelperLocal to provide SQL-based retrieval
    capabilities using a Text2SQL component. It initializes the necessary components
    for language model pipelines, data loading, and retrieval chains.
    """

    def __init__(self, logger):
        """
        Overwrite the constructor of the RAGHelperLocal class to initialize the Text2SQL.
        However make use of the RAGHelper base class constructor.
        """
        RAGHelper.__init__(self, logger)
        self.logger = logger
        self.tokenizer, self.model = self._initialize_llm()
        self.llm = self._create_llm_pipeline()
        self.text2sql_uri = os.getenv("TEXT2SQL_DB_URI")
        # Initialize the Text2SQL retrieval component and assign it to the ensemble retriever attribute
        # ensuring that the LLM chain code in RAGHelperLocal can be reused
        self.ensemble_retriever = PostgresText2SQLRetriever(
            connection_uri=self.text2sql_uri, llama=self.llm
        )

        # Load the data
        self.load_data()

        # Create RAG chains
        self.rag_fetch_new_chain = self._create_rag_chain()
        self.rewrite_ask_chain, self.rewrite_chain = self._initialize_rewrite_chains()

    def load_data(self):
        """
        The load_data method currently only supports loading CSV files from the data directory.
        """
        csv_file_paths = os.listdir(self.data_dir)
        for csv_file_path in csv_file_paths:
            self.ensemble_retriever.setup_table(self.data_dir + csv_file_path)
        self.ensemble_retriever.get_database_schema()

    def _create_llm_pipeline(self):
        """
        Creates the huggingface pipeline based on the LLama LLM also used for the generative part of this RAG framework.
        The pipeline differs slightly as return_full_text is set to False.
        """
        text_generation_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=float(os.getenv("temperature")),
            repetition_penalty=float(os.getenv("repetition_penalty")),
            return_full_text=False,
            max_new_tokens=int(os.getenv("max_new_tokens")),
            model_kwargs={
                "device_map": "auto",
            },
        )
        return HuggingFacePipeline(pipeline=text_generation_pipeline)