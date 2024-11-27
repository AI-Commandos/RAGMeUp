import csv
import json
import logging
import os
import re
import uuid
from typing import List

import psycopg2
import psycopg2.extras
import torch
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from psycopg2 import sql
from transformers import T5ForConditionalGeneration, T5Tokenizer


logger = logging.getLogger(__name__)


class PostgresText2SQLRetriever(BaseRetriever):
    connection_uri: str
    conn: psycopg2.extensions.connection = None
    cur: psycopg2.extensions.cursor = None

    def __init__(self, **data):
        super().__init__(**data)
        logger.info(f"Connection URI: {self.connection_uri}")
        self.conn = psycopg2.connect(self.connection_uri)
        self.cur = self.conn.cursor()
        self.setup_database()
        self.sql_generator = SQLGenerator()
        self.tables = os.getenv("tables")

    def setup_database(self):
        # Just simply setup a postgres database
        # Create a new database
        new_db_name = os.getenv("TEXT2SQL_DB_NAME")
        self.cur.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(new_db_name))
        )

    def setup_table(self, csv_file_path):
        # Create a table based on a CSV file
        table_name = csv_file_path.split("/")[-1].split(".")[0]

        with open(csv_file_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            columns = ", ".join([f"{header} TEXT" for header in headers])

            create_table_query = sql.SQL("CREATE TABLE {} ({})").format(
                sql.Identifier(table_name), sql.SQL(columns)
            )
            self.cur.execute(create_table_query)
            self.conn.commit()

            for row in reader:
                insert_query = sql.SQL("INSERT INTO {} VALUES ({})").format(
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(map(sql.Literal, row)),
                )
                self.cur.execute(insert_query)
            self.conn.commit()

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        sql_query = self.compute_query(query)
        # Get data with a prompt and return the result as a json object
        logger.info(f"SQL Query: {sql_query}")
        self.cur.execute(sql_query)
        rows = self.cur.fetchmany(50)
        documents = self._format_results_as_documents(rows)
        return documents

    def _format_results_as_documents(self, results):
        # We can adjust this method in order to format the context of the retrieved data
        documents = []
        for row in results:
            content = str(row)
            metadata = {}
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def compute_query(self, prompt):
        input_prompt = "tables:\n" + self.tables + "\n" + "query for:" + prompt
        generated_sql = self.sql_generator.generate_sql(input_prompt)
        return generated_sql

    def close(self):
        self.cur.close()
        self.conn.close()


class SQLGenerator:
    def __init__(self, model_name="cssupport/t5-small-awesome-text-to-sql"):
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate_sql(self, input_prompt):
        # Tokenize the input prompt
        inputs = self.tokenizer(
            input_prompt, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=512)

        # Decode the output IDs to a string (SQL query in this case)
        generated_sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_sql
