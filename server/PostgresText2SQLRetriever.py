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
from pydantic import Field, BaseModel
from langchain_huggingface.llms import HuggingFacePipeline
import sqlparse


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SQLQuery(BaseModel):
    table: str          # The name of the table being queried.
    columns: List[str]  # The list of column names to include in the SELECT statement.
    conditions: str     # The WHERE clause conditions for filtering data.


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


def validate_sql(sql_query):
    logger.info(f"Validating SQL: {sql_query}")
    try:
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            raise ValueError("Invalid SQL")
        return True
    except Exception as e:
        return False


import re

def extract_sql(output):
    logger.info(f"Extracting SQL from: {output}")
    match = re.search(r'```sql\s*(.*?)\s*```', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        # Fall back to other methods if no delimiters are found
        return output.strip()



class PostgresText2SQLRetriever(BaseRetriever):
    connection_uri: str
    prompt: str = Field(..., env="text2sql_prompt")
    conn: psycopg2.extensions.connection = None
    cur: psycopg2.extensions.cursor = None
    db_schema: str = None
    llama: HuggingFacePipeline

    def __init__(self, **data):
        super().__init__(**data)
        logger.info(f"Connection URI: {self.connection_uri}")
        self.conn = psycopg2.connect(self.connection_uri)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.setup_database()
        self.llama = self.llama
        self.prompt = os.getenv("text2sql_prompt")
        self.db_schema = None

    def setup_database(self):
        new_db_name = "text2sql"
        self.cur.execute(
            sql.SQL("SELECT 1 FROM pg_database WHERE datname = {}").format(sql.Literal(new_db_name))
        )
        exists = self.cur.fetchone()
        if not exists:
            self.cur.execute(
            sql.SQL("CREATE DATABASE {}").format(sql.Identifier(new_db_name))
            )

    def setup_table(self, csv_file_path):
        table_name = os.path.splitext(os.path.basename(csv_file_path))[0].lower()
        # Check if the table already exists
        self.cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """, (table_name,))
        table_exists = self.cur.fetchone()[0]
        
        if not table_exists:
            print(f"Table '{table_name}' does not exist. Creating...")
            # Read the CSV header to infer column names
            with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                
                # Dynamically create column definitions (all as TEXT for simplicity)
                columns = [f"{col.strip()} TEXT" for col in header]
                create_table_query = sql.SQL("""
                    CREATE TABLE {table} (
                        {fields}
                    );
                """).format(
                    table=sql.Identifier(table_name),
                    fields=sql.SQL(", ").join(map(sql.SQL, columns))
                )
                
                # Execute the CREATE TABLE query
                self.cur.execute(create_table_query)
                print(f"Table '{table_name}' created successfully.")
        
            # Use COPY to load data from the CSV into the table
            with open(csv_file_path, 'r', encoding='utf-8', errors='ignore') as csvfile:
                self.cur.copy_expert(
                    sql.SQL("COPY {} FROM STDIN WITH CSV HEADER").format(sql.Identifier(table_name)),
                    csvfile
                )
            print(f"Data loaded into '{table_name}' successfully.")
    
        # Commit the transaction
        self.conn.commit()

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        sql_query = self._compute_query(query)
        # Get data with a prompt and return the result as a json object
        logger.info(f"SQL Query: {sql_query}")
        if not validate_sql(sql_query):
            return []
        
        self.cur.execute(sql_query)
        rows = self.cur.fetchmany(50)
        documents = self._format_results_as_documents(rows)
        logger.info(f"Documents: {documents}")
        return documents

    def _format_results_as_documents(self, results, sql_query):
        # We can adjust this method in order to format the context of the retrieved data
        documents = []
        for row in results:
            content = str(row)
            metadata = {sql_query: content}
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def _compute_query(self, prompt):
        input_prompt = self.prompt.format(query=prompt, schema=self.db_schema)
        logger.info(f"Input Prompt: {input_prompt}")
        generated_text = self.llama(input_prompt)

        # Extract the SQL query from the generated text
        sql_query = extract_sql(generated_text)        

        return sql_query
    
    def get_database_schema(self):
        schema = {}
    
        query = """
        SELECT
            table_schema,
            table_name,
            column_name,
            data_type
        FROM
            information_schema.columns
        WHERE
            table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY
            table_schema,
            table_name,
            ordinal_position;
        """
        
        # Execute the query
        self.cur.execute(query)
        
        # Process the results
        rows = self.cur.fetchall()
        for row in rows:
            table_name = row[1]
            column_name = row[2]
            data_type = row[3]
            
            if table_name not in schema:
                schema[table_name] = []
            schema[table_name].append({"column_name": column_name, "data_type": data_type})
        logger.info(schema)
        self.db_schema = self.format_schema(schema)

    @staticmethod
    def format_schema(input_data):
        result = "Schema:\n"
        for table_name, columns in input_data.items():
            # Get column names as a comma-separated string
            column_names = ", ".join([col['column_name'] for col in columns])
            # Append the formatted table information to the result
            result += f"- Table: {table_name} ({column_names})\n"
        return result

    def close(self):
        self.cur.close()
        self.conn.close()


#uri = "postgresql://user:pass@localhost:5432/text2sql"
#retriever = PostgresText2SQLRetriever(connection_uri=uri)
#retriever.setup_table("/home/markiemark/JADS/NLP/assignment3/RAGMeUp/data/StudentGradesAndPrograms.csv")

#schema = retriever.get_database_schema()
#print(type(schema))
#print(schema)
