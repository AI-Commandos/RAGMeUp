import psycopg2
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#from text2_sql import TextToSQL

class TextToSQL:
    def __init__(self, model_name="suriya7/t5-base-text-to-sql", db_uri=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.db_uri = db_uri
        # Configure logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)


    def translate(self, user_query):
        self.logger.info(f"Translating user query: {user_query}")
        input_text = f"translate English to SQL: {user_query}"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
        sql_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated SQL query: {sql_query}")
        return sql_query

    def execute(self, sql_query):
        try:
            print(f"Executing SQL query: {sql_query}")
            connection = psycopg2.connect(self.db_uri)
            cursor = connection.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            connection.close()
            print(f"SQL query executed successfully, results: {results}")
            return results
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return {"error": str(e)}
