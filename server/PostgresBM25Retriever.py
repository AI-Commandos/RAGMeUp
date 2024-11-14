import psycopg2
import psycopg2.extras
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
import uuid
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List
import re
import json
import os

class PostgresBM25Retriever(BaseRetriever):
    connection_uri: str
    table_name: str
    k: int
    conn: psycopg2.extensions.connection = None
    cur: psycopg2.extensions.cursor = None

    def __init__(self, **data):
        super().__init__(**data)
        self.conn = psycopg2.connect(self.connection_uri)
        self.cur = self.conn.cursor()
        self.setup_database()

    def setup_database(self):
        # Ensure pg_search extension is installed
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS pg_search;")
        
        # Create table if not exists
        self.cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    hash char(32) UNIQUE,
                    content TEXT,
                    metadata TEXT
                );""")
        
        # Create BM25 index
        self.cur.execute(fr"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = '{self.table_name}'
                ) THEN
                    CALL paradedb.create_bm25(
                        index_name => '{self.table_name}_bm25',
                        table_name => '{self.table_name}',
                        key_field => 'id',
                        text_fields => paradedb.field('content') || paradedb.field('metadata')
                    );
                END IF;
            END $$;
        """)
        self.conn.commit()

    def add_documents(self, documents: List[Document], ids: List[str] = None) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        if len(ids) != len(documents):
            raise ValueError("Number of ids must match number of documents")
        
        records = [
            (doc_id, doc.page_content, psycopg2.extras.Json(doc.metadata))
            for doc, doc_id in zip(documents, ids)
        ]

        psycopg2.extras.execute_batch(
            self.cur,
            f"""
                INSERT INTO {self.table_name} (hash, content, metadata)
                VALUES (%s, %s, %s)
                ON CONFLICT (hash) DO NOTHING
            """,
            records
        )
        
        self.conn.commit()
        return ids

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # Perform BM25 search using pg_search
        if os.getenv("use_re2") == "True":
            os.getenv("re2_prompt")
            index = query.find(f"\n{os.getenv('re2_prompt')}")
            query = query[:index]
        query = re.sub(r'[\(\):]', '', query)
        
        search_command = f"""
            SELECT 
                sparse_vectors.id, 
                sparse_vectors.content, 
                sparse_vectors.metadata, 
                paradedb.score(sparse_vectors.id) AS score_bm25
            FROM {self.table_name}
            WHERE content @@@ '{query}'
            ORDER BY score_bm25 DESC
            LIMIT {self.k};
        """
        self.cur.execute(search_command)
        
        results = self.cur.fetchall()
        
        return [Document(page_content=content, metadata={**json.loads(metadata), 'id': id, 'relevance_score': score}) for id, content, metadata, score in results]

    def delete(self, ids: List[str]) -> None:
        placeholders = ','.join(['%s'] * len(ids))
        self.cur.execute(f"DELETE FROM {self.table_name} WHERE id IN ({placeholders});", tuple(ids))
        self.cur.execute(f"VACUUM {self.table_name};")
        self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()