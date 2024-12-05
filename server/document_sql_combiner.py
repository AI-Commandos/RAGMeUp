class DocumentSQLCombiner:
    @staticmethod
    def combine(retrieved_docs, sql_results):
        combined = []
        for doc in retrieved_docs:
            combined.append({
                "type": "document",
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        for result in sql_results:
            combined.append({
                "type": "sql_result",
                "content": result,
                "metadata": {}
            })
        return combined
