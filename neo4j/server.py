import csv
from neo4j import GraphDatabase
from flask import Flask, jsonify, request
from pyngrok import ngrok
import os

# Define the Graph_whisperer class to interact with Neo4j
class Graph_whisperer:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_instance(self, payload):
        with self.driver.session() as session:
            return session.execute_write(self._create_instance, payload)

    def add_document(self, payload):
        with self.driver.session() as session:
            return session.execute_write(self._add_document, payload)

    def get_meta_schema(self):
        """
        Retrieve detailed schema information, including node labels, properties, and relationship types.

        Returns:
            dict: A detailed schema including labels, properties, and relationship types.
        """
        with self.driver.session() as session:
            # Retrieve node labels and their properties
            nodes_query = """
            MATCH (n)
            UNWIND labels(n) AS label
            RETURN label, collect(DISTINCT keys(n)) AS properties
            """
            node_results = session.run(nodes_query)
            nodes = {}
            for record in node_results:
                label = record["label"]
                properties = set()
                for prop_list in record["properties"]:
                    properties.update(prop_list)
                nodes[label] = list(properties)

            # Retrieve relationship types and their properties
            rels_query = """
            MATCH ()-[r]->()
            RETURN type(r) AS type, collect(DISTINCT keys(r)) AS properties
            """
            rel_results = session.run(rels_query)
            relationships = {}
            for record in rel_results:
                rel_type = record["type"]
                properties = set()
                for prop_list in record["properties"]:
                    properties.update(prop_list)
                relationships[rel_type] = list(properties)

            return {"nodes": nodes, "relationships": relationships}

    def run_query(self, query):
        """
        Executes a Cypher query against the Neo4j database.

        Args:
            query (str): The Cypher query to execute.

        Returns:
            list: A list of query results, where each result is a dictionary.
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    @staticmethod
    def _create_instance(tx, payload):
        for instance in payload:
            tx.run(instance["query"], instance["parameters"])
        return instance

    @staticmethod
    def _add_document(self, csv_file_path):
        """
        Loads a CSV file into Neo4j by constructing and executing queries for each row.

        Args:
            csv_file_path (str): The path to the CSV file to be loaded.

        Returns:
            dict: A summary of the import process, including the number of records processed.
        """
        payloads = []
        try:
            with open(csv_file_path, mode="r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Construct the payload for each row
                    payloads.append(
                        {
                            "query": "MERGE (q:Quote {text: $quoteText}) "
                            "MERGE (t:Topic {name: $topicName}) "
                            "MERGE (q)-[:IS_PART_OF]->(t)",
                            "parameters": {
                                "quoteText": row.get("quoteText"),
                                "topicName": row.get("topicName"),
                            },
                        }
                    )
            # Execute all queries in the payload
            self._create_instance(self, payloads)
            return {
                "message": f"Successfully loaded {len(payloads)} records into Neo4j."
            }
        except Exception as e:
            return {"error": str(e)}


# Initialize Flask app
app = Flask(__name__)


neo4j_location = os.getenv('neo4j_location')
neo4j_user = os.getenv('neo4j_user')
neo4j_password = os.getenv('neo4j_password')
# Initialize Neo4j database connection
neo4j_db = Graph_whisperer(neo4j_location, neo4j_user, neo4j_password)


@app.route("/add_instances", methods=["POST"])
def add_instance():
    json_data = request.get_json()
    # print(json_data)
    try:
        # Use the json data to insert directly into Neo4j
        insert_result = neo4j_db.create_instance(json_data)
        return jsonify({"last inserted instance": insert_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/add_csv", methods=["POST"])
def add_csv():
    json_data = request.get_json()
    # print(json_data)
    try:
        # Use the json data to insert directly into Neo4j
        insert_result = neo4j_db.add_document(json_data)
        return jsonify({"last inserted instance": insert_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/close_db")
def close_db():
    try:
        neo4j_db.close()
        return jsonify({"message": "Database connection closed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/schema", methods=["GET"])
def get_meta_schema():
    try:
        schema = neo4j_db.get_meta_schema()
        app.logger.info(f"Retrieved schema: {schema}")
        return jsonify(schema)
    except Exception as e:
        app.logger.error(f"Error retrieving schema: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/run_query", methods=["POST"])
def run_query():
    try:
        # Extract the Cypher query from the request body
        query = request.json.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Execute the query
        results = neo4j_db.run_query(query)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    # # Set ngrok auth token and expose the app
    ngrok_token = os.getenv('ngrok_token')
    ngrok.set_auth_token(ngrok_token)  # Replace with your actual ngrok auth token
    public_url = ngrok.connect(4000)  # Expose port 5000
    print(f"ngrok tunnel available at: {public_url}")
    
    # Start Flask app
    app.run(host="0.0.0.0",port=4000)
