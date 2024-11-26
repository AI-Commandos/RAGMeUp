from neo4j import GraphDatabase
from flask import Flask, jsonify, request
from pyngrok import ngrok

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
            return session.execute_write(self._add_document,payload)
    
    def update_instance(self, message):
        with self.driver.session() as session:
            return session.execute_write(self._get_or_create_greeting, message)

    def delete_instance(self, message):
        with self.driver.session() as session:
            return session.execute_write(self._get_or_create_greeting, message)

    @staticmethod
    def _create_instance(tx, payload):
        for instance in payload:
            tx.run(
                    instance['query'],
                    instance['parameters']
                )
        return instance
    
    @staticmethod
    def _add_document(tx, payload):
        #change the code below, this is copied as an examle from the create_instance
        for instance in payload:
            tx.run(
                    instance['query'],
                    instance['parameters']
                )
        return instance


# Initialize Flask app
app = Flask(__name__)

# Initialize Neo4j database connection
neo4j_db = Graph_whisperer("bolt://localhost:7687", "neo4j", "TOPICdb1")

@app.route("/add_instances",methods=['POST'])
def add_instance():
    json_data = request.get_json()
    # print(json_data)
    try:
        # Use the json data to insert directly into Neo4j
        insert_result = neo4j_db.create_instance(json_data)
        return jsonify({"last inserted instance": insert_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/add_csv",methods=['POST'])
def add_csv():
    json_data = request.get_json()
    # print(json_data)
    try:
        # Use the json data to insert directly into Neo4j
        insert_result = neo4j_db.add_document(json_data)
        return jsonify({"last inserted instance": insert_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route("/bak")
# def home():
#     try:
#         # Retrieve or create greeting message from Neo4j
#         greeting = neo4j_db.get_or_create_greeting("Hello, Neo4j!")
#         return jsonify({"greeting": greeting})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route("/close_db")
def close_db():
    try:
        neo4j_db.close()
        return jsonify({"message": "Database connection closed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Set ngrok auth token and expose the app
    ngrok.set_auth_token("")  # Replace with your actual ngrok auth token
    public_url = ngrok.connect(5000)  # Expose port 5000
    print(f"ngrok tunnel available at: {public_url}")

    # Start Flask app
    app.run(port=5000)

# from flask import Flask
# from pyngrok import ngrok

# # Initialize Flask app
# app = Flask(__name__)

# # Define Flask route
# @app.route("/")
# def home():
#     return "Hello, Flask with ngrok!"

# if __name__ == "__main__":
#     # Step 3: Authenticate ngrok with your auth token
#     ngrok.set_auth_token("")  # Replace with your actual auth token

#     # Step 4: Start ngrok tunnel and integrate with Flask
#     public_url = ngrok.connect(5000)  # Expose port 5000
#     print(f"ngrok tunnel available at: {public_url}")

#     # Start Flask app
#     app.run(port=5000)