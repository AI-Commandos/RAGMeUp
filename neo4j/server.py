from neo4j import GraphDatabase
from flask import Flask, jsonify
from pyngrok import ngrok

# Define the HelloWorldExample class to interact with Neo4j
class HelloWorldExample:

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_or_create_greeting(self, message):
        with self.driver.session() as session:
            return session.execute_write(self._get_or_create_greeting, message)

    @staticmethod
    def _get_or_create_greeting(tx, message):
        # Check if a Greeting node exists
        result = tx.run("MATCH (a:Greeting) RETURN a.message LIMIT 1")
        record = result.single()

        # If no node exists, create one
        if not record:
            result = tx.run(
                "CREATE (a:Greeting) "
                "SET a.message = $message "
                "RETURN a.message + ', from node ' + id(a)",
                message=message
            )
            return result.single()[0]
        
        # If node exists, return its message
        return record["a.message"]

# Initialize Flask app
app = Flask(__name__)

# Initialize Neo4j database connection
neo4j_db = HelloWorldExample("bolt://localhost:7687", "neo4j", "TOPICdb1")

@app.route("/")
def home():
    try:
        # Retrieve or create greeting message from Neo4j
        greeting = neo4j_db.get_or_create_greeting("Hello, Neo4j!")
        return jsonify({"greeting": greeting})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/close_db")
def close_db():
    try:
        neo4j_db.close()
        return jsonify({"message": "Database connection closed."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Set ngrok auth token and expose the app
    ngrok.set_auth_token("TOKEN")  # Replace with your actual ngrok auth token
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