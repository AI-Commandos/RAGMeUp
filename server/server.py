from flask import Flask, request, jsonify, send_file
import logging
from dotenv import load_dotenv
import os
from RAGHelper_cloud import RAGHelperCloud
from RAGHelper_local import RAGHelperLocal
from pymilvus import Collection, connections
import pygraphviz as pgv


def load_bashrc():
    """
    Load environment variables from the user's .bashrc file.

    This function looks for the .bashrc file in the user's home directory
    and loads any environment variables defined with the 'export' command
    into the current environment.
    """
    bashrc_path = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc_path):
        with open(bashrc_path) as f:
            for line in f:
                if line.startswith("export "):
                    key, value = line.strip().replace("export ", "").split("=", 1)
                    value = value.strip(' "\'')
                    os.environ[key] = value


# Initialize Flask application
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable parallelism in tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_bashrc()
load_dotenv()

# Instantiate the RAG Helper class based on the environment configuration
if any(os.getenv(key) == "True" for key in ["use_openai", "use_gemini", "use_azure", "use_ollama"]):
    logger.info("Instantiating the cloud RAG helper.")
    raghelper = RAGHelperCloud(logger)
else:
    logger.info("Instantiating the local RAG helper.")
    raghelper = RAGHelperLocal(logger)


@app.route("/chat", methods=['POST'])
def chat():
    """
    Handle chat interactions with the RAG system.
    """
    try:
        # Parse incoming request JSON
        json_data = request.get_json()
        prompt = json_data.get('prompt')
        history = json_data.get('history', [])
        original_docs = json_data.get('docs', [])
        docs = original_docs

        # Call RAG helper for interaction
        new_history, response, graph_response = raghelper.handle_user_interaction(prompt, history)
        
        print('hoi')
        print(response)
        print(graph_response)
        print()
        print()

         # Validate response structure
        if not isinstance(response, dict) or 'answer' not in response:
            raise ValueError("Invalid response format from RAG helper.")

        # Process graph response if available
        graph_file_path = None
        if graph_response:
            try:
                graph_file_path = 'graph.png'
                graph = pgv.AGraph(directed=True)
                for node in graph_response.get('nodes', []):
                    graph.add_node(node['id'], label=node['label'])
                for edge in graph_response.get('edges', []):
                    graph.add_edge(edge['source'], edge['target'], label=edge['label'])
                graph.layout(prog='dot')
                graph.draw(graph_file_path)
            except Exception as e:
                logger.error(f"Graph processing error: {e}")
                graph_file_path = None

        # Format the documents for the frontend
        new_docs = [
            {
                's': doc.get('metadata', {}).get('source', 'Unknown'),
                'c': doc.get('page_content', ''),
                **({'pk': doc.get('metadata', {}).get('pk')} if 'pk' in doc.get('metadata', {}) else {}),
                **({'provenance': float(doc.get('metadata', {}).get('provenance'))} if 'provenance' in doc.get('metadata', {}) else {})
            }
            for doc in docs if 'source' in doc.get('metadata', {})
        ]

        # Build response dictionary
        response_dict = {
            "reply": response.get('answer', 'No answer available.'),
            "history": new_history,
            "documents": new_docs,
            "rewritten": False,
            "question": prompt,
            "fetched_new_documents": not original_docs,
            "graph": graph_file_path
        }

        # Handle rewritten questions
        if os.getenv("use_rewrite_loop") == "True" and prompt != response.get('question', prompt):
            response_dict["rewritten"] = True
            response_dict["question"] = response['question']

        return jsonify(response_dict), 200

    except KeyError as e:
        logger.error(f"Missing key in data: {e}")
        return jsonify({'error': f"Missing key in data: {e}"}), 400
    except Exception as e:
        logger.error(f"Unhandled error in /chat: {e}")
        return jsonify({'error': str(e)}), 500



@app.route("/add_document", methods=['POST'])
def add_document():
    """
    Add a document to the RAG helper.

    This endpoint expects a JSON payload containing the filename of the document to be added.
    It then invokes the addDocument method of the RAG helper to store the document.

    Returns:
        JSON response with the filename and HTTP status code 200.
    """
    json_data = request.get_json()
    filename = json_data.get('filename')

    if not filename:
        return jsonify({"error": "Filename is required"}), 400
    logger.info(f"Adding document {filename}")
    raghelper.add_document(filename)

    return jsonify({"filename": filename}), 200


@app.route("/get_documents", methods=['GET'])
def get_documents():
    """
    Retrieve a list of documents from the data directory.

    This endpoint checks the configured data directory and returns a list of files
    that match the specified file types.

    Returns:
        JSON response containing the list of files.
    """
    data_dir = os.getenv('data_directory')
    file_types = os.getenv("file_types", "").split(",")

    # Filter files based on specified types
    files = [f for f in os.listdir(data_dir)
             if os.path.isfile(os.path.join(data_dir, f)) and os.path.splitext(f)[1][1:] in file_types]

    return jsonify(files)


@app.route("/get_document", methods=['POST'])
def get_document():
    """
    Retrieve a specific document from the data directory.

    This endpoint expects a JSON payload containing the filename of the document to retrieve.
    If the document exists, it is sent as a file response.

    Returns:
        JSON response with the error message and HTTP status code 404 if not found,
        otherwise sends the file as an attachment.
    """
    json_data = request.get_json()
    filename = json_data.get('filename')
    data_dir = os.getenv('data_directory')
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path,
                     mimetype='application/octet-stream',
                     as_attachment=True,
                     download_name=filename)


@app.route("/delete", methods=['POST'])
def delete_document():
    """
    Delete a specific document from the data directory and the Milvus vector store.

    This endpoint expects a JSON payload containing the filename of the document to delete.
    It removes the document from the Milvus collection and the filesystem.

    Returns:
        JSON response with the count of deleted documents.
    """
    json_data = request.get_json()
    filename = json_data.get('filename')
    data_dir = os.getenv('data_directory')
    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    # Remove from Milvus
    connections.connect(uri=os.getenv('vector_store_uri'))
    collection = Collection("LangChainCollection")
    collection.load()
    result = collection.delete(f'source == "{file_path}"')
    collection.release()

    # Remove from disk too
    os.remove(file_path)

    # Reinitialize BM25
    raghelper.loadData()

    return jsonify({"count": result.delete_count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0")
