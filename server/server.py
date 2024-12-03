from flask import Flask, request, jsonify, send_file
import logging
from dotenv import load_dotenv
import os
from RAGHelper_cloud import RAGHelperCloud
from RAGHelper_local import RAGHelperLocal
from pymilvus import Collection, connections
import sqlite3


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


@app.route("/chat", methods=['POST'])
def chat():
    """
    Handle chat interactions with the RAG system.

    This endpoint processes the user's prompt, retrieves relevant documents,
    and returns the assistant's reply along with conversation history.

    Returns:
        JSON response containing the assistant's reply, history, documents, and other metadata.
    """
    json_data = request.get_json()
    prompt = json_data.get('prompt')
    history = json_data.get('history', [])
    original_docs = json_data.get('docs', [])
    docs = original_docs

    # Get the LLM response
    (new_history, response) = raghelper.handle_user_interaction(prompt, history)
    if not docs or 'docs' in response:
        docs = response['docs']

    # Break up the response for local LLMs
    if isinstance(raghelper, RAGHelperLocal):
        end_string = os.getenv("llm_assistant_token")
        reply = response['text'][response['text'].rindex(end_string) + len(end_string):]

        # Get updated history
        new_history = [{"role": msg["role"], "content": msg["content"].format_map(response)} for msg in new_history]
        new_history.append({"role": "assistant", "content": reply})
    else:
        # Populate history for other LLMs
        new_history = [{"role": msg[0], "content": msg[1].format_map(response)} for msg in new_history]
        new_history.append({"role": "assistant", "content": response['answer']})
        reply = response['answer']

    # Format documents
    fetched_new_documents = False
    if not original_docs or 'docs' in response:
        fetched_new_documents = True
        new_docs = [{
            's': doc.metadata['source'],
            'c': doc.page_content,
            **({'pk': doc.metadata['pk']} if 'pk' in doc.metadata else {}),
            **({'provenance': float(doc.metadata['provenance'])} if 'provenance' in doc.metadata and doc.metadata['provenance'] is not None else {})
        } for doc in docs if 'source' in doc.metadata]
    else:
        new_docs = docs

    # Build the response dictionary
    response_dict = {
        "reply": reply,
        "history": new_history,
        "documents": new_docs,
        "rewritten": False,
        "question": prompt,
        "fetched_new_documents": fetched_new_documents
    }

    # Check for rewritten question
    if os.getenv("use_rewrite_loop") == "True" and prompt != response['question']:
        response_dict["rewritten"] = True
        response_dict["question"] = response['question']

    return jsonify(response_dict), 200


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
    
    # Print the data directory and file types being used
    print(f"Data Directory in GET: {data_dir}")
    print(f"File Types in GET: {file_types}")

    # Filter files based on specified types
    files = [f for f in os.listdir(data_dir)
             if os.path.isfile(os.path.join(data_dir, f)) and os.path.splitext(f)[1][1:] in file_types]
    
    # Print the list of files found
    print(f"Files Found in GET: {files}")
    
    # Initialize result structure
    documents_with_feedback = []

    # Connect to feedback database
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()

    for filename in files:
        # Query for feedback related to this file
        cursor.execute(
            "SELECT query, answer, rating, timestamp FROM Feedback WHERE document_id = ?",
            (filename,)
        )
        feedback = cursor.fetchall()

        # Format feedback as a list of dictionaries
        feedback_list = [
            {"query": row[0], "answer": row[1], "rating": row[2], "timestamp": row[3]} for row in feedback
        ]

        # Append document with feedback
        documents_with_feedback.append({"filename": filename, "feedback": feedback_list})

        print(f"Files get docuemnts_with_feedback in GET: {documents_with_feedback}")
        
    conn.close()

    return jsonify(documents_with_feedback)
    
    # return jsonify(files)


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

    print(f"Received request for filename: {filename}")  # Debug print
    print(f"Constructed file path: {file_path}")         # Debug print

    if not os.path.exists(file_path):
        print("File not found.")  # Debug print
        return jsonify({"error": "File not found"}), 404
    
    # Connect to the feedback database to retrieve feedback for the file
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT query, answer, rating, timestamp 
        FROM Feedback 
        WHERE document_id = ?
    ''', (file_path,))
    
    feedback = cursor.fetchall()
    print(f"Feedback retrieved for {file_path}: {feedback}")  # Debug print
    conn.close()


    # Format feedback as a list of dictionaries
    feedback_list = [
        {"query": row[0], "answer": row[1], "rating": row[2], "timestamp": row[3]}
        for row in feedback
    ]
    print(f"Formatted feedback: {feedback_list}")  # Debug print
    
    # Return the document along with its feedback metadata
    response = {
        "filename": filename,
        "feedback": feedback_list
    }
    
    print(f"Response metadata: {response}")  # Debug print

    # Use `send_file` to include the actual file content as an attachment
    return send_file(
        file_path,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=filename,
        headers={"X-Metadata": json.dumps(response)}  # Optional: include metadata in headers
    )
    

    # return send_file(file_path,
    #                  mimetype='application/octet-stream',
    #                  as_attachment=True,
    #                  download_name=filename)


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

@app.route('/save_feedback', methods=['POST'])
def save_feedback():
    data = request.json
    query = data.get('query')
    answer = data.get('answer')
    document_id = data.get('document_id')
    rating = data.get('rating')
    timestamp = data.get('timestamp')
    
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO Feedback (query, answer, document_id, rating, timestamp) 
        VALUES (?, ?, ?, ?, ?)
    ''', (query, answer, document_id, rating, timestamp))
    
    conn.commit()
    conn.close()
    
    return jsonify({"status": "success", "message": "Feedback saved"})

if __name__ == "__main__":
    app.run(host="0.0.0.0")
