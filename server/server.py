from flask import Flask, request, jsonify, send_file
import logging
from dotenv import load_dotenv
import os
from RAGHelper_cloud import RAGHelperCloud
from RAGHelper_local import RAGHelperLocal
from pymilvus import Collection, connections

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instantiate the RAG Helper class
if os.getenv("use_openai") == "True" or os.getenv("use_gemini") == "True" or os.getenv("use_azure") == "True":
    raghelper = RAGHelperCloud(logger)
else:
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
    json = request.get_json()
    filename = json['filename']

    raghelper.addDocument(filename)

    return jsonify({"filename": filename}), 200


@app.route("/query", methods=['POST'])
def query():
    try:
        data = request.get_json()

        # Get response from RAG helper
        (new_history, response) = raghelper.handle_user_interaction(
            data['query'],
            data.get('history', [])
        )

        # Handle citation verification errors if any
        if response.get('error') == 'citation_verification_failed':
            return jsonify({
                'error': 'Invalid or ambiguous citations detected',
                'details': response.get('details', {})
            }), 400

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route("/chat", methods=['POST'])
def chat():
    """
    Handle chat interactions with the RAG system.

    This endpoint processes the user's prompt, retrieves relevant documents,
    and returns the assistant's reply along with conversation history.

    Returns:
        JSON response containing the assistant's reply, history, documents, and other metadata.
    """
    json = request.get_json()
    prompt = json['prompt']
    history = json.get('history', [])
    original_docs = json.get('docs', [])
    docs = original_docs

    # Get the LLM response
    (new_history, response) = raghelper.handle_user_interaction(prompt, history)

    # Handle citation verification errors if any
    if response.get('error') == 'citation_verification_failed':
        return jsonify({
            'error': 'Invalid or ambiguous citations detected',
            'details': response.get('details', {})
        }), 400

    if len(docs) == 0 or 'docs' in response:
        docs = response['docs']
    # Break up the response for OS LLMs
    if isinstance(raghelper, RAGHelperLocal):
        end_string = os.getenv("llm_assistant_token")
        reply = response['text'][response['text'].rindex(end_string) + len(end_string):]

        # Get history
        new_history = [{"role": msg["role"], "content": msg["content"].format_map(response)} for msg in new_history]
        new_history.append({"role": "assistant", "content": reply})
    else:
        # Populate history properly, also turning it into dict instead of tuple, so we can serialize
        new_history = [{"role": msg[0], "content": msg[1].format_map(response)} for msg in new_history]
        new_history.append({"role": "assistant", "content": response['answer']})
        reply = response['answer']
    # Make sure we format the docs properly

    if len(original_docs) == 0 or 'docs' in response:
        new_docs = [{
            's': doc.metadata['source'],
            'c': doc.page_content,
            **({'pk': doc.metadata['pk']} if 'pk' in doc.metadata else {}),
            **({'provenance': float(doc.metadata['provenance'])} if 'provenance' in doc.metadata else {}),
            **({'citation_verification': response.get('citation_verification',
                                                      {})} if 'citation_verification' in response else {})
        } for doc in docs if 'source' in doc.metadata]
    else:
        new_docs = docs

    result = {
        "reply": reply,
        "history": new_history,
        "documents": new_docs
    }

    # Add citation verification results if present
    if 'citation_verification' in response:
        result['citation_verification'] = response['citation_verification']

    return jsonify(result), 200


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
    file_types = os.getenv("file_types").split(",")
    files = [f for f in os.listdir(data_dir) if
             os.path.isfile(os.path.join(data_dir, f)) and os.path.splitext(f)[1][1:] in file_types]
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
    json = request.get_json()
    filename = json['filename']
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
    json = request.get_json()
    filename = json['filename']
    data_dir = os.getenv('data_directory')
    file_path = os.path.join(data_dir, filename)

    # Remove from Milvus
    connections.connect(uri=os.getenv('vector_store_path'))
    collection = Collection("LangChainCollection")
    collection.load()
    result = collection.delete(f'source == "{file_path}"')
    collection.release()

    # Remove from disk too
    os.remove(file_path)

    # BM25 needs to be re-initialized
    raghelper.loadData()

    return jsonify({"count": result.delete_count})


if __name__ == "__main__":
    app.run(host="0.0.0.0")
