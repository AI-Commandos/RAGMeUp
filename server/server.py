from flask import Flask, request, jsonify, send_file
import logging
from dotenv import load_dotenv
import os
from RAGHelper_cloud import RAGHelperCloud
from RAGHelper import RAGHelper
from pymilvus import Collection, connections

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instantiate the RAG Helper class
if os.getenv("use_openai") == "True" or os.getenv("use_gemini") == "True" or os.getenv("use_azure") == "True":
    raghelper = RAGHelperCloud(logger)
else:
    raghelper = RAGHelper(logger)

@app.route("/add_document", methods=['POST'])
def add_document():
    json = request.get_json()
    filename = json['filename']

    raghelper.addDocument(filename)

    return jsonify({"filename": filename}), 200

@app.route("/chat", methods=['POST'])
def chat():
    json = request.get_json()
    prompt = json['prompt']
    history = json.get('history', [])
    original_docs = json.get('docs', [])
    docs = original_docs

    # Get the LLM response
    (new_history, response) = raghelper.handle_user_interaction(prompt, history)
    if len(docs) == 0 or 'docs' in response:
        docs = response['docs']

    # Break up the response for OS LLMs
    if isinstance(raghelper, RAGHelper):
        end_string = os.getenv("llm_assistant_token")
        reply = response['text'][response['text'].rindex(end_string)+len(end_string):]

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
            **({'provenance': float(doc.metadata['provenance'])} if 'provenance' in doc.metadata else {})
        } for doc in docs if 'source' in doc.metadata]
    else:
        new_docs = docs

    return jsonify({"reply": reply, "history": new_history, "documents": new_docs}), 200

@app.route("/get_documents", methods=['GET'])
def get_documents():
    data_dir = os.getenv('data_directory')
    file_types = os.getenv("file_types").split(",")
    files =  [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and os.path.splitext(f)[1][1:] in file_types]
    return jsonify(files)

@app.route("/get_document", methods=['POST'])
def get_document():
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
