from flask import Flask, request, jsonify
import logging
from dotenv import load_dotenv
from RAGHelper import RAGHelper

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instantiate the RAG Helper class
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
    history = json.get('history', '')
    docs = json.get('docs', [])

    # Get the LLM response
    response = raghelper.handle_user_interaction(prompt, history)
    if len(docs) == 0 and 'docs' in response:
        docs = response['docs']
    
    # Break up the response
    term_symbol = raghelper.tokenizer.eos_token
    end_string = f"{term_symbol}assistant\n\n"
    previous_chat = response['text'][:response['text'].find(end_string)+len(term_symbol)]
    reply = response['text'][response['text'].find(end_string)+len(end_string):]
    
    # Format this properly using the template now so we can continue with this at a later point
    prompted = raghelper.tokenizer.apply_chat_template(
        [{'role': 'assistant', 'content': reply}], tokenize=False
    )
    prompted = prompted[len(raghelper.tokenizer.bos_token):]
    # Now glue them together
    full_history = previous_chat + prompted
    
    return jsonify({"reply": reply, "history": full_history, "documents": docs}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0")
