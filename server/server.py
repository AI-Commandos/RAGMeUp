from flask import Flask, request, jsonify
import logging
from dotenv import load_dotenv
import os
from RAGHelper_openai import RAGHelperOpenAI
from RAGHelper import RAGHelper

load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instantiate the RAG Helper class
if os.getenv("use_openai") == "True":
    raghelper = RAGHelperOpenAI(logger)
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
    docs = json.get('docs', [])

    # Get the LLM response
    (new_history, response) = raghelper.handle_user_interaction(prompt, history)
    if len(docs) == 0 and 'docs' in response:
        docs = response['docs']

    # Break up the response for OS LLMs
    if isinstance(raghelper, RAGHelper):
        term_symbol = raghelper.tokenizer.eos_token
        if os.getenv("llm_eos_token") != "None":
            term_symbol = os.getenv("llm_eos_token")
        end_string = f"{term_symbol}assistant\n\n"
        reply = response['text'][response['text'].find(end_string)+len(end_string):]

        # Get history
        new_history = [{"role": msg["role"], "content": msg["content"].format_map(response)} for msg in new_history]
        new_history.append({"role": "assistant", "content": reply})
    else:
        # Populate history properly, also turning it into dict instead of tuple, so we can serialize
        breakpoint()
        new_history = [{"role": msg[0], "content": msg[1].format_map(response)} for msg in new_history]
        new_history.append({"role": "assistant", "content": response['answer']})
        reply = response['answer']
    
    return jsonify({"reply": reply, "history": new_history, "documents": docs}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0")
