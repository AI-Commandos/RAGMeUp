import os
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.prompts import ChatPromptTemplate

# This is a clever little function that attempts to compute the attribution of each document retrieved from the RAG store towards the generated answer.
# The way this function works is by getting the (self-)attention scores of each token towards every other token and then computing, for each document:
# - The attention of the consecutive sequence of tokens from the user query towards the document
# - The attention of the consecutive sequence of tokens from the document towards the user query
# - The attention of the consecutive sequence of tokens from the answer towards the document
# - The attention of the consecutive sequence of tokens from the document towards the answer
# This is computed on the full thread with the proper template applied. The four attention scores are summed and divided by the total sum of all
# weights in the attention layer, which contains eg. other documents' attention but also query-query, query-answer, answer-query and query-query attention.
#
# This is by no means a foolproof way to attribute towards each of the documents properly but it is _a_ way.
def compute_attention(model, tokenizer, thread, query, context, answer):
    include_query=True
    if os.getenv("attribute_include_query") == "False":
        include_query=False

    # Encode the full thread
    thread_tokens = tokenizer.encode(thread, return_tensors="pt", add_special_tokens=False)

    # Compute the attention
    with torch.no_grad():
        output = model(input_ids=thread_tokens, output_attentions=True, attn_implementation="eager")
    
    # Use the last layer's attention
    attentions = output.attentions[-1]
    # Tokenize query, context parts, and answer
    query_tokens = tokenizer.encode(query, add_special_tokens=False)
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)

    # Find the start and end positions of query, context parts, and answer in the thread tokens
    query_start, query_end = find_sublist_positions(thread_tokens[0].tolist(), query_tokens)
    answer_start, answer_end = find_sublist_positions(thread_tokens[0].tolist(), answer_tokens)
    
    # Get the token offsets for each of the documents in the context
    context_offsets = []
    for part in context:
        part_tokens = tokenizer.encode(part, add_special_tokens=False)
        context_offsets.append(find_sublist_positions(thread_tokens[0].tolist(), part_tokens))
    
    # We sum the total attention seen but only from documents/query/answer to each other and themselves, excluding meta-characters and instruction prompt
    total_attention = []
    # Add the query/answer self- and cross-attentions to the total sum, make sure we only keep actual (positive) attention. Some models don't always
    # pay attention to all tokens and in those cases extremely long texts will diminish any attention paid.
    query_to_query = attentions[0, :, query_start:query_end, query_start:query_end].mean().item()
    if query_to_query > 0:
        total_attention.append(query_to_query)
    query_to_answer = attentions[0, :, query_start:query_end, answer_start:answer_end].mean().item()
    if query_to_answer > 0:
        total_attention.append(query_to_answer)
    answer_to_answer = attentions[0, :, answer_start:answer_end, answer_start:answer_end].mean().item()
    if answer_to_answer > 0:
        total_attention.append(answer_to_answer)
    answer_to_query = attentions[0, :, answer_start:answer_end, query_start:query_end].mean().item()
    if answer_to_query > 0:
        total_attention.append(answer_to_query)

    # Extract the attention weights for each document
    doc_attentions = []
    for start, end in context_offsets:
        # Focus on the attention from the answer to this document part
        doc_attention = []
        answer_to_doc = attentions[0, :, answer_start:answer_end, start:end].mean().item()
        if answer_to_doc > 0:
            doc_attention.append(answer_to_doc)
        doc_to_answer = attentions[0, :, start:end, answer_start:answer_end].mean().item()
        if doc_to_answer > 0:
            doc_attention.append(doc_to_answer)
        if include_query:
            # Also consider the attention from the query to this document part
            query_to_doc = attentions[0, :, query_start:query_end, start:end].mean().item()
            if query_to_doc > 0:
                doc_attention.append(query_to_doc)
            doc_to_query = attentions[0, :, start:end, query_start:query_end].mean().item()
            if doc_to_query > 0:
                doc_attention.append(doc_to_query)
        
        total_attention += doc_attention
        doc_attentions.append(np.mean(doc_attention))

    mean_total_attention = np.mean(total_attention)
    return [score / mean_total_attention for score in doc_attentions]

def find_sublist_positions(thread_tokens, part_tokens):
    len_thread = len(thread_tokens)
    len_part = len(part_tokens)

    for i in range(len_thread - len_part + 1):
        if thread_tokens[i:i + len_part] == part_tokens:
            return i, i + len_part - 1
    
    raise ValueError("Sublist not found")

def compute_rerank_provenance(reranker, query, documents, answer):
    if os.getenv("attribute_include_query") == "True":
        full_text = query + "\n" + answer
    else:
        full_text = answer
    
    # Score the documents, this will return the same document list but now with a relevance_score in metadata
    scored_documents = reranker.compress_documents(documents, full_text)
    return scored_documents

def compute_llm_provenance(tokenizer, model, query, context, answer):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    prompt = os.getenv("provenance_llm_prompt")
    end_string = "assistant\n\n"
    device = "cuda"
    if os.getenv("force_cpu") == "True":
        device = "cpu"

    # Go over all documents in the context
    provenance_scores = []
    for doc in context:
        # Create the thread to ask the LLM to assign a score to this document for provenance
        new_doc = doc
        # Replace brackets for format_map to work
        new_doc.page_content = new_doc.page_content.replace("{", "{{").replace("}", "}}")
        context_thread = [{'role': 'user', 'content': prompt.format_map({"query": query, "context": new_doc, "answer": answer})}]
        # Tokenize the thread containing our document
        input_chat = tokenizer.apply_chat_template(context_thread, tokenize=False)
        inputs = tokenizer(input_chat, return_tensors="pt", padding=True, truncation=True).to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        # Generate the output from the LLM and parse it as number/score
        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = generated_text[generated_text.rindex(end_string)+len(end_string):]
        score = re.findall("\d+\.?\d*", answer.replace(",", "."))[-1]
        provenance_scores.append(score)
    
    return provenance_scores

def compute_llm_provenance_cloud(llm, query, context, answer):
    prompt = os.getenv("provenance_llm_prompt")
    # Go over all documents in the context
    provenance_scores = []
    for doc in context:
        # Create the thread to ask the LLM to assign a score to this document for provenance
        new_doc = doc
        new_doc.page_content = new_doc.page_content.replace("{", "{{").replace("}", "}}")
        input_chat = [('human', prompt.format_map({"query": query, "context": new_doc, "answer": answer}))]
        generated_text = llm.invoke(input_chat)
        if hasattr(generated_text, 'content'):
            generated_text = generated_text.content
        elif hasattr(generated_text, 'answer'):
            generated_text = generated_text.answer
        elif 'answer' in generated_text:
            generated_text = generated_text["answer"]
        score = re.findall("\d+\.?\d*", generated_text.replace(",", "."))[-1]
        provenance_scores.append(score)
    
    return provenance_scores

class DocumentSimilarityAttribution:
    def __init__(self):
        device = 'cuda'
        if os.getenv('force_cpu') == "True":
            device = 'cpu'
        self.model = SentenceTransformer(os.getenv('provenance_similarity_llm'), device=device)

    def compute_similarity(self, query, context, answer):
        include_query=True
        if os.getenv("attribute_include_query") == "False":
            include_query=False
        # Encode the answer, query, and context documents
        answer_embedding = self.model.encode([answer])[0]
        context_embeddings = self.model.encode(context)
        
        if include_query:
            query_embedding = self.model.encode([query])[0]

        # Compute similarity scores
        similarity_scores = []
        for i, doc_embedding in enumerate(context_embeddings):
            # Similarity between document and answer
            doc_answer_similarity = cosine_similarity([doc_embedding], [answer_embedding])[0][0]
            
            if include_query:
                # Similarity between document and query
                doc_query_similarity = cosine_similarity([doc_embedding], [query_embedding])[0][0]
                # Average of answer and query similarities
                similarity_score = (doc_answer_similarity + doc_query_similarity) / 2
            else:
                similarity_score = doc_answer_similarity

            similarity_scores.append(similarity_score)

        # Normalize scores
        total_similarity = sum(similarity_scores)
        normalized_scores = [score / total_similarity for score in similarity_scores] if total_similarity > 0 else similarity_scores

        return normalized_scores