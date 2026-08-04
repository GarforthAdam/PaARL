import re
from os import listdir
from os.path import isfile, join
from chunking import chunk_by_hashtag
from chunking import is_chunk_closed
from chunking import token_count
from chunking import MD_with_llm
import ollama
from transformers import AutoTokenizer
import pickle
import time
import threading
import os

#function that will skip papers that take too long (possible inf loops)
def chunk_with_timeout(raw_text, embedding_model, result, timeout=600):
    def target():
        result.append(chunk_by_hashtag(raw_text, embedding_model))
    
    thread = threading.Thread(target = target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    return thread.is_alive() #true if still running (went past 10 mins)




tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')

#Define function for adding chunks to database

#Use embedding model and language model from Ollama
#replace language model with claude api if wanting inclusion?


EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf' 
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'



#embedding should be a list of floats as vectors
#define the vector database

if __name__ == "__main__":
    VECTOR_DB = []
    GROUP_INDEX = {}
    def add_chunk_to_database(chunk):
        #add a final check of chunk tokens to NOT CRASH THE DAMN MODEL AGAIN 
        text = chunk['text']
        #hard truncate at 500 tokens 
        tokens = tokenizer.encode(text)
        if len(tokens) > 500:
            print(f"WARNING: chunk over limit ({len(tokens)} tokens), truncating...")
            text = tokenizer.decode(tokens[:500], skip_special_tokens=True)
        response = ollama.embed(model=EMBEDDING_MODEL, input=text)
        VECTOR_DB.append((chunk, response['embeddings'][0]))
        
        #Build a group index to check through post similarity search
        if "label" in chunk["metadata"]:
            label = chunk["metadata"]["label"]
            if label not in GROUP_INDEX:
                GROUP_INDEX[label] = [] #create empty list for chunks
            GROUP_INDEX[label].append(chunk) #add all labelled chunks into the index



    filelist = [f for f in listdir("[path_to_parsed_files]/") if isfile(join("[path_to_parsed_files]/", f))]
    #Do we want it all in this for loop?
    all_chunks=[]
    i = 0
    bad_articles = []
    times = []
    for filename in filelist:
        if os.path.exists('all_chunks.pkl'):
            print("files already chunked, embedding...")
            with open('all_chunks.pkl', 'rb') as f:
                all_chunks = pickle.load(f)
            break
        time_start = time.time()
        with open(f"[path_to_parsed_files]/{filename}", "r", encoding='utf-8') as f:
            raw_text = f.read()
        
        result = []
        timed_out = chunk_with_timeout(raw_text, EMBEDDING_MODEL, result, timeout = 600)

        
        time_elapsed = time.time() - time_start
        if timed_out:
            print(f"TIMEOUT: skipping {filename}")
            bad_articles.append(filename)
        else:        
            all_chunks.extend(result[0])
            avg = sum(times) / len(times) if times else time_elapsed
            time_remaining = avg * (len(filelist) - i)
            print(f'parsing file {filename}, number {i}/{len(filelist)} ({time_elapsed:.1f}s), est. {time_remaining/60:0.1f} mins remain')
        
       
        times.append(time_elapsed)
        i += 1
    if not os.path.exists('all_chunks.pkl'):
        with open(f'all_chunks.pkl', 'wb') as f:
            pickle.dump(all_chunks, f)

    for i, chunk in enumerate(all_chunks):
        add_chunk_to_database(chunk)
        print(f'Added chunk {i+1}/{len(all_chunks)} to the database')
        
    with open('vector_db.pkl', 'wb') as f:
        pickle.dump(VECTOR_DB, f)
    print('Saved VECTOR_DB')

    with open('group_index.pkl', 'wb') as f:
        pickle.dump(GROUP_INDEX, f)
    print('Saved GROUP_INDEX')

    print(f'skipped {len(bad_articles)} articles for chunking')
    for f in bad_articles:
        print(f' {f}')



#Now we want to start doing the AI stuff, defining the prompt and searching for it
#First we need the cosine similarity
#Basically cos(theta) from the vectors

def cosine_similarity(Vector_DB, prompt_vector):
    dot_product = sum([x * y for x, y in zip(Vector_DB, prompt_vector)])
    norm_a = sum([x ** 2 for x in Vector_DB]) ** 0.5
    norm_b = sum([x ** 2 for x in prompt_vector]) ** 0.5

    return dot_product / (norm_a * norm_b)
#initial was 10
def retrieve(query, VECTOR_DB, GROUP_INDEX, top_n = 15):
    #embed the query
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query) ['embeddings'][0]
    # temp list for (chunk, similarity) pairs
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    #sort by similarity in DESCENDING order
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_chunks = similarities[:top_n]

    #group lookup for each top chunk
    final_results = list(top_chunks)
    #make an empty set of the seen labels
    seen_labels = set()
    for chunk, similarity in top_chunks:
        label = chunk["metadata"].get("label")
        if label and label not in seen_labels:
            seen_texts = {c["text"] for c,_ in final_results}
            for grouped_chunk in GROUP_INDEX[label]:
                if grouped_chunk["metadata"]["chunk_id"] not in seen_texts:
                    final_results.append((grouped_chunk, None))
            seen_labels.add(label)
    return final_results


def expand_groups(chunks):
    expanded = []
    seen = set()
    for chunk in chunks:
        group = chunk.get("group_label")
        if group and group in GROUP_INDEX:
            for c in GROUP_INDEX[group]:
                if c["text"] not in seen:
                    expanded.append(c)
                    seen.add(c["text"])
        elif chunk["text"] not in seen:
            expanded.append(chunk)
            seen.add(chunk["text"])
    return expanded
#initial was 5
def rrf_combine(v0_ranked, v3_ranked, k=60, top_k=7):
    # RRF: score = sum(1 / (k + rank + 1))
    score_map = {}
    chunk_map = {}  # text -> chunk (original object)
    for rank, (chunk, _) in enumerate(v0_ranked):
        key = chunk["text"]
        chunk_map[key] = chunk
        score_map[key] = score_map.get(key, 0.0) + 1.0 / (k + rank + 1)
    for rank, (chunk, _) in enumerate(v3_ranked):
        key = chunk["text"]
        chunk_map[key] = chunk
        score_map[key] = score_map.get(key, 0.0) + 1.0 / (k + rank + 1)
    ranked_keys = sorted(score_map, key=score_map.get, reverse=True)
    return [chunk_map[key] for key in ranked_keys[:top_k]]

#not currently used, have an llm decide if topic needs more context or not
def needs_more_context(new_message, current_topics, recent_history):
    response = client.messages.create(
        model = "claude-haiku-4-5",
        max_tokens= 200,
        messages = [{
            "role": "user",
            "content": f'''You are making a critical decision in this programs function: You are deciding
            whether a follow-up question in a physics and astronomy chatbot needs NEW document retrieval, or can
            be answered from context already loaded. Should you fail, the entire program fails.

            Currently loaded context covers these topics and sections:
            {current_topics}

            The recent conversation:
            {recent_history}

            The new message:
            "{new_message}"

            Return retrieval=true  if the message asks about a different topic, a different paper, 
            a different physical system, or requests specifics not covered by the topics stated above.
            Return retrieval=false if it's a clarification, elaboration, rephrasing, or a follow-up question 
            answerable from the topics already loaded.

            Return ONLY a JSON object in EXACTLY this format, with no other test:
            {{"needs_retrieval": true}}
        '''
        }]
    )
    try:
        raw = re.sub(r'```json|```', '', response.content[0].text).strip()
        return json.loads(raw).get("needs_retrieval", True)
    except (json.JSONDecodeError, IndexError):
        return True #return true as a failsafe, better to over retrieve in this case
