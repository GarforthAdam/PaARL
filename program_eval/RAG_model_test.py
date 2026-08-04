import re
import os
import sys
from os import listdir
from os.path import isfile, join
from chunking import chunk_by_hashtag
from chunking import is_chunk_closed
from chunking import token_count
from chunking import MD_with_llm
import ollama
import anthropic
import random
import json
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
import pickle
from RAG import retrieve
from RAG import cosine_similarity


#So need to take query, then generate HyDe answer, then cosine similarity, and then pass to your rerankers
#randomize answers and save the answers to a pickle file or something to ensure blind testing
#so test 1) no reranker 2) reranker V1 3) reranker v3 (v2 is just a less good v3) #if v3 performs significantly worse but v1 very well we can look into v2

if os.path.exists('[path_to]/vector_db.pkl'):
    with open('[path_to]/vector_db.pkl', 'rb') as f:
        VECTOR_DB = pickle.load(f)
    print(f'Loaded {len(VECTOR_DB)} chunks from disk')
else: 
    print("please run RAG embedding program to function properly!")
    sys.exit()

if os.path.exists('[path_to]/group_index.pkl'):
    with open('[path_to]/group_index.pkl', 'rb') as f:
        GROUP_INDEX = pickle.load(f)
    print(f'loaded {len(GROUP_INDEX)} tied chunks from disk ')


EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf' 
client = anthropic.Anthropic()


v1_reranker = CrossEncoder("[path_to]/V1_clean")
v3_reranker = CrossEncoder("[path_to]/V3_clean")

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


input_query = input('Ask me a question, ideally about physics please: ')
#generate a hypothetical answer to be embedded instead
response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f'''Generate a short hypothetical passage answering this query in academic-style writing,
                as if it was in a published scientific paper.

                -Use standard, established terminology from the field freely.
                -DO NOT invent specific named results, fabricated citations, or specific numeric values presented as findings
                -DO NOT assume a specific method is used IF NOT EXPLICITLY MENTIONED, 
                describe the general class of technique instead, but write with the tone of a real passage
                
                Return ONLY a JSON object in EXACTLY this format, with no other text:
                {{"answer": ["the passage you create"]}}
                USE THIS QUERY:{input_query}
                ''',
                }
        ]   ,
        )
try: #check JSON output for a list of questions to parse through
    raw = response.content[0].text
    raw = re.sub(r'```json|```','', raw).strip()
    result = json.loads(raw)
    #If the LLM returns a string by mistake, change to JSON output
    if isinstance(result,str):
        result = json.loads(result)
    passage = result.get("answer", [""])[0]
except json.JSONDecodeError:
    print("model returned answer in wrong format, rerun!")
    print(f"Raw was: {response.content[0].text[:200]}")
    sys.exit()

retrieved_knowledge = retrieve(passage, VECTOR_DB, GROUP_INDEX)

models = [
    #("v1", v1_reranker),
    #("v3", v3_reranker),
    ("rrf", v3_reranker),
    ("nothing", "nothing")
    ]
#shuffle the models to generate a blind study
random.shuffle(models)
all_responses = []

#Blind test of the three initial models: (make sure to adjust quotation to run properly)
#TRIAL ONE
"""
for label, model in models:
    all_context_chunks = []
#MAKE A FUNCTION TO CALL LLM IN FUTURE - PASS MODEL, INPUT QUERY, CONTEXT THROUGH
    if model == "nothing":
        chunks = [c for c, _ in retrieved_knowledge[:5]]
        all_context_chunks.extend(expand_groups(chunks))
        context = '\n'.join([
            f'Source: {chunk["metadata"]["article_title"]} - {chunk["metadata"]["heading"]} - {chunk["metadata"]["article_authors"]}\n{chunk["text"]}'
            for chunk in all_context_chunks
        ])
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=2500,
            system = f''' #You are a helpful chatbot providing assistance in matters concerning Physics and Astronomy. 
                    #Use only the following pieces of context to answer the question. Don't make up any new information. 
                    #After EACH claim made, LIST THE SOURCES THE INFORMATION CAME FROM. PROVIDE SECTION AND TITLE IF POSSIBLE.
                    
                    #Context to use:{context}
                    ''',
            messages=[
                {
                    "role": "user",
                    "content": input_query,
                    }
            ]   ,
            )
        print("CHATBOT RESPONSE:")
        print(response)
        all_responses.append((label, response))
        continue

    pairs = [[input_query, chunk["text"]] for chunk, _ in retrieved_knowledge]
    scores = model.predict(pairs)
    ranked = sorted(zip(retrieved_knowledge, scores), key=lambda x: x[1], reverse=True)
    #c grabs chunk dict, _ gets rid of cos sim, not needed, score grabs rerankers score, to log
    top_chunks = [c for (c, _), score in ranked[:5]]
    #extend by tied chunks if they exist
    
    all_context_chunks.extend(expand_groups(top_chunks))
    context = '\n'.join([
        f'Source: {chunk["metadata"]["article_title"]} - {chunk["metadata"]["heading"]} - {chunk["metadata"]["article_authors"]}\n{chunk["text"]}'
        for chunk in all_context_chunks
    ])
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2500,
        system = f''' #You are a helpful chatbot providing assistance in matters concerning Physics and Astronomy. 
                    #Use only the following pieces of context to answer the question. Don't make up any new information. 
                    #After EACH claim made, LIST THE SOURCES THE INFORMATION CAME FROM. PROVIDE SECTION AND TITLE IF POSSIBLE.
                    
                    #Context to use:{context}
                    ''',
        messages=[
            {
                "role": "user",
                "content": input_query,
                }
        ]   ,
        )
    print("CHATBOT RESPONSE:")
    print(response)
    all_responses.append((label, response))
"""

#TRIAL TWO:
#RRF test comparing V0 (highest scorer) and V03 responses: 
def rrf_combine(v0_ranked, v3_ranked, k=60, top_k=5):
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

for label, model in models:
    all_context_chunks = []
    if model == "nothing":
        chunks = [c for c, _ in retrieved_knowledge[:5]]
        all_context_chunks.extend(expand_groups(chunks))
        context = '\n'.join([
            f'Source: {chunk["metadata"]["article_title"]} - {chunk["metadata"]["heading"]} - {chunk["metadata"]["article_authors"]}\n{chunk["text"]}'
            for chunk in all_context_chunks
            ])
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=2500,
            system = f''' You are a helpful chatbot providing assistance in matters concerning Physics and Astronomy. 
                    Use only the following pieces of context to answer the question. Don't make up any new information. 
                    After EACH claim made, LIST THE SOURCES THE INFORMATION CAME FROM. PROVIDE SECTION AND TITLE IF POSSIBLE.
                            
                    Context to use:{context}
                    ''',
            messages=[
                {
                    "role": "user",
                    "content": input_query,
                    }
               ]   ,
            )
        print("CHATBOT RESPONSE:")
        print(response)
        all_responses.append((label, response))
        continue
    pairs = [[input_query, chunk["text"]] for chunk, _ in retrieved_knowledge]
    scores = model.predict(pairs)
    ranked = sorted(zip(retrieved_knowledge, scores), key=lambda x: x[1], reverse=True)
    v3_unzipped = [(c, score) for (c, _), score in ranked]
    #unzip reranker output from tuple and pass through rrf
    rrf_list = rrf_combine(retrieved_knowledge, v3_unzipped)

    #extend by tied chunks if they exist     
    all_context_chunks.extend(expand_groups(rrf_list))
    context = '\n'.join([
        f'Source: {chunk["metadata"]["article_title"]} - {chunk["metadata"]["heading"]} - {chunk["metadata"]["article_authors"]}\n{chunk["text"]}'
        for chunk in all_context_chunks
    ])
    response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=2500,
            system = f''' You are a helpful chatbot providing assistance in matters concerning Physics and Astronomy. 
                        Use only the following pieces of context to answer the question. Don't make up any new information. 
                        After EACH claim made, LIST THE SOURCES THE INFORMATION CAME FROM. PROVIDE SECTION AND TITLE IF POSSIBLE.
                        
                        #Context to use:{context}
                        ''',
            messages=[
                {
                    "role": "user",
                    "content": input_query,
                    }
            ]   ,
            )
    print("CHATBOT RESPONSE:")
    print(response)
    all_responses.append((label, response))



#print the blind text and the answers here:
with open("blind_responses_t2.txt", "a", encoding="utf-8") as f:
    f.write(f"QUERY: {input_query}\n")
    for i, (label, response) in enumerate(all_responses, 1):
        f.write(f"\nResponse {i}:\n{response}\n")
    f.write("\n" + "="*80 + "\n\n")

with open("rrnkr_model_key_t2.txt", "a", encoding="utf-8") as f:
    f.write(f"QUERY: {input_query}\n")
    f.write(f"Order: {[label for label, _ in all_responses]}\n\n")
