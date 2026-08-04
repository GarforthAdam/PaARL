import re
import os
import sys
from os import listdir
from os.path import isfile, join
from chunking import chunk_by_hashtag
from chunking import is_chunk_closed
from chunking import token_count
from chunking import MD_with_llm
import anthropic
import json
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer
import pickle
import streamlit as st
from RAG import retrieve
from RAG import cosine_similarity
from RAG import expand_groups
from RAG import rrf_combine
from definitions import load_vector_db, load_group_index, load_reranker, get_client, EMBEDDING_MODEL

#load necessary data
VECTOR_DB = load_vector_db()
GROUP_INDEX = load_group_index()
v3_reranker = load_reranker()
client = get_client()

#use streamlit here

st.markdown("# PaARL: A Physics and Astronomy RAG Layer")
st.markdown("- A RAG layer added to Claude that is able to better reference needed articles")
st.markdown("- Ask a question and the response should return relevant information/articles!")
st.markdown("- If you feel more context is needed for an additional query, simply click the button below!")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_context" not in st.session_state:
    #define empty text for past context/topics
    st.session_state.current_context = ""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#button for more context:
override = st.toggle("More context", value = False)
#prompt for query
if prompt := st.chat_input("Ask me a physics or astronomy question, I will answer to the best of my ability!"):
    #save query as a part of the conversation
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    #confirm that this is the first message sent to the chat by confirming no previous context
    is_first_turn = st.session_state.current_context == ""                             #use the prev 4 messages?
    if is_first_turn or override:
        #run the full pipeline to get context
        #consider nixing the need more context and just have a button to click to add more by user discretion
        #generate a hypothetical answer to be embedded instead - HyDE response
        #LATER VERSIONS - replace with either a better ollama or a higher quality claude
        #LATER VERSONS - split input query into multiple parts to search instead (query decomposition)
        prior_messages = st.session_state.messages[:-1]
        has_history = len(prior_messages) > 0
        recent_turns = "\n".join(f'{m["role"]}: {m["content"]}' for m in st.session_state.messages[-4:])
        if has_history:
            HyDE_prompt = f'''Given this recent conversation for context:
                        {recent_turns}
                        
                        Generate a short hypothetical passage answering this query in academic-style writing,
                        as if it was in a published scientific paper.  Resolve any pronouns or references in the 
                        query using the conversation above, but the passage itself must read as a self-contained 
                        academic excerpt with no reference to "the conversation," "as discussed above," or similar.

                        -Use standard, established terminology from the field freely.
                        -DO NOT invent specific named results, fabricated citations, or specific numeric values presented as findings
                        -DO NOT assume a specific method is used IF NOT EXPLICITLY MENTIONED, 
                        describe the general class of technique instead, but write with the tone of a real passage
                
                        Return ONLY a JSON object in EXACTLY this format, with no other text:
                        {{"answer": ["the passage you create"]}}
                        USE THIS QUERY:{prompt}
                        '''
        else:
            HyDE_prompt = f'''
                                
                        Generate a short hypothetical passage answering this query in academic-style writing,
                        as if it was in a published scientific paper.
        
                        -Use standard, established terminology from the field freely.
                        -DO NOT invent specific named results, fabricated citations, or specific numeric values presented as findings
                        -DO NOT assume a specific method is used IF NOT EXPLICITLY MENTIONED, 
                        describe the general class of technique instead, but write with the tone of a real passage
                        
                        Return ONLY a JSON object in EXACTLY this format, with no other text:
                        {{"answer": ["the passage you create"]}}
                        USE THIS QUERY:{prompt}
                        '''
        response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": HyDE_prompt,
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
            print("JSON DECODE FAILURE - if rerunning continues to fail, try upgrading HyDE llm or contact admin")
            print(f"Raw was: {response.content[0].text[:200]}")
            sys.exit()

        #Retrieve list of knowledge from vector database (top 15)
        retrieved_knowledge = retrieve(passage, VECTOR_DB, GROUP_INDEX)
        #sort through reponses
        all_context_chunks = []

        #have reranker reorder retrieved knowledge
        pairs = [[prompt, chunk["text"]] for chunk, _ in retrieved_knowledge]
        scores = v3_reranker.predict(pairs)
        #order in rerankers grade (descending order)
        ranked = sorted(zip(retrieved_knowledge, scores), key=lambda x: x[1], reverse=True)
        #unzip response from tuple
        v3_unzipped = [(c, score) for (c, _), score in ranked]
        #pass both the initial retrieved knowledge and the reranked data through RRF (return top 7 responses)
        rrf_list = rrf_combine(retrieved_knowledge, v3_unzipped)
        #extend by tied chunks if they exist     
        all_context_chunks.extend(expand_groups(rrf_list))
        context = '\n'.join([
            f'Source: {chunk["metadata"]["article_title"]} - {chunk["metadata"]["heading"]} - {chunk["metadata"]["article_authors"]}\n{chunk["text"]}'
            for chunk in all_context_chunks
        ])

        #add context to the stream data
        st.session_state.current_context = context

    system_prompt = f'''You are a helpful chatbot providing assistance in matters concerning Physics and Astronomy. 
                Use only the following pieces of context to answer the question. Any logical leaps you make MUST be grounded in the context provided. 
                After EACH claim made, LIST THE SOURCES THE INFORMATION CAME FROM. PROVIDE SECTION AND TITLE IF POSSIBLE.

                IF there are gaps of information in the context presented - ADD AN ADDENDUM to the bottom of response and add
                information from your OWN data. MAKE IT CLEAR THIS IS BEING DONE, SUPPORT ANY CLAIMS YOU MAKE WITH A SOURCE.
                        
                Context to use:{st.session_state.current_context}
                '''

    with st.chat_message("assistant"):
        def stream_answer():
            with client.messages.stream(
                model="claude-sonnet-5",
                max_tokens = 3500,
                system = system_prompt,
                messages = st.session_state.messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        full_response = st.write_stream(stream_answer)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


