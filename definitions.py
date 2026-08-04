import streamlit as st
import pickle
import anthropic
from sentence_transformers import CrossEncoder

@st.cache_resource
def load_vector_db():
    with open('vector_db.pkl', 'rb') as f:
        db = pickle.load(f)
    print(f'Loaded {len(db)} chunks from disk')
    return db

@st.cache_resource
def load_group_index():
    with open('group_index.pkl', 'rb') as f:
        gi = pickle.load(f)
    print(f'Loaded {len(gi)} tied chunks from disk')
    return gi

@st.cache_resource
def load_reranker():
    return CrossEncoder("Reranker_models/V3_clean")

@st.cache_resource
def get_client():
    return anthropic.Anthropic()

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'