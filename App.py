import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI  # ✅ nouvelle importation

# Charger la clé d’API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # ✅ nouveau client

# Fonction pour extraire le texte du PDF
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Fonction pour créer des embeddings & index FAISS
def build_index(text, model_name="all-miniLM-L6-v2", chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks, embeddings, model

# Fonction pour récupérer les meilleurs k fragments selon la requête
def retrieve(query, index, chunks, model, k=5):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    results = [chunks[i] for i in I[0]]
    return results

# Fonction pour générer la réponse via OpenAI (nouvelle API)
def generate_answer(context, question, model_name="gpt-3.5-turbo"):
    prompt = f"""Vous êtes un assistant très utile. Utilisez le contexte suivant pour répondre à la question :\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nRéponse :"""
    
    # ✅ nouvelle syntaxe
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Vous êtes un assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        n=1,
        temperature=0.2,
    )

    answer = resp.choices[0].message.content.strip()
    return answer

# UI Streamlit
st.set_page_config(page_title="ChatPDF 📄")
st.title("Chat avec un document PDF")

uploaded_file = st.file_uploader("Téléversez un document PDF", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Extraction du texte…"):
        text = load_pdf(uploaded_file)
    st.success("Texte extrait !")

    if "index" not in st.session_state:
        with st.spinner("Création de l’index…"):
            idx, chunks, embeddings, model = build_index(text)
            st.session_state.index = idx
            st.session_state.chunks = chunks
            st.session_state.model_emb = model
        st.success("Index créé !")

    question = st.text_input("Posez votre question au document :")
    if question:
        with st.spinner("Recherche en cours…"):
            retrieved_chunks = retrieve(
                question,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.model_emb,
                k=5
            )
            context = "\n\n".join(retrieved_chunks)

        with st.spinner("Génération en cours…"):
            answer = generate_answer(context, question)

        st.markdown("**Réponse :**")
        st.write(answer)
