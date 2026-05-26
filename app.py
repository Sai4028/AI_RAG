# =========================================================
# AI GTM RAG MVP - PHASE 1
# Streamlit + Gemini + FAISS + SentenceTransformers
# =========================================================

import streamlit as st
import google.generativeai as genai
import tempfile
import os
import numpy as np
import faiss
import fitz

from docx import Document
from pptx import Presentation

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from sentence_transformers import (
    SentenceTransformer
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI GTM RAG MVP",
    layout="wide"
)

st.title("AI GTM RAG MVP")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Configuration")

gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password"
)

if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model

embedding_model = load_embedding_model()

# =========================================================
# SESSION STATE
# =========================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "metadata" not in st.session_state:
    st.session_state.metadata = []

if "processed" not in st.session_state:
    st.session_state.processed = False

# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_text(file_path):

    text = ""

    doc = fitz.open(file_path)

    for page in doc:
        text += page.get_text()

    return text

# =========================================================
# DOCX EXTRACTION
# =========================================================

def extract_docx_text(file_path):

    doc = Document(file_path)

    text = "\n".join(
        [para.text for para in doc.paragraphs]
    )

    return text

# =========================================================
# PPTX EXTRACTION
# =========================================================

def extract_pptx_text(file_path):

    prs = Presentation(file_path)

    text = ""

    for slide in prs.slides:

        for shape in slide.shapes:

            if hasattr(shape, "text"):
                text += shape.text + "\n"

    return text

# =========================================================
# GENERIC FILE EXTRACTION
# =========================================================

def extract_text(uploaded_file):

    suffix = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{suffix}"
    ) as tmp:

        tmp.write(uploaded_file.read())

        temp_path = tmp.name

    try:

        if suffix == "pdf":
            text = extract_pdf_text(temp_path)

        elif suffix == "docx":
            text = extract_docx_text(temp_path)

        elif suffix == "pptx":
            text = extract_pptx_text(temp_path)

        elif suffix == "txt":

            with open(
                temp_path,
                "r",
                encoding="utf-8"
            ) as f:

                text = f.read()

        else:
            text = ""

    finally:

        os.unlink(temp_path)

    return text

# =========================================================
# CHUNKING
# =========================================================

def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    return chunks

# =========================================================
# EMBEDDINGS
# =========================================================

def get_embedding(text):

    text = text[:8000]

    embedding = embedding_model.encode(text)

    return embedding

# =========================================================
# VECTOR STORE CREATION
# =========================================================

def create_vector_store(chunks):

    embeddings = []

    progress_bar = st.progress(0)

    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks):

        embedding = get_embedding(chunk)

        embeddings.append(embedding)

        progress_bar.progress(
            (idx + 1) / total_chunks
        )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_chunks(query, top_k=5):

    query_embedding = get_embedding(query)

    query_embedding = np.array(
        [query_embedding]
    ).astype("float32")

    distances, indices = st.session_state.vector_store.search(
        query_embedding,
        top_k
    )

    retrieved = []

    for idx in indices[0]:

        retrieved.append({
            "chunk": st.session_state.chunks[idx],
            "metadata": st.session_state.metadata[idx]
        })

    return retrieved

# =========================================================
# GEMINI OUTPUT GENERATION
# =========================================================

def generate_output(role, output_type):

    retrieval_query = f"""
    Enterprise implementation details for:
    {role}
    related to:
    {output_type}
    """

    retrieved_chunks = retrieve_chunks(
        retrieval_query
    )

    context = ""

    for item in retrieved_chunks:

        context += f"""
        FILE: {item['metadata']['file_name']}

        CONTENT:
        {item['chunk']}

        -----------------------------------
        """

    prompt = f"""
    You are an enterprise AI assistant.

    ROLE:
    {role}

    OUTPUT TYPE:
    {output_type}

    CONTEXT:
    {context}

    INSTRUCTIONS:
    - Use only provided context
    - Do not hallucinate
    - Generate enterprise-ready output
    - Use structured formatting
    - Be practical and implementation-focused
    - Mention assumptions if context is insufficient
    """

    model = genai.GenerativeModel(
        "gemini-1.5-pro"
    )

    response = model.generate_content(
        prompt
    )

    return response.text

# =========================================================
# SCREEN 1 - UPLOAD
# =========================================================

st.header("1. Upload Artifacts")

uploaded_files = st.file_uploader(
    "Upload PDF / DOCX / PPTX / TXT",
    type=["pdf", "docx", "pptx", "txt"],
    accept_multiple_files=True
)

# =========================================================
# PROCESS FILES
# =========================================================

if uploaded_files:

    if st.button("Process Files"):

        if not gemini_api_key:

            st.error(
                "Please enter Gemini API Key"
            )

            st.stop()

        all_chunks = []
        all_metadata = []

        st.subheader("Processing Files")

        for file in uploaded_files:

            st.write(
                f"Reading: {file.name}"
            )

            extracted_text = extract_text(file)

            if not extracted_text.strip():

                st.warning(
                    f"No readable text found in {file.name}"
                )

                continue

            chunks = chunk_text(
                extracted_text
            )

            for idx, chunk in enumerate(chunks):

                all_chunks.append(chunk)

                all_metadata.append({
                    "file_name": file.name,
                    "chunk_id": idx
                })

        if len(all_chunks) == 0:

            st.error(
                "No valid content extracted"
            )

            st.stop()

        st.subheader(
            "Creating Vector Store"
        )

        vector_store = create_vector_store(
            all_chunks
        )

        st.session_state.vector_store = (
            vector_store
        )

        st.session_state.chunks = (
            all_chunks
        )

        st.session_state.metadata = (
            all_metadata
        )

        st.session_state.processed = True

        st.success(
            "Files Processed Successfully"
        )

# =========================================================
# SCREEN 2 - AI UNDERSTANDING
# =========================================================

if st.session_state.processed:

    st.header("2. AI Understanding")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Total Chunks",
            len(st.session_state.chunks)
        )

    with col2:

        st.metric(
            "Documents Processed",
            len(uploaded_files)
        )

    with st.expander(
        "View Sample Chunks"
    ):

        for idx, chunk in enumerate(
            st.session_state.chunks[:5]
        ):

            st.markdown(
                f"### Chunk {idx + 1}"
            )

            st.write(
                chunk[:1000]
            )

# =========================================================
# SCREEN 3 - ROLE BASED OUTPUTS
# =========================================================

if st.session_state.processed:

    st.header(
        "3. Role-Based Output Generation"
    )

    col1, col2 = st.columns(2)

    with col1:

        role = st.selectbox(
            "Select Role",
            [
                "Functional Team",
                "Technical Team",
                "Support Team"
            ]
        )

    with col2:

        output_type = st.selectbox(
            "Select Output Type",
            [
                "SOP",
                "Release Notes",
                "FAQ",
                "Training Guide",
                "Technical Summary",
                "Troubleshooting Guide"
            ]
        )

    if st.button(
        "Generate Output"
    ):

        with st.spinner(
            "Generating AI Output..."
        ):

            output = generate_output(
                role,
                output_type
            )

        st.subheader(
            "Generated Output"
        )

        st.markdown(output)

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Phase-1 MVP | Streamlit + Gemini + FAISS + SentenceTransformers"
)
