# =========================================================
# AI IMPLEMENTATION ENABLEMENT ASSISTANT
# STABLE MVP VERSION
# =========================================================

import streamlit as st
import google.generativeai as genai
import tempfile
import os
import uuid
import numpy as np
import faiss
import fitz

from docx import Document
from pptx import Presentation
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Implementation Enablement Assistant",
    layout="wide"
)

st.title(
    "AI Implementation Enablement Assistant"
)

st.caption(
    "Upload implementation artifacts and generate team-specific enablement outputs"
)

# =========================================================
# CONFIG
# =========================================================

MAX_CONTEXT_LENGTH = 12000
TOP_K_RESULTS = 3

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Configuration")

gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password"
)

if gemini_api_key:

    genai.configure(
        api_key=gemini_api_key
    )

# =========================================================
# IMAGE STORAGE
# =========================================================

IMAGE_DIR = "extracted_images"

os.makedirs(
    IMAGE_DIR,
    exist_ok=True
)

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

if "images" not in st.session_state:
    st.session_state.images = []

if "processed" not in st.session_state:
    st.session_state.processed = False

# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf_content(
    file_path,
    file_name
):

    text = ""

    extracted_images = []

    try:

        doc = fitz.open(file_path)

        for page in doc:

            page_text = page.get_text()

            text += page_text + "\n"

            image_list = page.get_images(
                full=True
            )

            for img in image_list:

                try:

                    xref = img[0]

                    base_image = (
                        doc.extract_image(xref)
                    )

                    image_bytes = (
                        base_image["image"]
                    )

                    image_ext = (
                        base_image["ext"]
                    )

                    image_name = (
                        f"{uuid.uuid4()}.{image_ext}"
                    )

                    image_path = os.path.join(
                        IMAGE_DIR,
                        image_name
                    )

                    with open(
                        image_path,
                        "wb"
                    ) as f:

                        f.write(image_bytes)

                    extracted_images.append({
                        "image_path": image_path,
                        "source_file": file_name,
                        "related_text":
                            page_text[:500]
                    })

                except:
                    pass

    except:
        pass

    return text, extracted_images

# =========================================================
# DOCX EXTRACTION
# =========================================================

def extract_docx_content(
    file_path,
    file_name
):

    text = ""

    extracted_images = []

    try:

        doc = Document(file_path)

        text = "\n".join(
            [
                para.text
                for para in doc.paragraphs
            ]
        )

        rels = doc.part._rels

        for rel in rels:

            rel_obj = rels[rel]

            if "image" in rel_obj.target_ref:

                try:

                    image_data = (
                        rel_obj.target_part.blob
                    )

                    image_name = (
                        f"{uuid.uuid4()}.png"
                    )

                    image_path = os.path.join(
                        IMAGE_DIR,
                        image_name
                    )

                    with open(
                        image_path,
                        "wb"
                    ) as f:

                        f.write(image_data)

                    extracted_images.append({
                        "image_path": image_path,
                        "source_file": file_name,
                        "related_text":
                            text[:500]
                    })

                except:
                    pass

    except:
        pass

    return text, extracted_images

# =========================================================
# PPTX EXTRACTION
# =========================================================

def extract_pptx_content(
    file_path,
    file_name
):

    text = ""

    extracted_images = []

    try:

        prs = Presentation(file_path)

        for slide in prs.slides:

            slide_text = ""

            for shape in slide.shapes:

                if hasattr(shape, "text"):

                    slide_text += (
                        shape.text + "\n"
                    )

                # Picture shape
                if shape.shape_type == 13:

                    try:

                        image = shape.image

                        image_bytes = image.blob

                        image_ext = image.ext

                        image_name = (
                            f"{uuid.uuid4()}.{image_ext}"
                        )

                        image_path = os.path.join(
                            IMAGE_DIR,
                            image_name
                        )

                        with open(
                            image_path,
                            "wb"
                        ) as f:

                            f.write(image_bytes)

                        extracted_images.append({
                            "image_path":
                                image_path,

                            "source_file":
                                file_name,

                            "related_text":
                                slide_text[:500]
                        })

                    except:
                        pass

            text += slide_text + "\n"

    except:
        pass

    return text, extracted_images

# =========================================================
# TXT EXTRACTION
# =========================================================

def extract_txt_content(
    file_path
):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()

        return text, []

    except:

        return "", []

# =========================================================
# GENERIC EXTRACTION
# =========================================================

def extract_content(uploaded_file):

    suffix = (
        uploaded_file.name
        .split(".")[-1]
        .lower()
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{suffix}"
    ) as tmp:

        tmp.write(
            uploaded_file.read()
        )

        temp_path = tmp.name

    try:

        if suffix == "pdf":

            text, images = (
                extract_pdf_content(
                    temp_path,
                    uploaded_file.name
                )
            )

        elif suffix == "docx":

            text, images = (
                extract_docx_content(
                    temp_path,
                    uploaded_file.name
                )
            )

        elif suffix == "pptx":

            text, images = (
                extract_pptx_content(
                    temp_path,
                    uploaded_file.name
                )
            )

        elif suffix == "txt":

            text, images = (
                extract_txt_content(
                    temp_path
                )
            )

        else:

            text = ""
            images = []

    finally:

        os.unlink(temp_path)

    return text, images

# =========================================================
# CHUNKING
# =========================================================

def chunk_text(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return chunks

# =========================================================
# EMBEDDINGS
# =========================================================

def get_embedding(text):

    try:

        text = text[:3000]

        embedding = (
            embedding_model.encode(text)
        )

        return embedding

    except:

        return np.zeros(384)

# =========================================================
# VECTOR STORE
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

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index

# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_chunks(
    query,
    top_k=TOP_K_RESULTS
):

    try:

        query_embedding = get_embedding(
            query
        )

        query_embedding = np.array(
            [query_embedding]
        ).astype("float32")

        distances, indices = (
            st.session_state.vector_store.search(
                query_embedding,
                top_k
            )
        )

        retrieved = []

        for idx in indices[0]:

            retrieved.append({
                "chunk":
                    st.session_state.chunks[idx],

                "metadata":
                    st.session_state.metadata[idx]
            })

        return retrieved

    except:

        return []

# =========================================================
# IMAGE RETRIEVAL
# =========================================================

def retrieve_related_images(
    retrieved_chunks
):

    matched_images = []

    try:

        for chunk in retrieved_chunks:

            chunk_text = (
                chunk["chunk"].lower()
            )

            chunk_words = (
                chunk_text.split()[:15]
            )

            for image_data in (
                st.session_state.images
            ):

                related_text = (
                    image_data["related_text"]
                    .lower()
                )

                if any(
                    word in related_text
                    for word in chunk_words
                ):

                    matched_images.append(
                        image_data
                    )

        unique_images = []

        seen = set()

        for img in matched_images:

            if (
                img["image_path"]
                not in seen
            ):

                unique_images.append(img)

                seen.add(
                    img["image_path"]
                )

        return unique_images[:3]

    except:

        return []

# =========================================================
# PROMPT BUILDER
# =========================================================

def build_prompt(
    team,
    context,
    additional_instruction
):

    if team == "Functional Team":

        instructions = """
        Generate a functional enablement document.

        Include:
        - business flow
        - process explanation
        - validations
        - dependencies
        - configurations
        - assumptions
        """

    elif team == "Technical Team":

        instructions = """
        Generate a technical analysis document.

        Include:
        - impacted modules
        - integrations
        - technical dependencies
        - customization points
        - validations
        """

    else:

        instructions = """
        Generate a support readiness document.

        Include:
        - troubleshooting guidance
        - FAQs
        - issue scenarios
        - checkpoints
        - limitations
        """

    prompt = f"""
    You are an enterprise AI assistant.

    TARGET TEAM:
    {team}

    CONTEXT:
    {context}

    ADDITIONAL INSTRUCTIONS:
    {additional_instruction}

    TASK:
    {instructions}

    RULES:
    - Use only provided context
    - Do not hallucinate
    - Keep concise
    - Use structured formatting
    - Mention assumptions if needed
    """

    return prompt

# =========================================================
# OUTPUT GENERATION
# =========================================================

def generate_output(
    team,
    additional_instruction
):

    retrieval_query = f"""
    Enterprise implementation details
    for {team}

    {additional_instruction}
    """

    retrieved_chunks = retrieve_chunks(
        retrieval_query
    )

    if len(retrieved_chunks) == 0:

        return (
            "No relevant content found.",
            []
        )

    context = ""

    for item in retrieved_chunks:

        context += f"""

        FILE:
        {item['metadata']['file_name']}

        CONTENT:
        {item['chunk']}

        --------------------------------
        """

    # VERY IMPORTANT
    context = context[:MAX_CONTEXT_LENGTH]

    prompt = build_prompt(
        team,
        context,
        additional_instruction
    )

    related_images = (
        retrieve_related_images(
            retrieved_chunks
        )
    )

    try:

        model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        response = model.generate_content(
            prompt
        )

        output_text = response.text

    except Exception as e:

        output_text = f"""
        Error generating output:

        {str(e)}
        """

    return (
        output_text,
        related_images
    )

# =========================================================
# SCREEN 1 - UPLOAD
# =========================================================

st.header(
    "1. Upload Implementation Artifacts"
)

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
        all_images = []

        st.subheader(
            "Processing Files..."
        )

        for file in uploaded_files:

            st.write(
                f"Reading: {file.name}"
            )

            extracted_text, extracted_images = (
                extract_content(file)
            )

            all_images.extend(
                extracted_images
            )

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
            "Creating Knowledge Index..."
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

        st.session_state.images = (
            all_images
        )

        st.session_state.processed = True

        st.success(
            "Knowledge Processing Completed"
        )

# =========================================================
# SCREEN 2 - GENERATE OUTPUT
# =========================================================

if st.session_state.processed:

    st.header(
        "2. Generate Team-Specific Output"
    )

    team = st.selectbox(
        "Generate For",
        [
            "Functional Team",
            "Technical Team",
            "Support Team"
        ]
    )

    additional_instruction = st.text_area(
        "Additional Instructions (Optional)",
        placeholder="""
Examples:
- Focus on approval workflow
- Include troubleshooting
- Explain integrations
- Keep concise
        """
    )

    if st.button(
        "Generate Output"
    ):

        with st.spinner(
            "Generating Output..."
        ):

            output, images = (
                generate_output(
                    team,
                    additional_instruction
                )
            )

        st.subheader(
            "Generated Output"
        )

        st.markdown(output)

        if images:

            st.subheader(
                "Relevant Screenshots"
            )

            for img in images:

                try:

                    st.image(
                        img["image_path"],
                        caption=img["source_file"],
                        use_container_width=True
                    )

                except:
                    pass

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Stable MVP | AI Implementation Enablement Assistant"
)
