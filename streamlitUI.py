import base64
import os
import re
import tempfile
import uuid

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env
load_dotenv()

# ---------------------------------------------------
# Streamlit page settings
# ---------------------------------------------------
st.set_page_config(page_title="PDF Assistant", layout="wide")

# ---------------------------------------------------
# App configuration
# ---------------------------------------------------
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
CHROMA_DIR = "vectorstore_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------
# Session state
# ---------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")


# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def clean_filename(filename: str) -> str:
    """
    Clean the uploaded file name so it can safely be used as a Chroma collection name.
    """
    filename = re.sub(r"\s\(\d+\)", "", filename)
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return filename


def display_pdf(uploaded_file):
    """
    Show the uploaded PDF in the right side panel.
    """
    bytes_data = uploaded_file.getvalue()
    base64_pdf = base64.b64encode(bytes_data).decode("utf-8")

    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="900"
        type="application/pdf">
    </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


def get_pdf_text(uploaded_file):
    """
    Read the uploaded PDF file and convert it into LangChain documents.
    """
    temp_file = None
    try:
        file_bytes = uploaded_file.getvalue()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(file_bytes)
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        return documents

    finally:
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except FileNotFoundError:
                pass


def split_document(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split PDF text into smaller chunks so retrieval works better.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " "],
    )
    return splitter.split_documents(documents)


@st.cache_resource
def get_embedding_function():
    """
    Hugging Face embeddings run locally, so no API key is needed here.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_vectorstore(chunks, embedding_function, vectorstore_path, file_name):
    """
    Create a Chroma vector store from PDF chunks and save it locally.
    """
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]

    unique_ids = set()
    unique_chunks = []
    final_ids = []

    for chunk, doc_id in zip(chunks, ids):
        if doc_id not in unique_ids:
            unique_ids.add(doc_id)
            unique_chunks.append(chunk)
            final_ids.append(doc_id)

    vectorstore = Chroma.from_documents(
        documents=unique_chunks,
        embedding=embedding_function,
        ids=final_ids,
        persist_directory=vectorstore_path,
        collection_name=clean_filename(file_name),
    )

    return vectorstore


def create_vectorstore_from_texts(documents, file_name):
    """
    Split the PDF into chunks, create embeddings, and store them in Chroma.
    """
    chunks = split_document(documents, chunk_size=1000, chunk_overlap=200)
    embedding_function = get_embedding_function()

    return create_vectorstore(
        chunks=chunks,
        embedding_function=embedding_function,
        vectorstore_path=CHROMA_DIR,
        file_name=file_name,
    )


def format_docs(docs):
    """
    Join retrieved documents into one text block.
    """
    return "\n\n".join(doc.page_content for doc in docs)


class PaperInfo(BaseModel):
    """
    Structured output for research-paper extraction.
    Keep the fields flat so table display is simple and reliable.
    """
    title: str = Field(description="Paper title")
    summary: str = Field(description="Paper summary")
    publication_date: str = Field(description="Publication date or year")
    authors: str = Field(description="Paper authors")


def query_document(vectorstore, query, groq_api_key):
    """
    Retrieve the most relevant PDF chunks and ask Groq to extract paper metadata.
    Returns a pandas DataFrame so Streamlit can show it as a table.
    """
    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        groq_api_key=groq_api_key,
        temperature=0,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    prompt_template = ChatPromptTemplate.from_template(
        """
You are extracting structured data from a research paper.

Use ONLY the context below. Do not invent anything.
If a field is missing, write "Not found in document".

Context:
{context}

Task:
{question}
"""
    )

    # Try structured output first.
    # If the model/provider returns something unexpected, fall back to plain parsing.
    structured_llm = llm.with_structured_output(PaperInfo)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | structured_llm
    )

    try:
        result = rag_chain.invoke(query)

        # result can be a Pydantic object or dict depending on runtime/version
        if isinstance(result, dict):
            paper = PaperInfo(**result)
        else:
            paper = result

        table_df = pd.DataFrame(
            [
                {"Field": "Title", "Value": paper.title},
                {"Field": "Summary", "Value": paper.summary},
                {"Field": "Publication Date", "Value": paper.publication_date},
                {"Field": "Authors", "Value": paper.authors},
            ]
        )
        return table_df

    except Exception as e:
        # Fallback: ask for plain text and parse manually.
        fallback_prompt = ChatPromptTemplate.from_template(
            """
You are extracting structured data from a research paper.

Use ONLY the context below. Do not invent anything.
Return EXACTLY in this format:

Title: ...
Summary: ...
Publication Date: ...
Authors: ...

Context:
{context}

Task:
{question}
"""
        )

        fallback_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | fallback_prompt
            | llm
        )

        response = fallback_chain.invoke(query)
        text = response.content

        def extract_value(label, source_text):
            match = re.search(rf"{label}:\s*(.*)", source_text)
            return match.group(1).strip() if match else "Not found in document"

        table_df = pd.DataFrame(
            [
                {"Field": "Title", "Value": extract_value("Title", text)},
                {"Field": "Summary", "Value": extract_value("Summary", text)},
                {"Field": "Publication Date", "Value": extract_value("Publication Date", text)},
                {"Field": "Authors", "Value": extract_value("Authors", text)},
            ]
        )
        return table_df


# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.title("PDF Assistant")
st.caption("Upload a PDF, create embeddings with Hugging Face, and extract paper details with Groq.")

col1, col2 = st.columns([0.42, 0.58], gap="large")

with col1:
    st.subheader("Settings")

    # Groq API key: user can enter it here, or load from .env
    st.session_state.groq_api_key = st.text_input(
        "Groq API key",
        type="password",
        value=st.session_state.groq_api_key,
        help="Used only for the Groq chat model.",
    )

    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
    )

    st.divider()

    # Fixed extraction task for structured output
    query_text = "Extract the title, summary, publication date, and authors of the research paper."

    generate_clicked = st.button("Generate table")

with col2:
    if uploaded_file is not None:
        display_pdf(uploaded_file)

# ---------------------------------------------------
# Build vector store when a new PDF is uploaded
# ---------------------------------------------------
if uploaded_file is not None:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        with st.spinner("Reading PDF and building vector store..."):
            documents = get_pdf_text(uploaded_file)

            st.session_state.vector_store = create_vectorstore_from_texts(
                documents=documents,
                file_name=uploaded_file.name,
            )

            st.session_state.uploaded_file_name = uploaded_file.name
            st.success("PDF processed successfully.")

# ---------------------------------------------------
# Generate structured table
# ---------------------------------------------------
with col1:
    if generate_clicked:
        if not st.session_state.groq_api_key:
            st.error("Please enter your Groq API key first.")
        elif st.session_state.vector_store is None:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Extracting structured data..."):
                table_df = query_document(
                    vectorstore=st.session_state.vector_store,
                    query=query_text,
                    groq_api_key=st.session_state.groq_api_key,
                )

            st.subheader("Extracted Data")
            st.dataframe(table_df, use_container_width=True, hide_index=True)