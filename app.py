import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

st.title("RAG Study Assistant")
st.write("Upload a PDF and ask questions about it")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    st.write(f"Loaded {len(chunks)} chunks from your PDF")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    llm = OllamaLLM(model="llama3.2")

    question = st.text_input("Ask a question about your PDF")

    if question:
        with st.spinner("Thinking..."):
            docs = retriever.invoke(question)
            context = "\n\n".join([d.page_content for d in docs])
            prompt = f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
            answer = llm.invoke(prompt)
            st.write(answer)

            with st.expander("Sources"):
                for i, doc in enumerate(docs):
                    st.write(f"**Chunk {i + 1}:**")
                    st.write(doc.page_content)
                    st.divider()