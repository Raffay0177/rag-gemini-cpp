"""
RAG with LangChain + Gemini + ChromaDB + Local Embeddings (SentenceTransformers)

Steps:
1. pip install langchain langchain-community langchain-google-genai chromadb sentence-transformers pdfminer.six streamlit tqdm
2. export GOOGLE_API_KEY="your_api_key"
3. Place your PDF in ./books/programming_fundamentals_cpp.pdf
4. Build vectorstore:
   python rag_gemini_cpp_langchain.py --pdf ./books/programming_fundamentals_cpp.pdf
5. Run query:
   python rag_gemini_cpp_langchain.py --query "Explain pointers in C++"
6. Launch UI:
   streamlit run rag_gemini_cpp_langchain.py -- --run-ui
"""

import os
import argparse
from pathlib import Path
import re
from tqdm import tqdm
import streamlit as st

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -----------------------
# Config
# -----------------------
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
INDEX_DIR = Path("./chroma_store")
INDEX_DIR.mkdir(exist_ok=True)
COLLECTION_NAME = "cpp_book"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Set GOOGLE_API_KEY in your environment.")

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------
# Chunking
# -----------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(i + chunk_size, len(words))
        chunk = " ".join(words[i:j])
        chunks.append(chunk)
        i = j - overlap
        if i <= 0:
            i = j
    return chunks

# -----------------------
# PDF Parsing + Indexing
# -----------------------

def ingest_pdf(pdf_path: str):
    texts = []
    print(f"📖 Parsing PDF: {pdf_path}")
    for page_layout in tqdm(extract_pages(pdf_path), desc="Extracting pages"):
        page_text = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                page_text.append(element.get_text())
        clean_text = " ".join(page_text)
        clean_text = re.sub(r"\s+", " ", clean_text)
        page_chunks = chunk_text(clean_text)
        texts.extend(page_chunks)

    print(f"✅ Parsed {len(texts)} chunks. Now generating embeddings...")

    # Initialize vectorstore
    vectordb = Chroma(
        persist_directory=str(INDEX_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model
    )

    # Add with progress bar
    for i in tqdm(range(0, len(texts), 20), desc="Embedding + Storing"):
        batch = texts[i:i+20]
        vectordb.add_texts(batch)

    vectordb.persist()
    print("✅ Index saved in Chroma.")

def load_vectorstore():
    return Chroma(
        persist_directory=str(INDEX_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model
    )

# -----------------------
# RAG Chain
# -----------------------

def build_qa_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    prompt = PromptTemplate(
        template="""
You are an expert C++ programming instructor.
Use the following context (book excerpts) to answer the question step by step.
Give clear explanations and include short C++ code examples if helpful.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="LangChain RAG with Gemini + Chroma + SentenceTransformers")
    parser.add_argument("--pdf", type=str, help="Path to PDF for indexing")
    parser.add_argument("--query", type=str, help="Ask a question about the book")
    parser.add_argument("--run-ui", action="store_true", help="Run Streamlit UI")
    args = parser.parse_args()

    if args.pdf:
        ingest_pdf(args.pdf)
    elif args.query:
        qa_chain = build_qa_chain()
        ans = qa_chain.invoke({"query": args.query})
        print("\n🤖 Answer:\n")
        print(ans)
    elif args.run_ui:
        run_streamlit_app()
    else:
        print("Usage:\n --pdf <path>\n --query \"...\"\n --run-ui")

# -----------------------
# Streamlit UI
# -----------------------

def run_streamlit_app():
    st.title("📘 LangChain RAG — Programming Fundamentals in C++")
    q = st.text_input("Ask a question about C++:")
    top_k = st.slider("Top k retrieved chunks", 1, 10, 5)
    if st.button("Search & Explain") and q.strip():
        vectordb = load_vectorstore()
        retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
        qa_chain = build_qa_chain()
        with st.spinner("Generating answer..."):
            ans = qa_chain.invoke({"query": q})
        st.markdown("**Answer:**")
        st.write(ans)

if __name__ == "__main__":
    main()
