import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.base import LLM
from typing import Optional, List, Any
import os
from dotenv import load_dotenv

load_dotenv()

class GroqLLM(LLM):
    api_key: str
    model_name: str = "llama-3.1-8b-instant"  # ✅ updated to supported Groq model
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": self.temperature
        }
        if stop:
            data["stop"] = stop

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=data,
                headers=headers
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            st.error(f"Error calling Groq API: {e}")
            if e.response:
                st.error(f"Response body: {e.response.text}")
            return ""

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
st.title("📄 PDF Q&A Chatbot (with Groq API)")

with st.sidebar:
    st.header("Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")
    groq_api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", "")
    )

if pdf_file is None:
    st.info("👈 Upload a PDF to start.")
    st.stop()

if not groq_api_key:
    st.warning("Please enter your Groq API Key in the sidebar or set it in your .env file.")
    st.stop()

try:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text:
        st.error("Could not extract text from the PDF. Please ensure it is not an image-only PDF.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("💬 Ask a question about the PDF:")
    if user_question:
        docs = vector_store.similarity_search(user_question)
        llm = GroqLLM(api_key=groq_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        with st.spinner("Thinking..."):
            answer = chain.run(input_documents=docs, question=user_question)

        st.markdown("**Answer:**")
        st.write(answer)

except Exception as e:
    st.error(f"An error occurred: {e}")