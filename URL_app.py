import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_classic.chains import RetrievalQA

st.set_page_config(page_title="🌐 Webpage Q&A (Chroma + Mistral)", layout="centered")
st.title("🌐 Webpage Q&A Chatbot")

mistral_key = os.getenv("MISTRAL_API_KEY")

# --- Input URL ---
url = st.text_input("Enter a webpage URL:")

# --- Initialize session state ---
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if st.button("Ingest Webpage"):
    if not url.strip():
        st.warning("Please enter a valid URL.")
    else:
        with st.spinner("Fetching and embedding webpage content..."):
            # Load and process
            loader = WebBaseLoader(url)
            docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-V2")

            # Create Chroma DB
            db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_store")
            db.persist()

            # Initialize Mistral LLM
            llm = ChatMistralAI(model="open-mistral-7b", api_key=mistral_key)

            # Create QA chain and store it
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(),
            )

        st.success("✅ Webpage ingested successfully! You can now ask questions.")

# --- Ask questions if QA chain is available ---
if st.session_state.qa_chain:
    query = st.text_input("Ask a question about the webpage:")
    if st.button("Ask"):
        if query.strip():
            with st.spinner("Generating answer..."):
                result = st.session_state.qa_chain.invoke({"query": query})
                st.markdown("### 🧠 Answer:")
                st.write(result["result"])
        else:
            st.warning("Please enter a question.")
else:
    st.info("👆 Ingest a webpage first to enable Q&A.")
