# 🌐 Webpage Q&A Chatbot

This Streamlit app lets you enter a **webpage URL**, reads the content, stores it using **Chroma**, and answers your questions with the **Mistral AI model**.

---

## 🚀 Features

* Enter any webpage URL
* Extracts and splits webpage text
* Stores embeddings in **Chroma**
* Uses **Mistral** LLM for answers
* Built with **LangChain** and **Streamlit**

---

## 🛠️ Setup

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```txt
streamlit
langchain
langchain-community
langchain-classic
langchain-mistralai
sentence-transformers
chromadb
beautifulsoup4
requests
```

### 2️⃣ Add your API key

In GitHub Codespaces or locally:

```
MISTRAL_API_KEY=your_mistral_api_key
```

---

## ▶️ Run the app

```bash
streamlit run URL_app.py
```

Then open the link shown in the terminal.

