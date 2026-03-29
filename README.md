# 📄 PDF Assistant (RAG App)

A simple AI-powered app that extracts structured information (title, summary, authors, etc.) from research papers using **Hugging Face embeddings + Groq LLM + Streamlit UI**.

---

## 🚀 Features

* 📂 Upload any PDF
* 🧠 Local embeddings using Hugging Face (no API needed)
* ⚡ Fast responses using Groq LLM
* 📊 Structured output in table format
* 🖥️ Clean Streamlit UI with PDF preview

---

## 🧩 Tech Stack

* **Frontend:** Streamlit
* **Embeddings:** Hugging Face (`sentence-transformers`)
* **Vector DB:** Chroma
* **LLM:** Groq (`llama-3.1-8b-instant`)
* **Framework:** LangChain

---

## ⚙️ Installation

```bash
git clone https://github.com/HimCodex/Structured-Data-FromPDFs.git
cd Structured-Data-FromPDFs

pip install -U streamlit python-dotenv langchain langchain-community langchain-core \
langchain-text-splitters langchain-huggingface langchain-groq langchain-chroma \
sentence-transformers chromadb pypdf pydantic
```

---

## 🔑 Setup

Create a `.env` file in the root folder:

```
GROQ_API_KEY=your_api_key_here
```

Or enter your API key directly in the app UI.

---

## ▶️ Run the App

```bash
streamlit run streamlitUI.py
```

---

## 🧠 How It Works

1. Upload PDF
2. Text is extracted and split into chunks
3. Chunks → converted into embeddings (Hugging Face)
4. Stored in Chroma vector database
5. Relevant chunks retrieved for query
6. Groq LLM generates structured output

---

## 📊 Output Example

| Field            | Value |
| ---------------- | ----- |
| Title            | ...   |
| Summary          | ...   |
| Publication Date | ...   |
| Authors          | ...   |

---

## 📌 Notes

* No embedding API cost (runs locally)
* Groq is used only for answer generation
* Works best with research papers / structured PDFs

---

## 🚀 Future Improvements

* Download results as CSV
* Multi-model support (OpenAI / Gemini)
* Chat with PDF (multi-turn)
* Better UI styling

---

## 👨‍💻 Author

Himanshu

--------
