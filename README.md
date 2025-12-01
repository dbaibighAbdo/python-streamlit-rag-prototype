# python-streamlit-rag-prototype

A small Streamlit prototype demonstrating a Retrieval-Augmented Generation (RAG) flow using LangChain and Google Generative AI (Gemini). This project shows how to index documents with FAISS, build a retriever, and run a prompt/LLM pipeline to answer user questions using context from the knowledge base.

**Status:** Prototype — experimental code for learning and proof-of-concept.

**Contents**
- `main.py`: Streamlit app entrypoint.
- `helpers/`: helper modules for file processing and RAG service logic.
- `FAISS_DB/`: a directory intended to hold FAISS index files (already contains an example index folder).

**Key libraries**
- `streamlit` — UI framework used for the demo.
- `langchain` and `langchain-google-genai` — orchestration of retriever + LLM.
- `python-dotenv` — load `.env` environment variables (e.g. `GOOGLE_API_KEY`).

## Prerequisites
- Python 3.12 or newer (pyproject requires `>=3.12`).
- A Google API key with access to Google Generative AI (Gemini) if you plan to use `langchain-google-genai`.

## Setup (Windows PowerShell)
1. Clone the repo and change directory:

```powershell
git clone <repo-url>
cd python-streamlit-rag-prototype
```

2. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies. This project uses `pyproject.toml` for metadata; install with `pip`:

```powershell
pip install --upgrade pip
pip install -r requirements.txt  # if you have a requirements file
# OR install directly from pyproject dependencies using pip:
pip install langchain langchain-google-genai langchain-ollama langchain-openai langchain-text-splitters pypdf2 python-dotenv streamlit
```

Note: If you prefer a packaging tool, you can use `poetry` to install dependencies from `pyproject.toml`.

4. Create a `.env` file in the project root with the Google API key (example):

```text
GOOGLE_API_KEY=your_google_api_key_here
# other keys if needed
```

## Running the app
Start the Streamlit app from the project root:

```powershell
streamlit run main.py
```

Open the URL shown by Streamlit in your browser (usually `http://localhost:8501`).

## How the RAG flow works (high level)
- Documents are processed and indexed into a FAISS index (see `helpers/fileProccessing.py`).
- A retriever is created from that index and used to fetch context documents for a user query.
- The `helpers/ragService.py` builds a small LangChain-style pipeline that composes a prompt with the retriever output and sends it to the LLM (Google Generative AI by default). Results are returned to the Streamlit UI.

## Troubleshooting
- TypeError about "Expected a Runnable, callable or dict": ensure that LangChain pipeline components are passed the correct types — do not wrap strings in `{}` which creates a `set` in Python. See `helpers/ragService.py` for an example of correct usage.
- If you get authentication errors from Google, verify `GOOGLE_API_KEY` in `.env` and that it's exported in your environment.
- If you see missing package errors, re-run the install step or install the specific package shown in the error message.

## Project structure (quick)
- `main.py` — Streamlit UI and wiring.
- `helpers/fileProccessing.py` — ingestion and document splitting helpers.
- `helpers/ragService.py` — builds the LangChain flow (prompts, retriever, LLM invocation).
- `FAISS_DB/` — place to store generated FAISS indexes.

## Notes & next steps
- This repository is intended as a minimal prototype. For production use, add error handling, secrets management, input validation, and unit tests.
- Consider adding a `requirements.txt` for simpler installs or integrate `poetry`/`pip-tools` to manage pinned dependencies.

If you want, I can:
- run the app here and inspect runtime tracebacks
- add a `requirements.txt` generated from the `pyproject.toml`
- include a short example of how to build the FAISS index from a folder of PDFs

---
Generated on December 1, 2025.
