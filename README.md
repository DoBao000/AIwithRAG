# AIwithRAG

A Retrieval-Augmented Generation (RAG) application that allows you to upload documents and ask questions about their content using AI-powered semantic search and generation.

---

## Project Structure

```
AIwithRAG/
├── rag/
│   ├── __init__.py         # Package initializer
│   ├── chunks.py           # Text chunking logic
│   ├── embedder.py         # Embedding model loader
│   └── generator.py        # Answer generation with Claude/LLM
├── static/
│   └── index.html          # Frontend UI
├── uploads/                # Uploaded documents (auto-created)
├── .env                    # Environment variables (API keys, config)
├── .gitignore
├── app.py                  # Flask server — main entry point
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Implementation Process

The RAG pipeline works in the following stages:

1. **Document Upload** — User uploads a `.txt` or `.pdf` file via the frontend. The server saves it to the `uploads/` folder.

2. **Text Extraction** — `app.py` reads the file content. For PDFs, it uses `PyPDF2 (PdfReader)` to extract text page by page.

3. **Chunking** — The extracted text is split into smaller overlapping chunks using `rag/chunks.py` (`chunk_text`). This ensures each chunk fits within the model's context window.

4. **Embedding** — Each chunk is converted into a vector embedding using the model loaded in `rag/embedder.py`. All embeddings are stored in an in-memory `VECTOR_DB` list.

5. **Retrieval** — When the user asks a question, the question is also embedded and compared against stored chunk embeddings using cosine similarity to find the most relevant chunks.

6. **Generation** — The top retrieved chunks are passed as context to `rag/generator.py` (`generate`), which calls an LLM to produce a grounded answer.

---

## How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file and add your API key:

```
API_KEY=your_api_key_here
```

### 3. Start the server

```bash
python app.py
```

The server runs locally at `http://127.0.0.1:5500`.

### 4. Open the frontend

Open your browser and navigate to:

```
http://127.0.0.1:5500
```

The `index.html` frontend is served directly from the `static/` folder by Flask.

### 5. Upload & Query

- Click **Upload** to upload a `.txt` or `.pdf` file.
- Type a question in the input box and hit **Ask**.
- The app will retrieve relevant chunks from your document and return an AI-generated answer.

---

## Tech Stack

- **Backend:** Python, Flask
- **Embeddings:** Sentence Transformers (via `rag/embedder.py`)
- **Generation:** LLM API (configured in `rag/generator.py`)
- **Frontend:** HTML/CSS/JS (`static/index.html`)