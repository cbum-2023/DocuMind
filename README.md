# DocuMind

Ask questions to your PDFs. Upload a document, ask in plain English, get answers with sources.

Built as a simple, production-ready RAG project.

## How it works

1. Upload PDF -> text is split into chunks
2. Chunks are turned into embeddings and saved in ChromaDB
3. When you ask, we find the most similar chunks
4. GPT answers using only those chunks + shows sources

## Run it

```bash
cp .env.example .env
# put your OPENAI_API_KEY in .env

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `frontend/index.html` in browser, or serve it:
```bash
cd frontend && python -m http.server 8080
```

With Docker:
```bash
docker compose up --build
```

## API

- `GET /health` -> check if up
- `POST /upload` -> upload PDF (form-data, key=file)
- `POST /ask` -> {"question": "..."} -> {"answer": "...", "sources": [...]}

## Notes

- Chunk size 1000 chars with 200 overlap works well for most PDFs
- If answer says "I don't know", it means nothing relevant was found - that's on purpose to avoid hallucinations
