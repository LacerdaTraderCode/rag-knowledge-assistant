# RAG Knowledge Assistant

A small, complete retrieval-augmented generation pipeline: ingest text, retrieve the passages relevant to a question with TF-IDF, and ask Claude to answer using only what was retrieved — with every answer citing the sources it came from.

## Why TF-IDF instead of a dense embedding model

This is a deliberate choice, not a shortcut. TF-IDF needs no model download and no API key, so the entire pipeline — ingestion, retrieval, and its tests — runs offline and deterministically. The vector store's public interface (`add_documents`, `search`) says nothing about TF-IDF specifically, so swapping in a dense embedding model later is a change to one file, not a redesign.

## How a question gets answered

1. `POST /documents` splits the text into overlapping word chunks and indexes them.
2. `POST /query` ranks the indexed chunks against the question with cosine similarity over TF-IDF vectors.
3. The top chunks are placed in a prompt that instructs the model to answer only from what it was given and to cite each source it uses.
4. The response returns the answer alongside the exact chunks it was grounded in, so an answer can always be checked against its sources.

The app ships with a small sample knowledge base in `docs_sample/` (original short notes on REST APIs, database indexing, the testing pyramid, and caching) so it is queryable immediately, in addition to whatever is ingested through `/documents`.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

```bash
curl -X POST localhost:8000/documents -d '{"document_id": "notes", "text": "..."}'
curl -X POST localhost:8000/query -d '{"question": "..."}'
```

Interactive docs are served at `/docs`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers chunking (including that no word from the source text is lost), retrieval ranking, prompt construction, and the API end to end — all with the Claude call mocked, so no API key or network access is needed to run it.

## License

MIT — see [LICENSE](LICENSE).
