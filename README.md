# RAG Knowledge Assistant

A small, complete retrieval-augmented generation pipeline: ingest text, retrieve the passages relevant to a question with TF-IDF, embeddings, or both, and ask Claude to answer using only what was retrieved — with every answer citing the sources it came from.

## Retrieval modes

The mode is chosen with `RETRIEVAL_MODE` (default `hybrid`, documented in `.env.example`):

| Mode | How chunks are ranked | Needs |
| --- | --- | --- |
| `tfidf` | Cosine similarity over TF-IDF vectors | Nothing: no model download, runs fully offline |
| `semantic` | Cosine similarity between embeddings, searched in Qdrant | An embedding model (about 220 MB, downloaded on first start) |
| `hybrid` | The TF-IDF and semantic rankings merged with Reciprocal Rank Fusion | The same as `semantic` |

All three implement the same `add_documents` / `search` interface, so the pipeline and the API do not change with the mode. TF-IDF stays a first-class option: it is the baseline the other modes are measured against, and the one to pick when nothing should be downloaded.

- **Embeddings** come from [fastembed](https://github.com/qdrant/fastembed) running `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` as an ONNX model, so the install has no PyTorch. The model sits behind a small `Embedder` interface that the store never looks behind.
- **Vector database** is Qdrant. By default it runs in memory inside the process, so there is nothing to start. For a real server, run `docker compose up -d` and set `QDRANT_URL=http://localhost:6333`. The collection is rebuilt every time the app starts, so the semantic index always matches the lexical one.
- **Reciprocal Rank Fusion** scores a chunk as the sum of `1 / (60 + rank)` over the rankings it appears in. Only ranks are used, so the two scorers never need their scores normalized against each other.
- **`score`** in a response belongs to the mode that produced it (a TF-IDF cosine, an embedding cosine, or a fused rank score), so it is only comparable within one response.

Semantic and hybrid retrieval always return the closest chunks, even for a question the corpus cannot answer. The prompt is what tells the model to say when the sources do not contain the answer.

## Evaluation

Each mode is measured on a golden set of 40 questions over the 18 sample notes (`evaluation/golden_set.json`). Every question is labeled with the note, or notes, that answer it. A question is a hit when one of the top 3 retrieved chunks comes from a labeled note, and MRR is the mean of 1 / the position of the first such chunk (0 if none is in the top 10).

The questions come in three kinds: 14 that reuse many of the words of the answering note, 14 paraphrases that share few of them, and 12 exact identifiers or terms such as `B-tree` or `(user_id, created_at)`. A test checks that each kind keeps that definition and that no question repeats a phrase of the notes. The questions were written once, before any mode was run against them, and have not been edited since.

Results with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, as printed by the `retrieval-evaluation` CI job:

| Mode | Hit rate@3 | MRR |
| --- | --- | --- |
| tfidf | 0.875 (35/40) | 0.852 |
| semantic | 0.900 (36/40) | 0.877 |
| hybrid | 0.975 (39/40) | 0.941 |

Hit rate@3 by question type:

| Mode | Lexical overlap (14) | Paraphrase (14) | Exact term (12) |
| --- | --- | --- | --- |
| tfidf | 1.000 (14/14) | 0.643 (9/14) | 1.000 (12/12) |
| semantic | 1.000 (14/14) | 0.857 (12/14) | 0.833 (10/12) |
| hybrid | 1.000 (14/14) | 0.929 (13/14) | 1.000 (12/12) |

MRR by question type:

| Mode | Lexical overlap (14) | Paraphrase (14) | Exact term (12) |
| --- | --- | --- | --- |
| tfidf | 1.000 | 0.577 | 1.000 |
| semantic | 0.964 | 0.818 | 0.845 |
| hybrid | 1.000 | 0.830 | 1.000 |

What the numbers say, including where the new modes do not win:

- Hybrid has the highest hit rate overall and is never below TF-IDF on any question type. It matches TF-IDF on word-overlap and exact-term questions, which TF-IDF answers perfectly, and gains on paraphrases (13/14 against 9/14).
- Semantic retrieval on its own does not beat TF-IDF everywhere. It is better on paraphrases (12/14 against 9/14), equal on word-overlap questions, and worse on exact terms (10/12 against 12/12): it missed `B-tree` and `(user_id, created_at)`. Its MRR is also a little lower on word-overlap questions (0.964 against 1.000).
- Fusion is not free. Hybrid missed one question that semantic retrieval alone answered, `paraphrase-05` (a service receiving the same work item twice): TF-IDF ranked two chunks of the OAuth note first, and fusion let them push the right note out of the top 3.
- Hybrid answered two paraphrases (`paraphrase-03`, `paraphrase-04`) that neither TF-IDF nor semantic retrieval had in its own top 3, so combining the two rankings pulled the right note up.

Forty questions over 18 short notes is a small sample. One question is 2.5 points of hit rate, so the one-question gap between semantic and TF-IDF overall says little by itself. The gains on paraphrases and the weakness of semantic retrieval on exact terms are larger. The notes and the questions share an author, which favors questions the notes answer cleanly; a real corpus is messier.

To reproduce it:

```bash
pip install -r requirements.txt
python -m evaluation.run
```

The first run downloads the embedding model. The command prints the tables above and logs every question each mode missed in its top 3. With `--enforce-gate` it exits with status 1 when hybrid's hit rate@3 is below the TF-IDF baseline, which is what CI does on every pull request, with the model cached between runs. The dependency versions are pinned, but a different model or library version can move a result by a question.

## How a question gets answered

1. `POST /documents` splits the text into overlapping word chunks and indexes them.
2. `POST /query` ranks the indexed chunks against the question using the configured retrieval mode.
3. The top chunks are placed in a prompt that instructs the model to answer only from what it was given and to cite each source it uses.
4. The response returns the answer alongside the exact chunks it was grounded in, so an answer can always be checked against its sources.

The app ships with a small sample knowledge base in `docs_sample/` (18 original short notes, from REST APIs and database transactions to sourdough baking and Roman aqueducts) so it is queryable immediately, in addition to whatever is ingested through `/documents`. The same notes are the corpus of the retrieval evaluation.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set ANTHROPIC_API_KEY, and RETRIEVAL_MODE if you want another mode
uvicorn app.main:app --reload --env-file .env
```

With the default `hybrid` mode, the first start downloads the embedding model (about 220 MB) and indexes the sample notes. Set `RETRIEVAL_MODE=tfidf` to skip the download entirely.

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

The suite covers chunking (including that no word from the source text is lost), the three stores, the fusion arithmetic, mode selection, lazy startup, the API contract in every mode, prompt construction, the evaluation metrics and the integrity of the golden set. It runs offline: the Claude call is mocked and a deterministic stand-in replaces the embedding model, so no API key, model download or network access is needed. The real model is only exercised by the evaluation.

## License

MIT — see [LICENSE](LICENSE).
