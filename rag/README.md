# RAG Few-Shot Retrieval

Supplies the DSPy evaluator with examples of prompts that previously worked, so its
`revised_prompt` output is guided by what has actually scored well before.

## Files

- `document_processor.py` — the whole module: builds the FAISS index and retrieves from it
- `main.py` — CLI to eyeball what the app would retrieve for a given prompt
- `vectorstore/` — FAISS index, gitignored and rebuilt on demand

## What gets indexed

Rows from `app/imagesdata/images_log.csv` where **both** hold:

- `overall_score > 8`
- `original_user_prompt != prompt` — the loop actually rewrote it

That second filter is the point: an unrevised prompt that scored well says nothing about
how to revise. As of the committed log this yields 34 unique prompts.

## Usage

The app calls this automatically. To inspect it directly:

```bash
cd rag
python main.py
```

## Integration

`app/services/dspy_optimization.py` retrieves once per request, keyed on the user's raw
input, and passes the result to every iteration via the `few_shot_examples` field of
`ImageEvaluationSignature`.

**Retrieval is best-effort by design.** Missing `UPSTAGE_API_KEY`, an unbuilt index, or an
empty corpus all degrade to no examples and log a line. RAG never fails a generation.

## Index lifecycle

Built once per process from the log as it stands at first use, then reused. Rows written
during a run are indexed on the next restart. `process_document(csv, rebuild=True)` forces
a rebuild, as does deleting `vectorstore/`.

## Notes

- No text splitter — these prompts average ~360 characters, so chunking at 1000 was a no-op.
- `langchain-community` (used for the FAISS wrapper) is being sunset upstream. Pinned and
  working, but it will need migrating eventually.
