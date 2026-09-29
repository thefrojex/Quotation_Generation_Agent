# Quotation Generation Agent

A small FastAPI + React tool that turns an uploaded price-sheet PDF and a
pasted customer requirement email into a structured manufacturing quotation,
using an LLM to read the price sheet and draft the quote.

## What it actually does

1. **Upload price sheets.** You upload one or more PDF price lists. Each gets
   parsed one of two ways (chosen per upload):
   - `local_pdf_text` — plain text extraction via `pypdf`. Fast, free, but
     produces nothing useful for scanned/image-only PDFs (no OCR).
   - `llm_extract_51` — the extracted text is sent to GPT-5.1 with a prompt
     that asks it to return structured JSON (`products`, `quantity_breaks`,
     `moq`, `lead_time`, etc.). Falls back to the raw local extraction if the
     LLM call or JSON parsing fails.
2. **Manage active sheets.** Each uploaded sheet can be toggled active/inactive
   and deleted; a manifest (`storage/price_sheets/manifest.json`) tracks
   what's currently in play.
3. **Edit the agent's rules.** A single editable `agent.md` file (served/updated
   via `GET`/`PUT /agent/rules`) is injected as the system prompt for every
   quote — e.g. "use only the uploaded active price sheets," "do not invent
   prices," "cite source file names and pages."
4. **Generate a quote.** You paste the customer's requirement text (and
   optionally attach design files). The backend concatenates the full text of
   every *active* price sheet plus the customer text plus the rules file into
   one prompt (no retrieval/chunking — this is a full-context approach, and
   the code explicitly rejects the request if that combined context exceeds
   700k characters) and sends it to GPT-5.1 asking for a fixed JSON shape:
   `quotation` (title/currency/total/notes), `line_items`, `missing_required_fields`,
   `assumptions`, and `clarification_questions`. If the model doesn't return
   valid JSON, it retries once with an explicit "return JSON only" follow-up
   prompt before falling back to a placeholder draft response.
5. **Every generated quote is persisted** to `storage/quotes/<uuid>.json`,
   including the raw model output — useful for debugging prompt/response
   quality over time.

## Architecture

```
Frontend (React + Vite)  →  FastAPI backend  →  OpenAI (GPT-5.1)
     Quatation_gen/            Backend.py          Responses API
                                    │
                                    ▼
                        storage/ (local filesystem)
                        ├── price_sheets/<id>/source.pdf + parsed.json
                        ├── quotes/<id>.json
                        └── config/agent.md
```

There's no database — the backend uses the local filesystem (`storage/`) as
its persistence layer, tracked via a JSON manifest. No vector store, no
retrieval step: every active price sheet's full extracted text goes into the
prompt every time.

## Tech stack

- **Backend:** FastAPI, `pypdf` (text extraction), OpenAI Python SDK
  (`client.responses.create`, GPT-5.1), Python's stdlib `json`/`pathlib` for
  the filesystem-backed storage.
- **Frontend:** React + Vite, plain `axios` calls to a hardcoded
  `http://127.0.0.1:8000` API base (single-user/local-dev tool, not
  configured for a deployed API URL).

## Running it

```bash
# Backend
cd Backend
pip install -r requirement.txt
export OPENAI_API_KEY=sk-...
uvicorn Backend:app --reload   # http://127.0.0.1:8000

# Frontend
cd Frontend/Quatation_gen
npm install
npm run dev                    # http://localhost:5173
```

Environment variables the backend reads: `OPENAI_API_KEY` (required),
`OPENAI_MODEL` / `OPENAI_EXTRACTION_MODEL` (default to `gpt-5.1`),
`CORS_ORIGINS` (defaults to `*`).

## Example flow

1. Upload `price-list.pdf` with extraction mode `llm_extract_51`.
2. Paste a customer email like:
   > "Need 500 units of the M8 hex bolt, zinc-plated, delivered to Pune by
   > end of month."
3. `POST /quote/generate` returns something like:
   ```json
   {
     "result": {
       "quotation": {
         "title": "Preliminary Quotation",
         "currency": "INR",
         "total": "...",
         "notes": ["..."]
       },
       "line_items": [
         {"item": "M8 Hex Bolt (Zinc)", "quantity": "500", "unit_price": "...", "source": "price-list.pdf p.4"}
       ],
       "missing_required_fields": [],
       "assumptions": ["..."],
       "clarification_questions": []
     }
   }
   ```
   If the price sheet doesn't clearly cover the request, the agent is
   instructed to say so explicitly via `missing_required_fields` and
   `clarification_questions` rather than guess a price.

## Known limitations

- **No RAG/chunking** — every active price sheet's full text is stuffed into
  the prompt on every request. Works for a handful of price sheets; doesn't
  scale to a large catalog without hitting the 700k-character cutoff or
  becoming slow/expensive.
- **Scanned/image PDFs aren't OCR'd** — `local_pdf_text` mode returns empty
  text for image-only pages; only the LLM-extraction mode has any chance of
  reading them, and only via text extraction, not vision.
- **No auth** — the API has no authentication; it's built for local/single-user
  use, not multi-user deployment as-is.
- **`PdfParsing.py` is present but unused** — a vision-based image/PDF
  analysis helper that isn't imported or called anywhere in `Backend.py`.
- **Filesystem storage, no database** — fine for local use; would need a real
  datastore for concurrent/multi-user use.
