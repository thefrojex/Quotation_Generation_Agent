import json
import math
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
PRICE_SHEETS_DIR = STORAGE_DIR / "price_sheets"
DESIGN_INPUTS_DIR = STORAGE_DIR / "design_inputs"
QUOTES_DIR = STORAGE_DIR / "quotes"
CONFIG_DIR = STORAGE_DIR / "config"
MANIFEST_PATH = PRICE_SHEETS_DIR / "manifest.json"
AGENT_RULES_PATH = CONFIG_DIR / "agent.md"

MAX_CONTEXT_CHARACTERS = 700_000
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.1")
EXTRACTION_MODEL_NAME = os.getenv("OPENAI_EXTRACTION_MODEL", "gpt-5.1")
LLM_EXTRACTION_TEXT_LIMIT = 300_000

DEFAULT_AGENT_RULES = """# Quotation Agent Rules

You are a manufacturing quotation assistant.

Core instructions:
1. Use only the uploaded active price sheets and customer input.
2. Do not invent prices, terms, or specs.
3. Flag missing or ambiguous requirements.
4. Return clear assumptions if a quote is still possible.
5. Cite source file names and pages for each line item.
"""


class QuoteSchema(BaseModel):
    customer_email_text: str
    selected_price_sheet_ids: list[str] = []


def ensure_storage() -> None:
    for path in [STORAGE_DIR, PRICE_SHEETS_DIR, DESIGN_INPUTS_DIR, QUOTES_DIR, CONFIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        MANIFEST_PATH.write_text(json.dumps({"price_sheets": []}, indent=2), encoding="utf-8")

    if not AGENT_RULES_PATH.exists():
        AGENT_RULES_PATH.write_text(DEFAULT_AGENT_RULES, encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def load_manifest() -> dict[str, Any]:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="Price sheet manifest is corrupted") from exc


def save_manifest(data: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def extract_pdf_text(file_path: Path) -> tuple[str, list[dict[str, Any]]]:
    if PdfReader is None:
        raise HTTPException(status_code=500, detail="pypdf is not installed in backend environment")

    reader = PdfReader(str(file_path))
    pages: list[dict[str, Any]] = []
    full_text_parts: list[str] = []

    for idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append({"page": idx, "text": page_text})
        full_text_parts.append(f"\n\n[Page {idx}]\n{page_text}")

    return "".join(full_text_parts).strip(), pages


def extract_structured_with_llm(filename: str, raw_text: str, pages: list[dict[str, Any]]) -> dict[str, Any]:
    client = get_openai_client()
    truncated = raw_text[:LLM_EXTRACTION_TEXT_LIMIT]
    was_truncated = len(raw_text) > LLM_EXTRACTION_TEXT_LIMIT

    system_prompt = (
        "You extract structured price sheet data for manufacturing quotations. "
        "Return only valid JSON. Do not add markdown fences."
    )
    user_prompt = (
        "Extract pricing-relevant fields from this price sheet text.\n"
        "JSON shape:\n"
        "{\n"
        "  \"document_summary\": string,\n"
        "  \"products\": [\n"
        "    {\n"
        "      \"item\": string,\n"
        "      \"material\": string,\n"
        "      \"specifications\": [string],\n"
        "      \"quantity_breaks\": [{\"quantity\": string, \"unit_price\": string}],\n"
        "      \"moq\": string,\n"
        "      \"lead_time\": string,\n"
        "      \"notes\": [string],\n"
        "      \"source_page\": string\n"
        "    }\n"
        "  ],\n"
        "  \"global_terms\": [string],\n"
        "  \"missing_or_unclear_sections\": [string]\n"
        "}\n\n"
        f"Filename: {filename}\n"
        f"Page count: {len(pages)}\n"
        f"Text may be truncated for token safety: {str(was_truncated).lower()}\n\n"
        "Price sheet text:\n"
        f"{truncated}"
    )

    completion = client.responses.create(
        model=EXTRACTION_MODEL_NAME,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    data = parse_json_response(completion.output_text)

    data["llm_input_truncated"] = was_truncated
    return data


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def clean_json_response(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_json_response(raw_text: str) -> dict[str, Any]:
    cleaned = clean_json_response(raw_text)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("Model returned non-object JSON")
    return data


def get_openai_client() -> Any:
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="openai package is not installed in backend environment")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing")
    return OpenAI(api_key=api_key)


def build_system_prompt(agent_rules: str) -> str:
    return (
        f"{agent_rules}\n\n"
        "Return ONLY valid JSON with this exact top-level shape:\n"
        "{\n"
        "  \"quotation\": {\"title\": string, \"currency\": string, \"total\": string, \"notes\": [string]},\n"
        "  \"line_items\": [{\"item\": string, \"quantity\": string, \"unit_price\": string, \"line_total\": string, \"source\": string}],\n"
        "  \"missing_required_fields\": [string],\n"
        "  \"assumptions\": [string],\n"
        "  \"clarification_questions\": [string]\n"
        "}\n"
        "If information is missing, keep quotation conservative and add explicit clarification_questions."
    )


def build_json_retry_prompt(previous_output: str) -> str:
    return (
        "Your previous response was not raw JSON. Return the same answer again as a single valid JSON object only. "
        "Do not use markdown fences, explanation text, or surrounding quotes.\n\n"
        "Previous response:\n"
        f"{previous_output}"
    )


def build_user_context(
    email_text: str,
    design_docs: list[dict[str, Any]],
    price_sheets: list[dict[str, Any]],
) -> str:
    design_context = []
    for item in design_docs:
        design_context.append(f"Design File: {item['filename']}\n{item['text']}")

    sheet_context = []
    for sheet in price_sheets:
        structured_section = ""
        if sheet.get("structured_prices"):
            structured_section = (
                "\nStructured Extraction:\n"
                f"{json.dumps(sheet['structured_prices'], ensure_ascii=True)}"
            )
        sheet_context.append(
            f"Price Sheet: {sheet['filename']}\n"
            f"Uploaded: {sheet['uploaded_at']}\n"
            f"Extraction Mode: {sheet.get('extraction_mode', 'local_pdf_text')}\n"
            f"Content:\n{sheet['text']}"
            f"{structured_section}"
        )

    return (
        "Customer Email:\n"
        f"{email_text}\n\n"
        "Design Inputs:\n"
        f"{'\n\n'.join(design_context) if design_context else 'None'}\n\n"
        "Active Price Sheets:\n"
        f"{'\n\n'.join(sheet_context)}"
    )


ensure_storage()

app = FastAPI(title="Quotation Generator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "time": utc_now_iso()}


@app.get("/agent/price-sheets")
def list_price_sheets() -> dict[str, Any]:
    return load_manifest()


@app.post("/agent/price-sheets/upload")
async def upload_price_sheets(
    files: list[UploadFile] = File(...),
    extraction_mode: str = Form("local_pdf_text"),
) -> dict[str, Any]:
    if extraction_mode not in {"local_pdf_text", "llm_extract_51"}:
        raise HTTPException(status_code=400, detail="Invalid extraction_mode")

    manifest = load_manifest()
    uploaded: list[dict[str, Any]] = []

    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        sheet_id = str(uuid.uuid4())
        sheet_dir = PRICE_SHEETS_DIR / sheet_id
        sheet_dir.mkdir(parents=True, exist_ok=True)

        file_path = sheet_dir / "source.pdf"
        content = await upload.read()
        file_path.write_bytes(content)

        extracted_text, pages = extract_pdf_text(file_path)
        structured_prices: dict[str, Any] | None = None
        extraction_status = "parsed"
        warnings: list[str] = []

        if extraction_mode == "llm_extract_51":
            try:
                structured_prices = extract_structured_with_llm(upload.filename, extracted_text, pages)
            except Exception as exc:
                extraction_status = "fallback_to_local"
                warnings.append(f"LLM extraction failed: {str(exc)}")

        (sheet_dir / "parsed.json").write_text(
            json.dumps(
                {
                    "id": sheet_id,
                    "filename": upload.filename,
                    "uploaded_at": utc_now_iso(),
                    "text": extracted_text,
                    "pages": pages,
                    "extraction_mode": extraction_mode,
                    "extraction_status": extraction_status,
                    "structured_prices": structured_prices,
                    "warnings": warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        row = {
            "id": sheet_id,
            "filename": upload.filename,
            "uploaded_at": utc_now_iso(),
            "active": True,
            "extraction_mode": extraction_mode,
            "extraction_status": extraction_status,
            "warnings": warnings,
            "path": str(file_path.relative_to(BASE_DIR)),
            "parsed_path": str((sheet_dir / "parsed.json").relative_to(BASE_DIR)),
        }
        manifest["price_sheets"].append(row)
        uploaded.append(row)

    save_manifest(manifest)
    return {"uploaded": uploaded, "count": len(uploaded)}


@app.patch("/agent/price-sheets/{sheet_id}/active")
def set_sheet_active(sheet_id: str, active: bool = Form(...)) -> dict[str, Any]:
    manifest = load_manifest()
    found = False
    for sheet in manifest["price_sheets"]:
        if sheet["id"] == sheet_id:
            sheet["active"] = active
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="Price sheet not found")

    save_manifest(manifest)
    return {"ok": True, "sheet_id": sheet_id, "active": active}


@app.delete("/agent/price-sheets/{sheet_id}")
def delete_price_sheet(sheet_id: str) -> dict[str, Any]:
    manifest = load_manifest()
    rows = manifest.get("price_sheets", [])
    target: dict[str, Any] | None = None

    for sheet in rows:
        if sheet.get("id") == sheet_id:
            target = sheet
            break

    if target is None:
        raise HTTPException(status_code=404, detail="Price sheet not found")

    warnings: list[str] = []
    sheet_dir = PRICE_SHEETS_DIR / sheet_id
    if sheet_dir.exists() and sheet_dir.is_dir():
        shutil.rmtree(sheet_dir)
    else:
        warnings.append("Price sheet folder already missing")

    manifest["price_sheets"] = [sheet for sheet in rows if sheet.get("id") != sheet_id]
    save_manifest(manifest)

    return {
        "ok": True,
        "deleted": {
            "id": target.get("id"),
            "filename": target.get("filename"),
        },
        "warnings": warnings,
    }


@app.get("/agent/rules")
def get_agent_rules() -> dict[str, str]:
    return {"content": AGENT_RULES_PATH.read_text(encoding="utf-8")}


@app.put("/agent/rules")
def update_agent_rules(content: str = Form(...)) -> dict[str, str]:
    AGENT_RULES_PATH.write_text(content.strip() + "\n", encoding="utf-8")
    return {"status": "saved"}


@app.post("/quote/generate")
async def generate_quote(
    customer_email_text: str = Form(...),
    selected_price_sheet_ids: str = Form("[]"),
    design_files: list[UploadFile] = File(default=[]),
) -> dict[str, Any]:
    try:
        selected_ids = json.loads(selected_price_sheet_ids)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="selected_price_sheet_ids must be JSON list") from exc

    if not isinstance(selected_ids, list):
        raise HTTPException(status_code=400, detail="selected_price_sheet_ids must be list")

    manifest = load_manifest()
    sheets_by_id = {sheet["id"]: sheet for sheet in manifest["price_sheets"]}

    active_sheets = [sheet for sheet in manifest["price_sheets"] if sheet.get("active")]
    if selected_ids:
        active_sheets = [sheets_by_id[sid] for sid in selected_ids if sid in sheets_by_id]

    if not active_sheets:
        raise HTTPException(status_code=400, detail="No active or selected price sheets available")

    loaded_sheets: list[dict[str, Any]] = []
    for sheet in active_sheets:
        parsed_path = BASE_DIR / sheet["parsed_path"]
        if not parsed_path.exists():
            continue
        parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        loaded_sheets.append(
            {
                "id": parsed["id"],
                "filename": parsed["filename"],
                "uploaded_at": parsed["uploaded_at"],
                "text": parsed["text"],
                "structured_prices": parsed.get("structured_prices"),
                "extraction_mode": parsed.get("extraction_mode", "local_pdf_text"),
            }
        )

    if not loaded_sheets:
        raise HTTPException(status_code=400, detail="Selected price sheets have no parsed content")

    design_docs: list[dict[str, Any]] = []
    for design_file in design_files:
        if not design_file.filename:
            continue
        if design_file.filename.lower().endswith(".pdf"):
            temp_id = str(uuid.uuid4())
            temp_path = DESIGN_INPUTS_DIR / f"{temp_id}.pdf"
            temp_path.write_bytes(await design_file.read())
            text, _ = extract_pdf_text(temp_path)
            design_docs.append({"filename": design_file.filename, "text": text})
        else:
            blob = await design_file.read()
            design_docs.append(
                {
                    "filename": design_file.filename,
                    "text": blob.decode("utf-8", errors="ignore")[:100_000],
                }
            )

    agent_rules = AGENT_RULES_PATH.read_text(encoding="utf-8")
    system_prompt = build_system_prompt(agent_rules)
    full_context = build_user_context(customer_email_text, design_docs, loaded_sheets)
    est_tokens = estimate_tokens(system_prompt + full_context)

    if len(full_context) > MAX_CONTEXT_CHARACTERS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Context too large for no-RAG mode. Reduce selected price sheets or split customer requirements."
            ),
        )

    client = get_openai_client()
    completion = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_context},
        ],
    )
    response_text = completion.output_text
    retry_response_text = ""

    try:
        parsed_output = parse_json_response(response_text)
    except (json.JSONDecodeError, ValueError):
        retry_completion = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_context},
                {"role": "user", "content": build_json_retry_prompt(response_text)},
            ],
        )
        retry_response_text = retry_completion.output_text

        try:
            parsed_output = parse_json_response(retry_response_text)
        except (json.JSONDecodeError, ValueError):
            debug_output = clean_json_response(retry_response_text or response_text)
            parsed_output = {
                "quotation": {
                    "title": "Draft quotation",
                    "currency": "Unknown",
                    "total": "Model returned non-JSON",
                    "notes": [debug_output],
                },
                "line_items": [],
                "missing_required_fields": ["Model output format invalid"],
                "assumptions": [],
                "clarification_questions": [],
            }

    quote_raw_output = clean_json_response(retry_response_text or response_text)

    quote_id = str(uuid.uuid4())
    quote_record = {
        "id": quote_id,
        "created_at": utc_now_iso(),
        "model": MODEL_NAME,
        "estimated_input_tokens": est_tokens,
        "used_price_sheet_ids": [sheet["id"] for sheet in loaded_sheets],
        "customer_email_text": customer_email_text,
        "raw_model_output": quote_raw_output,
        "result": parsed_output,
    }
    (QUOTES_DIR / f"{quote_id}.json").write_text(json.dumps(quote_record, indent=2), encoding="utf-8")

    return {
        "quote_id": quote_id,
        "estimated_input_tokens": est_tokens,
        "used_price_sheets": [{"id": s["id"], "filename": s["filename"]} for s in loaded_sheets],
        "result": parsed_output,
    }
