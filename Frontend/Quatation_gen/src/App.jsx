import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [view, setView] = useState("setup");
  const [priceSheets, setPriceSheets] = useState([]);
  const [rules, setRules] = useState("");
  const [uploadFiles, setUploadFiles] = useState([]);
  const [extractionMode, setExtractionMode] = useState("local_pdf_text");
  const [mailText, setMailText] = useState("");
  const [designFiles, setDesignFiles] = useState([]);
  const [selectedSheetIds, setSelectedSheetIds] = useState([]);
  const [quoteResponse, setQuoteResponse] = useState(null);
  const [isBusy, setIsBusy] = useState(false);
  const [message, setMessage] = useState("");

  const activeSheets = useMemo(
    () => priceSheets.filter((sheet) => sheet.active),
    [priceSheets]
  );

  const fetchSheets = async () => {
    const res = await axios.get(`${API_BASE}/agent/price-sheets`);
    setPriceSheets(res.data.price_sheets || []);
  };

  const fetchRules = async () => {
    const res = await axios.get(`${API_BASE}/agent/rules`);
    setRules(res.data.content || "");
  };

  useEffect(() => {
    const load = async () => {
      try {
        await Promise.all([fetchSheets(), fetchRules()]);
      } catch (error) {
        console.error(error);
        setMessage("Failed to load data from backend.");
      }
    };
    load();
  }, []);

  const onUploadPriceSheets = async () => {
    if (!uploadFiles.length) {
      setMessage("Please select at least one PDF.");
      return;
    }

    const formData = new FormData();
    uploadFiles.forEach((file) => formData.append("files", file));
    formData.append("extraction_mode", extractionMode);

    try {
      setIsBusy(true);
      await axios.post(`${API_BASE}/agent/price-sheets/upload`, formData);
      setUploadFiles([]);
      setMessage("Price sheets uploaded and parsed.");
      await fetchSheets();
    } catch (error) {
      console.error(error);
      setMessage(error?.response?.data?.detail || "Upload failed.");
    } finally {
      setIsBusy(false);
    }
  };

  const onToggleActive = async (sheetId, active) => {
    const formData = new FormData();
    formData.append("active", String(active));
    try {
      await axios.patch(`${API_BASE}/agent/price-sheets/${sheetId}/active`, formData);
      await fetchSheets();
    } catch (error) {
      console.error(error);
      setMessage("Failed to update price sheet status.");
    }
  };

  const onDeleteSheet = async (sheetId, filename) => {
    const confirmed = window.confirm(`Delete price sheet "${filename}"?`);
    if (!confirmed) {
      return;
    }

    try {
      setIsBusy(true);
      await axios.delete(`${API_BASE}/agent/price-sheets/${sheetId}`);
      setSelectedSheetIds((prev) => prev.filter((id) => id !== sheetId));
      setMessage("Price sheet deleted.");
      await fetchSheets();
    } catch (error) {
      console.error(error);
      setMessage(error?.response?.data?.detail || "Failed to delete price sheet.");
    } finally {
      setIsBusy(false);
    }
  };

  const onSaveRules = async () => {
    const formData = new FormData();
    formData.append("content", rules);
    try {
      setIsBusy(true);
      await axios.put(`${API_BASE}/agent/rules`, formData);
      setMessage("Agent rules saved.");
    } catch (error) {
      console.error(error);
      setMessage("Failed to save agent rules.");
    } finally {
      setIsBusy(false);
    }
  };

  const onGenerateQuote = async () => {
    if (!mailText.trim()) {
      setMessage("Please paste customer requirement email text.");
      return;
    }
    if (!activeSheets.length && !selectedSheetIds.length) {
      setMessage("Upload and activate at least one price sheet.");
      return;
    }

    const formData = new FormData();
    formData.append("customer_email_text", mailText);
    formData.append("selected_price_sheet_ids", JSON.stringify(selectedSheetIds));
    designFiles.forEach((file) => formData.append("design_files", file));

    try {
      setIsBusy(true);
      const res = await axios.post(`${API_BASE}/quote/generate`, formData);
      setQuoteResponse(res.data);
      setMessage("Quotation generated.");
    } catch (error) {
      console.error(error);
      setQuoteResponse(null);
      setMessage(error?.response?.data?.detail || "Quote generation failed.");
    } finally {
      setIsBusy(false);
    }
  };

  const renderSetup = () => (
    <section className="panel">
      <h2>Agent Setup</h2>

      <div className="card">
        <h3>Upload Price Sheets (PDF)</h3>
        <label>Extraction Method</label>
        <select
          value={extractionMode}
          onChange={(e) => setExtractionMode(e.target.value)}
        >
          <option value="local_pdf_text">Fast local extraction (pypdf)</option>
          <option value="llm_extract_51">AI extraction with GPT-5.1</option>
        </select>
        <input
          type="file"
          accept="application/pdf"
          multiple
          onChange={(e) => setUploadFiles(Array.from(e.target.files || []))}
        />
        <button onClick={onUploadPriceSheets} disabled={isBusy}>
          Upload PDFs
        </button>
      </div>

      <div className="card">
        <h3>Price Sheet Library</h3>
        {!priceSheets.length ? (
          <p>No price sheets uploaded yet.</p>
        ) : (
          <ul className="sheet-list">
            {priceSheets.map((sheet) => (
              <li key={sheet.id}>
                <div>
                  <strong>{sheet.filename}</strong>
                  <small>{sheet.uploaded_at}</small>
                  <small>Mode: {sheet.extraction_mode || "local_pdf_text"}</small>
                  <small>Status: {sheet.extraction_status || "parsed"}</small>
                </div>
                <label>
                  <input
                    type="checkbox"
                    checked={Boolean(sheet.active)}
                    onChange={(e) => onToggleActive(sheet.id, e.target.checked)}
                  />
                  Active
                </label>
                <button
                  type="button"
                  className="danger"
                  onClick={() => onDeleteSheet(sheet.id, sheet.filename)}
                  disabled={isBusy}
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="card">
        <h3>agent.md Rules</h3>
        <textarea
          rows={14}
          value={rules}
          onChange={(e) => setRules(e.target.value)}
        />
        <button onClick={onSaveRules} disabled={isBusy}>
          Save Rules
        </button>
      </div>
    </section>
  );

  const renderQuote = () => {
    const result = quoteResponse?.result;
    return (
      <section className="panel quote-panel">
        <div className="input-side card">
          <h2>Create Quotation</h2>
          <label>Customer Requirement Email</label>
          <textarea
            rows={12}
            value={mailText}
            onChange={(e) => setMailText(e.target.value)}
            placeholder="Paste customer email and requirement details here"
          />

          <label>Optional Design Files</label>
          <input
            type="file"
            multiple
            onChange={(e) => setDesignFiles(Array.from(e.target.files || []))}
          />

          <label>Pick most relevant price sheets (optional)</label>
          <div className="sheet-picker">
            {priceSheets.map((sheet) => (
              <label key={sheet.id}>
                <input
                  type="checkbox"
                  checked={selectedSheetIds.includes(sheet.id)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedSheetIds((prev) => [...prev, sheet.id]);
                    } else {
                      setSelectedSheetIds((prev) => prev.filter((id) => id !== sheet.id));
                    }
                  }}
                />
                {sheet.filename}
              </label>
            ))}
          </div>

          <button onClick={onGenerateQuote} disabled={isBusy}>
            Generate Quotation
          </button>
        </div>

        <div className="output-side card">
          <h2>Generated Quotation</h2>
          {!result ? (
            <p>Quotation output will appear here.</p>
          ) : (
            <>
              <h3>{result.quotation?.title || "Quotation"}</h3>
              <p><strong>Total:</strong> {result.quotation?.total}</p>
              <p><strong>Currency:</strong> {result.quotation?.currency}</p>

              <h4>Line Items</h4>
              <ul>
                {(result.line_items || []).map((item, idx) => (
                  <li key={`${item.item}-${idx}`}>
                    {item.item} | Qty: {item.quantity} | Unit: {item.unit_price} | Total: {item.line_total} | Source: {item.source}
                  </li>
                ))}
              </ul>

              <h4>Missing Required Fields</h4>
              <ul>
                {(result.missing_required_fields || []).map((field) => (
                  <li key={field}>{field}</li>
                ))}
              </ul>

              <h4>Assumptions</h4>
              <ul>
                {(result.assumptions || []).map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>

              <h4>Clarification Questions</h4>
              <ul>
                {(result.clarification_questions || []).map((q) => (
                  <li key={q}>{q}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      </section>
    );
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h1>Quotation Ops</h1>
        <button
          className={view === "setup" ? "active" : ""}
          onClick={() => setView("setup")}
        >
          Agent Setup
        </button>
        <button
          className={view === "quote" ? "active" : ""}
          onClick={() => setView("quote")}
        >
          Create Quotation
        </button>
      </aside>

      <main className="content">
        {message && <p className="message">{message}</p>}
        {view === "setup" ? renderSetup() : renderQuote()}
      </main>
    </div>
  );
}

export default App;
