# -*- coding: utf-8 -*-
"""
kyc_agent.py  –  Agentic KYC Workflow using LangChain ReAct + Google Gemini
=============================================================================
Architecture:
    User Input (customer_id + document)
        ↓
    [KYC ReAct Agent  —  Gemini LLM as the "brain"]
        ↓ Tool 1: identity_check_tool
        ↓ Tool 2: news_check_tool          (skipped if identity critically fails)
        ↓ Tool 3: risk_scoring_tool
        ↓
    Final JSON report  +  LLM natural-language explanation

What makes this truly "Agentic" vs the old pipeline:
  • The LLM decides which tools to call and in what order (Thought → Action → Observation loop)
  • It can short-circuit (e.g. REJECT immediately on critical identity failure)
  • Every reasoning step is visible/logged (verbose=True)
  • The final answer is an LLM-written explanation, not just a number

Install:
    pip install langchain langchain-google-genai google-generativeai pydantic python-dotenv
    (existing deps: pandas, pytesseract/Pillow optional, pypdf optional)

Usage:
    python kyc_agent.py --customer-id C001 --document data/id_C001.txt
    python kyc_agent.py --customer-id C003 --document data/id_C003_mismatch.txt --entity-name SingPost --reports-dir data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# ── Tesseract path (conda env) ─────────────────────────────────────────────
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/Users/luocx/miniconda3/envs/project6/bin/tesseract"

# ── LangChain ──────────────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

# ── RAG dependencies ───────────────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ── Your existing modules (unchanged) ─────────────────────────────────────
from identity_check import load_customers, run_identity_check
from news_check import load_adverse_media_db, run_news_check
from risk_scoring import run_risk_scoring

# =============================================================================
# 0.  Configuration & Setup
# =============================================================================

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file!  "
                     "Get one free at https://aistudio.google.com")

# Gemini 1.5 Flash — free tier, fast, good for agentic tasks
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
    temperature=0,          # deterministic for compliance tasks
    convert_system_message_to_human=True,  # Gemini quirk
)


# =============================================================================
# 0b.  Auto-detection helpers
# =============================================================================

def _auto_detect_entity_name(report_files: list[str]) -> str:
    """Infer entity name from PDF filenames.
    e.g. ['SingPost AR 2022_23.pdf', 'SingPost AR 2023_24.pdf'] → 'SingPost'
    """
    candidates = []
    for f in report_files:
        stem = Path(f).stem  # e.g. "SingPost AR 2022_23"
        match = re.match(r'^(.+?)\s+(?:AR|Annual|Report)\b', stem, re.IGNORECASE)
        if match:
            candidates.append(match.group(1).strip())
        else:
            candidates.append(re.split(r'[\s_]', stem)[0])
    if not candidates:
        return ""
    return Counter(candidates).most_common(1)[0][0]


# =============================================================================
# 1.  Global state shared between tools
#     (Agent tools are plain functions; they read from this context dict)
# =============================================================================

_KYC_CONTEXT: dict = {}   # populated by parse_args() before agent runs

# Cache directory for persisted FAISS vector stores
_VECTOR_STORE_DIR = Path("output/vector_store_cache")


# =============================================================================
# 1b. RAG — Embedding detection + vector store build / cache
# =============================================================================

def _detect_embeddings(api_key: str):
    """
    Try Google Generative AI embeddings first.
    Fall back to local sentence-transformers if unavailable.
    Returns (embeddings_object, backend_name).
    """
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key,
        )
        # Quick probe to confirm the API key has embedding access
        embeddings.embed_query("test")
        print("[RAG] Using Google Generative AI embeddings (text-embedding-004)")
        return embeddings, "google"
    except Exception as e:
        print(f"[RAG] Google embeddings unavailable ({e.__class__.__name__}), "
              "falling back to local sentence-transformers (all-MiniLM-L6-v2)")
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings, "local"


def _compute_corpus_hash(report_files: list[str]) -> str:
    """SHA-256 of all PDF file sizes + mtimes — fast proxy for content change."""
    h = hashlib.sha256()
    for f in sorted(report_files):
        p = Path(f)
        if p.exists():
            stat = p.stat()
            h.update(f"{p.name}:{stat.st_size}:{stat.st_mtime}".encode())
    return h.hexdigest()[:16]


def build_or_load_vector_store(report_files: list[str], api_key: str):
    """
    Build a FAISS vector store from the given PDF files, or load from cache
    if the corpus hasn't changed since the last run.

    Cache layout:
        output/vector_store_cache/<hash>/index.faiss
        output/vector_store_cache/<hash>/index.pkl
        output/vector_store_cache/<hash>/meta.json   ← backend + file list
    """
    if not report_files:
        return None

    embeddings, backend = _detect_embeddings(api_key)
    corpus_hash = _compute_corpus_hash(report_files)
    cache_dir = _VECTOR_STORE_DIR / corpus_hash

    # ── Try loading from cache ────────────────────────────────────────────
    meta_path = cache_dir / "meta.json"
    if cache_dir.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("backend") == backend:
            print(f"[RAG] Loading vector store from cache ({cache_dir})")
            try:
                vs = FAISS.load_local(
                    str(cache_dir),
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(f"[RAG] Cache hit — {meta['num_chunks']} chunks loaded")
                return vs
            except Exception as e:
                print(f"[RAG] Cache load failed ({e}), rebuilding...")

    # ── Build from scratch ────────────────────────────────────────────────
    print(f"[RAG] Building vector store from {len(report_files)} PDF(s)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    all_docs = []
    for f in report_files:
        p = Path(f)
        if not p.exists():
            print(f"[RAG] Skipping missing file: {p.name}")
            continue
        try:
            loader = PyPDFLoader(str(p))
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)
            print(f"[RAG]   {p.name} → {len(chunks)} chunks")
        except Exception as e:
            print(f"[RAG]   Failed to load {p.name}: {e}")

    if not all_docs:
        print("[RAG] No documents loaded, skipping vector store.")
        return None

    vs = FAISS.from_documents(all_docs, embeddings)

    # ── Persist to cache ──────────────────────────────────────────────────
    cache_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(cache_dir))
    meta_path.write_text(json.dumps({
        "backend": backend,
        "corpus_hash": corpus_hash,
        "num_chunks": len(all_docs),
        "files": [Path(f).name for f in report_files],
    }, indent=2))
    print(f"[RAG] Vector store saved to cache ({len(all_docs)} chunks, backend={backend})")
    return vs


# =============================================================================
# 2.  Tools  —  each wraps one of your existing modules
# =============================================================================

@tool
def identity_check_tool(customer_id: str) -> str:
    """
    Verifies the customer's identity by extracting fields from their
    ID document (image or .txt) using OCR (pytesseract) and matching
    them against the trusted daily customer register (CSV).

    Returns a JSON string with:
      - confidence_score (0-100): how well the document matches the register
      - name_match, id_match, dob_match: individual field results
      - ocr_fields: what was extracted from the document
      - reasons: list of mismatches found
    """
    ctx = _KYC_CONTEXT
    customers_df = ctx["customers_df"]
    doc_path     = ctx["doc_path"]

    result = run_identity_check(
        customer_id=customer_id,
        customers_df=customers_df,
        doc_path=doc_path,
    )

    result_dict = {
        "customer_id":       result.customer_id,
        "confidence_score":  result.confidence_score,
        "name_match":        result.name_match,
        "id_match":          result.id_match,
        "dob_match":         result.dob_match,
        "ocr_fields":        result.ocr_fields,
        "reasons":           result.reasons,
    }
    _KYC_CONTEXT["_identity_result"] = result_dict
    return json.dumps(result_dict, indent=2)


@tool
def news_check_tool(customer_id: str) -> str:
    """
    Screens the customer against adverse media sources and, optionally,
    annual report PDF files for risk-related keywords (fraud, sanction,
    money laundering, etc.).

    Returns a JSON string with:
      - news_risk_score (0-100): severity of adverse media hits
      - matched_alerts: list of individual hits with source and severity
      - reasons: human-readable explanation of each finding
    """
    ctx = _KYC_CONTEXT
    customers_df  = ctx["customers_df"]
    adverse_db    = ctx["adverse_db"]
    entity_name   = ctx.get("entity_name", "")
    report_files  = ctx.get("report_files", [])
    vector_store  = ctx.get("vector_store")   # pre-built RAG index (may be None)

    row = customers_df.loc[customers_df["customer_id"] == customer_id]
    if row.empty:
        return json.dumps({"error": f"Unknown customer_id: {customer_id}"})
    customer_name = str(row.iloc[0]["name"])

    # First pass: check adverse media only (no annual report screening)
    result = run_news_check(
        customer_id=customer_id,
        customer_name=customer_name,
        adverse_media_db=adverse_db,
        entity_name=None,
        report_files=[],
        vector_store=None,
    )

    # Only run annual report screening if the customer already has adverse media hits.
    # This prevents entity report keywords from polluting unrelated customers.
    if result.matched_alerts and entity_name and (vector_store is not None or report_files):
        result = run_news_check(
            customer_id=customer_id,
            customer_name=customer_name,
            adverse_media_db=adverse_db,
            entity_name=entity_name,
            report_files=report_files,
            vector_store=vector_store,   # RAG path; fallback to keyword if None
        )

    result_dict = {
        "customer_id":     result.customer_id,
        "customer_name":   result.customer_name,
        "news_risk_score": result.news_risk_score,
        "matched_alerts":  result.matched_alerts,
        "reasons":         result.reasons,
    }
    _KYC_CONTEXT["_news_result"] = result_dict
    return json.dumps(result_dict, indent=2)


@tool
def risk_scoring_tool(customer_id: str) -> str:
    """
    Perception-stage agent: combines identity check and news check results
    to compute a final weighted risk score and produce a KYC decision.

    Scoring weights:
      - Identity risk  contributes 60 %  (higher weight: identity is primary)
      - News / adverse media risk contributes 40 %

    Decision thresholds:
      - Score >= 60  →  REJECT
      - Score >= 30  →  MANUAL_REVIEW
      - Score <  30  →  APPROVE

    Args:
        customer_id: the customer ID being processed

    Returns a JSON string with final_risk_score, risk_level, decision, reasons.
    """
    from identity_check import IdentityCheckResult
    from news_check import NewsCheckResult

    id_data   = _KYC_CONTEXT.get("_identity_result", {})
    news_data = _KYC_CONTEXT.get("_news_result", {})

    # Reconstruct lightweight result objects expected by run_risk_scoring
    id_result = IdentityCheckResult(
        customer_id      = id_data["customer_id"],
        name_match       = id_data["name_match"],
        id_match         = id_data["id_match"],
        dob_match        = id_data["dob_match"],
        ocr_available    = True,
        ocr_fields       = id_data["ocr_fields"],
        confidence_score = id_data["confidence_score"],
        reasons          = id_data["reasons"],
    )

    news_result = NewsCheckResult(
        customer_id     = news_data["customer_id"],
        customer_name   = news_data["customer_name"],
        matched_alerts  = news_data["matched_alerts"],
        news_risk_score = news_data["news_risk_score"],
        reasons         = news_data["reasons"],
    )

    result = run_risk_scoring(id_result, news_result)

    return json.dumps({
        "final_risk_score": result.final_risk_score,
        "risk_level":       result.risk_level,
        "decision":         result.decision,
        "reasons":          result.reasons,
    }, indent=2)


# =============================================================================
# 3.  KYC Agent  —  wraps all three tools in a ReAct AgentExecutor
# =============================================================================

KYC_SYSTEM_PROMPT = """You are a KYC (Know Your Customer) compliance agent for a financial institution.

You have three tools available:
  - identity_check_tool: verifies the customer's ID document against the register
  - news_check_tool: screens the customer for adverse media and sanctions
  - risk_scoring_tool: computes the final weighted risk score and decision
                       (NOTE: this tool reads from internal state — you MUST call
                        identity_check_tool AND news_check_tool first)

Reason step by step. After each tool observation, think about what you learned
and what you should do next. Do NOT assume a fixed order blindly — use your
judgment:
  • If identity check returns confidence_score == 0 with all fields missing,
    that is a critical failure. Reason about whether to continue or reject early.
  • You must gather both identity and news results before calling risk_scoring_tool.
  • If an unexpected result occurs, explain your reasoning before proceeding.

Your final answer MUST include:
  1. The decision (APPROVE / MANUAL_REVIEW / REJECT) — on its own line, in caps.
  2. The final risk score.
  3. A 3-5 sentence explanation for the compliance record.
"""


def build_kyc_agent():
    """Creates the ReAct KYC agent using LangGraph prebuilt."""

    tools = [identity_check_tool, news_check_tool, risk_scoring_tool]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=KYC_SYSTEM_PROMPT,
    )
    return agent


# =============================================================================
# 4.  Main entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic KYC workflow with LangChain ReAct + Gemini"
    )
    parser.add_argument("--customer-id",       required=True)
    parser.add_argument("--document",          required=True,
                        help="Path to ID document (.png/.jpg/.txt)")
    parser.add_argument("--customers-csv",     default="data/customers_daily.csv")
    parser.add_argument("--adverse-media-json",default="data/adverse_media_mock.json")
    parser.add_argument("--entity-name",       default="",
                        help="Entity name for annual-report screening (auto-detected if omitted)")
    parser.add_argument("--reports-dir",       default="data/raw",
                        help="Directory with annual report PDFs (default: data/raw)")
    parser.add_argument("--output",            default="output/kyc_report_agent.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Populate shared context so tools can access file paths ────────────
    _KYC_CONTEXT["customers_df"]  = load_customers(args.customers_csv)
    _KYC_CONTEXT["doc_path"]      = args.document
    _KYC_CONTEXT["adverse_db"]    = load_adverse_media_db(args.adverse_media_json)

    # ── Auto-detect report PDFs ───────────────────────────────────────────
    reports_dir = Path(args.reports_dir) if args.reports_dir else None
    if reports_dir and reports_dir.exists():
        report_files = sorted(str(p) for p in reports_dir.glob("*.pdf"))
    else:
        report_files = []

    if report_files:
        print(f"[Auto] Found {len(report_files)} PDF(s) in '{args.reports_dir}'.")
    else:
        print(f"[Auto] No PDFs found in '{args.reports_dir}', skipping RAG.")

    # ── Auto-detect entity name from filenames if not provided ───────────
    entity_name = args.entity_name
    if not entity_name and report_files:
        entity_name = _auto_detect_entity_name(report_files)
        if entity_name:
            print(f"[Auto] Detected entity name: '{entity_name}'")

    _KYC_CONTEXT["entity_name"]  = entity_name
    _KYC_CONTEXT["report_files"] = report_files

    # ── Build RAG vector store only if PDFs are available ────────────────
    if report_files:
        vector_store = build_or_load_vector_store(report_files, api_key)
    else:
        vector_store = None
    _KYC_CONTEXT["vector_store"] = vector_store

    # ── Build and run the agent ────────────────────────────────────────────
    agent_executor = build_kyc_agent()

    task = (
        f"Process the KYC onboarding for customer_id='{args.customer_id}'. "
        f"Run all three checks (identity, news, risk scoring) and provide "
        f"a final compliance decision with explanation."
    )

    print("\n" + "="*60)
    print("  KYC AGENTIC WORKFLOW  (LangChain ReAct + Gemini)")
    print("="*60)
    print(f"  Customer : {args.customer_id}")
    print(f"  Document : {args.document}")
    print(f"  Entity   : {entity_name or '(none)'}")
    print(f"  RAG      : {'enabled (' + str(len(report_files)) + ' PDFs)' if vector_store else 'disabled'}")
    print("="*60 + "\n")

    result = agent_executor.invoke(
        {"messages": [("user", task)]},
        config={"recursion_limit": 16},
    )

    # ── Print and save the final report ───────────────────────────────────
    raw_content = result["messages"][-1].content
    if isinstance(raw_content, list):
        final_answer = " ".join(
            block.get("text", "") for block in raw_content
            if isinstance(block, dict) and "text" in block
        )
    else:
        final_answer = raw_content

    print("\n--- Agent Reasoning Steps ---")
    for msg in result["messages"]:
        msg_type = type(msg).__name__
        if msg_type == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[Thought→Action] LLM chose to call: {tc['name']}")
                    print(f"  with args: {tc['args']}")
            elif msg.content:
                # LLM thought without a tool call (intermediate reasoning)
                text = msg.content if isinstance(msg.content, str) else str(msg.content)
                print(f"\n[Thought] {text[:300]}")
        elif msg_type == "ToolMessage":
            print(f"\n[Observation] {msg.name} returned:")
            print(f"  {msg.content[:300]}...")

    print("\n" + "="*60)
    print("  FINAL KYC DECISION")
    print("="*60)
    print(final_answer)
    print("="*60 + "\n")

    # Collect intermediate tool outputs for the JSON report
    tool_outputs = {}
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            tool_outputs[msg.name] = msg.content
    intermediate_count = len(tool_outputs)

    report = {
        "customer_id":     args.customer_id,
        "agent_decision":  final_answer,
        "tool_outputs":    tool_outputs,
        "reasoning_steps": intermediate_count,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Report saved to: {output_path}\n")


if __name__ == "__main__":
    main()