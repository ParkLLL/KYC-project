# AgenticAI for Financial Compliance (KYC)

This project implements an agentic KYC onboarding workflow using **LangChain ReAct + Google Gemini**. The LLM acts as the reasoning brain, dynamically deciding which tools to call based on each tool's observation — not a fixed pipeline.

1. **Customer Identification** (OCR + registry verification)
2. **News Check** (adverse media screening + optional annual report screening)
3. **Risk Scoring** (weighted decision: APPROVE / MANUAL_REVIEW / REJECT)

## Project Structure

- `main.py`: agentic orchestration entrypoint (ReAct agent)
- `identity_check.py`: OCR/text extraction + field matching
- `news_check.py`: adverse media matching + annual report keyword screening
- `risk_scoring.py`: final risk score and decision
- `data/customers_daily.csv`: trusted customer register
- `data/adverse_media_mock.json`: mock adverse media source
- `data/raw/`: annual report PDFs for entity screening
- `output/`: generated reports

## Setup

```bash
conda activate project6
pip install -r requirements.txt
```

If you want OCR on image files, install Tesseract:

```bash
brew install tesseract
```

## Run Examples

Low risk (clean customer):

```bash
python main.py --customer-id C001 --document data/id_C001.txt
```

Adverse media hit (procurement dispute), with SingPost annual report screening:

```bash
python main.py \
  --customer-id C002 \
  --document data/id_C002.txt \
  --entity-name "SingPost" \
  --reports-dir data/raw \
  --output output/kyc_report_singpost.json
```

> Note: annual report screening only applies when the customer already has an adverse media hit.
> C001 (no adverse hit) will not be affected by SingPost reports even if `--entity-name` is passed.

Medium risk (sanctions-adjacent), with SingPost annual report screening:

```bash
python main.py \
  --customer-id C003 \
  --document data/id_C003.txt \
  --entity-name "SingPost" \
  --reports-dir data/raw \
  --output output/kyc_report_singpost.json
```

High risk (identity mismatch + adverse hit):

```bash
python main.py --customer-id C003 --document data/id_C003_mismatch.txt
```

## Output

Each run writes a JSON report to `output/kyc_report_agent.json` by default (override with `--output`).

The report includes:

- identity check match details
- adverse media hits (and annual report keyword hits if applicable)
- final risk score, level, and decision reasons
- full agent reasoning trace (tool calls and observations)

## How This Meets Assignment

- **Action Stage Agent 1**: identity parser using `pytesseract` (or `.txt` input for demo)
- **Action Stage Agent 2**: adverse media + annual report screening module
- **Perception Stage Agent**: explainable weighted risk scoring
- **ReAct Agent (LLM brain)**: Gemini reasons step-by-step — Thought → Action → Observation — dynamically deciding tool call order and handling unexpected results

## Annual Report Screening (SingPost Files)

Annual report PDFs are placed in `data/raw/`:

- `SingPost AR 2022_23.pdf`
- `SingPost AR 2023_24.pdf`
- `SingPost AR 2024_25.pdf`

Screening scans these PDFs for risk-related keywords (fraud, sanction, money laundering, etc.) and combines the score with the adverse media score. **Screening is only triggered if the customer already has an adverse media hit**, preventing unrelated customers from being affected by the entity's report contents.

## Split IC Image + Batch Run

Create 3 card images from the combined sample, then run IC001-IC003 in one shot:

```bash
python batch_run_ic_samples.py \
  --entity-name "SingPost" \
  --reports-dir data/raw \
  --output-dir output/batch_ic
```

Outputs:

- split cards: `data/processed/cards/IC001.png`, `IC002.png`, `IC003.png`
- per-customer reports: `output/batch_ic/IC001.json` ... `IC003.json`
- summary table: `output/batch_ic/summary.csv`

If your machine has Tesseract installed, the script uses OCR on the split images.
If not, it automatically falls back to seeded text samples for demo continuity.
