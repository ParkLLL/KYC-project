# AgenticAI for Financial Compliance (KYC)

This project implements a simple, end-to-end KYC onboarding workflow aligned with the assignment:

1. **Customer Identification** (OCR + registry verification)
2. **News Check** (adverse media screening)
3. **Risk Scoring** (weighted decision: APPROVE / MANUAL_REVIEW / REJECT)

## Project Structure

- `main.py`: orchestration entrypoint
- `identity_check.py`: OCR/text extraction + field matching
- `news_check.py`: adverse media matching
- `risk_scoring.py`: final risk score and decision
- `data/customers_daily.csv`: trusted customer register
- `data/adverse_media_mock.json`: mock adverse media source
- `output/`: generated reports

## Setup

```bash
cd /Users/ianxyliu/kyc_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want OCR on image files, install Tesseract:

```bash
brew install tesseract
```

## Run Examples

Low risk example:

```bash
python main.py --customer-id C001 --document data/id_C001.txt
```

Low risk with one adverse hit (still auto-approve in current weights):

```bash
python main.py --customer-id C002 --document data/id_C002.txt
```

Medium risk example:

```bash
python main.py --customer-id C003 --document data/id_C003.txt
```

High risk (reject) example with identity mismatch + adverse hit:

```bash
python main.py --customer-id C003 --document data/id_C003_mismatch.txt
```

## Output

Each run writes a JSON report to:

- `output/kyc_report.json` (default)

The report includes:

- identity check match details
- adverse media hits
- final risk score, level, and decision reasons

## How This Meets Assignment

- **Action Stage Agent 1**: identity parser using `pytesseract` (or `.txt` input for demo)
- **Action Stage Agent 2**: adverse media screening module
- **Perception Stage Agent**: explainable weighted risk scoring

You can easily replace mock news data with live API data later.

## Annual Report Screening (Your SingPost Files)

You already placed these files in `data/raw/`:

- `SingPost AR 2022_23.pdf`
- `SingPost AR 2023_24.pdf`
- `SingPost AR 2024_25.pdf`

Run with report screening enabled:

```bash
.venv/bin/python main.py \
  --customer-id C001 \
  --document data/id_C001.txt \
  --entity-name "SingPost" \
  --reports-dir data/raw \
  --output output/kyc_report_singpost.json
```

This will combine:

- adverse-media JSON hits
- annual-report keyword screening results

## Split IC Image + Batch Run

Create 3 card images from the combined sample, then run IC001-IC003 in one shot:

```bash
.venv/bin/python batch_run_ic_samples.py \
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
