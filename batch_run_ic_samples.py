from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from identity_check import load_customers, run_identity_check
from news_check import load_adverse_media_db, run_news_check
from risk_scoring import run_risk_scoring
from split_ic_cards import split_cards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split sample IC image and run batch KYC for IC001-IC003."
    )
    parser.add_argument(
        "--source-image",
        default="data/raw/Hypothetical images of ICs.png",
        help="Combined source image containing 3 ID cards.",
    )
    parser.add_argument(
        "--customers-csv",
        default="data/customers_ic_sample.csv",
        help="Customer registry for IC samples.",
    )
    parser.add_argument(
        "--adverse-media-json",
        default="data/adverse_media_ic_sample.json",
        help="Adverse media source for IC samples.",
    )
    parser.add_argument(
        "--entity-name",
        default="SingPost",
        help="Entity name for annual report screening.",
    )
    parser.add_argument(
        "--reports-dir",
        default="data/raw",
        help="Directory containing annual report PDFs.",
    )
    parser.add_argument(
        "--use-seed-text",
        action="store_true",
        help="Use seed text files instead of OCR image input.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/batch_ic",
        help="Directory to write per-customer reports and summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always split once so user can see each generated card image.
    split_cards(args.source_image, "data/processed/cards")

    customers_df = load_customers(args.customers_csv)
    adverse_db = load_adverse_media_db(args.adverse_media_json)
    report_files = sorted(str(p) for p in Path(args.reports_dir).glob("*.pdf"))

    summary_rows: list[dict[str, object]] = []

    for _, row in customers_df.iterrows():
        customer_id = str(row["customer_id"])
        customer_name = str(row["name"])

        if args.use_seed_text:
            doc_path = f"data/ic_seed_text/{customer_id}.txt"
        else:
            doc_path = f"data/processed/cards/{customer_id}.png"

        try:
            identity_result = run_identity_check(customer_id, customers_df, doc_path)
        except Exception as exc:
            # Graceful fallback in case OCR engine is unavailable.
            fallback_doc = f"data/ic_seed_text/{customer_id}.txt"
            identity_result = run_identity_check(customer_id, customers_df, fallback_doc)
            doc_path = fallback_doc
            print(f"[{customer_id}] OCR unavailable, fallback to seed text: {exc}")

        news_result = run_news_check(
            customer_id=customer_id,
            customer_name=customer_name,
            adverse_media_db=adverse_db,
            entity_name=args.entity_name,
            report_files=report_files,
        )
        risk_result = run_risk_scoring(identity_result, news_result)

        report = {
            "customer_id": customer_id,
            "document_used": doc_path,
            "identity_check": {
                "confidence_score": identity_result.confidence_score,
                "name_match": identity_result.name_match,
                "id_match": identity_result.id_match,
                "dob_match": identity_result.dob_match,
                "ocr_fields": identity_result.ocr_fields,
                "reasons": identity_result.reasons,
            },
            "news_check": {
                "news_risk_score": news_result.news_risk_score,
                "matched_alerts": news_result.matched_alerts,
                "reasons": news_result.reasons,
                "reports_scanned": report_files,
            },
            "risk_scoring": {
                "final_risk_score": risk_result.final_risk_score,
                "risk_level": risk_result.risk_level,
                "decision": risk_result.decision,
                "reasons": risk_result.reasons,
            },
        }

        report_path = output_dir / f"{customer_id}.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_rows.append(
            {
                "customer_id": customer_id,
                "document_used": doc_path,
                "identity_confidence": identity_result.confidence_score,
                "news_risk_score": news_result.news_risk_score,
                "final_risk_score": risk_result.final_risk_score,
                "risk_level": risk_result.risk_level,
                "decision": risk_result.decision,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "summary.csv"
    summary_json = output_dir / "summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 50)
    print("Batch KYC completed.")
    print(f"Per-customer reports: {output_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    print("=" * 50)


if __name__ == "__main__":
    main()
