from __future__ import annotations

import argparse
import json
from pathlib import Path

from identity_check import load_customers, run_identity_check
from news_check import load_adverse_media_db, run_news_check
from risk_scoring import run_risk_scoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic KYC workflow: identity check -> news check -> risk scoring."
    )
    parser.add_argument("--customer-id", required=True, help="Customer ID to process.")
    parser.add_argument(
        "--document",
        required=True,
        help="Path to ID document image (.png/.jpg) or text sample (.txt).",
    )
    parser.add_argument(
        "--customers-csv",
        default="data/customers_daily.csv",
        help="Trusted customer register CSV path.",
    )
    parser.add_argument(
        "--adverse-media-json",
        default="data/adverse_media_mock.json",
        help="Adverse media JSON path.",
    )
    parser.add_argument(
        "--entity-name",
        default="",
        help="Entity name for annual-report screening (e.g., SingPost).",
    )
    parser.add_argument(
        "--reports-dir",
        default="",
        help="Directory containing annual report PDFs for screening.",
    )
    parser.add_argument(
        "--output",
        default="output/kyc_report.json",
        help="Output report path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    customers_df = load_customers(args.customers_csv)
    id_result = run_identity_check(
        customer_id=args.customer_id,
        customers_df=customers_df,
        doc_path=args.document,
    )

    customer_row = customers_df.loc[customers_df["customer_id"] == args.customer_id].iloc[0]
    customer_name = str(customer_row["name"])

    adverse_db = load_adverse_media_db(args.adverse_media_json)
    report_files: list[str] = []
    if args.reports_dir:
        report_files = sorted(str(p) for p in Path(args.reports_dir).glob("*.pdf"))
    news_result = run_news_check(
        customer_id=args.customer_id,
        customer_name=customer_name,
        adverse_media_db=adverse_db,
        entity_name=args.entity_name or customer_name,
        report_files=report_files,
    )

    risk_result = run_risk_scoring(id_result, news_result)

    report = {
        "customer_id": args.customer_id,
        "identity_check": {
            "confidence_score": id_result.confidence_score,
            "name_match": id_result.name_match,
            "id_match": id_result.id_match,
            "dob_match": id_result.dob_match,
            "ocr_fields": id_result.ocr_fields,
            "reasons": id_result.reasons,
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 40)
    print(f"Customer ID: {args.customer_id}")
    print(f"Risk score : {risk_result.final_risk_score}")
    print(f"Risk level : {risk_result.risk_level}")
    print(f"Decision   : {risk_result.decision}")
    print("=" * 40)
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
