"""Candidate solution for the local CalibForge reconciliation acceptance task."""

import csv
import json
from datetime import datetime
from pathlib import Path

DATA = Path("/app/data")
OUTPUT = Path("/app/output")
COLUMNS = (
    "bank_id",
    "ledger_id",
    "match_type",
    "bank_date",
    "ledger_date",
    "bank_amount",
    "ledger_amount",
    "amount_diff",
    "days_diff",
    "bank_description",
    "ledger_description",
)
ORDER = {"BANK_ONLY": 0, "AMOUNT_MISMATCH": 1, "MATCHED": 2, "LEDGER_ONLY": 3}


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def amount(row):
    if "Amount" in row:
        return float(row["Amount"])
    return float(row["Debit"] or row["Credit"])


def date(row):
    if "TransDate" in row:
        return datetime.strptime(row["TransDate"], "%m/%d/%Y").date()
    return datetime.strptime(row["EntryDate"], "%Y-%m-%d").date()


def report_row(bank=None, ledger=None, match_type=""):
    bank_date = date(bank).strftime("%d-%m-%Y") if bank else ""
    ledger_date = date(ledger).strftime("%d-%m-%Y") if ledger else ""
    return {
        "bank_id": bank["BankRef"] if bank else "",
        "ledger_id": ledger["LedgerID"] if ledger else "",
        "match_type": match_type,
        "bank_date": bank_date,
        "ledger_date": ledger_date,
        "bank_amount": bank["Amount"] if bank else "",
        "ledger_amount": f"{amount(ledger):.2f}" if ledger else "",
        "amount_diff": f"{abs(amount(bank) - amount(ledger)):.2f}" if bank and ledger else "",
        "days_diff": str(abs((date(bank) - date(ledger)).days)) if bank and ledger else "",
        "bank_description": bank["Description"] if bank else "",
        "ledger_description": ledger["Description"] if ledger else "",
    }


def reconcile(banks, ledgers):
    remaining = set(range(len(ledgers)))
    rows = []
    for bank in sorted(banks, key=date):
        close = [
            (index, ledger)
            for index, ledger in enumerate(ledgers)
            if index in remaining and abs((date(bank) - date(ledger)).days) <= 3
        ]
        exact = [(index, ledger) for index, ledger in close if abs(amount(bank) - amount(ledger)) <= 0.50]
        if exact:
            index, ledger = min(
                exact,
                key=lambda pair: (
                    abs((date(bank) - date(pair[1])).days),
                    abs(amount(bank) - amount(pair[1])),
                ),
            )
            rows.append(report_row(bank, ledger, "MATCHED"))
        else:
            if not close:
                rows.append(report_row(bank, match_type="BANK_ONLY"))
                continue
            index, ledger = min(
                close,
                key=lambda pair: (
                    abs((date(bank) - date(pair[1])).days),
                    abs(amount(bank) - amount(pair[1])),
                ),
            )
            rows.append(report_row(bank, ledger, "AMOUNT_MISMATCH"))
        remaining.remove(index)
    rows.extend(report_row(ledger=ledgers[index], match_type="LEDGER_ONLY") for index in sorted(remaining))
    rows.sort(
        key=lambda row: (ORDER[row["match_type"]], row["bank_date"][6:] + row["bank_date"][3:5] + row["bank_date"][:2])
    )
    return rows


def main():
    banks = read_rows(DATA / "bank_statement.csv")
    ledgers = read_rows(DATA / "ledger.csv")
    rows = reconcile(banks, ledgers)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "reconciliation_report.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "total_bank_records": len(banks),
        "total_ledger_records": len(ledgers),
        "matched_count": sum(row["match_type"] == "MATCHED" for row in rows),
        "amount_mismatch_count": sum(row["match_type"] == "AMOUNT_MISMATCH" for row in rows),
        "bank_only_count": sum(row["match_type"] == "BANK_ONLY" for row in rows),
        "ledger_only_count": sum(row["match_type"] == "LEDGER_ONLY" for row in rows),
        "total_amount_diff": sum(float(row["amount_diff"]) for row in rows if row["match_type"] == "AMOUNT_MISMATCH"),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
