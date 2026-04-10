#!/usr/bin/env python3
"""
Check temperature relationship claims on reefer_release.csv.

Usage:
  python3 main/doc/check_temperature_quality.py
  python3 main/doc/check_temperature_quality.py /path/to/reefer_release.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def to_float(value: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    # Dataset uses decimal comma, e.g. "6,25"
    value = value.replace(",", ".")
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    default_path = Path("/Users/omkarsomeshwarkondhalkar/Movies/project/eurogate/participant_package/reefer_release.csv")
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    total_rows = 0
    valid_rows = 0
    supply_lt_return = 0
    supply_ge_return = 0
    supply_eq_return = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            total_rows += 1
            ret = to_float(row.get("TemperatureReturn", ""))
            sup = to_float(row.get("RemperatureSupply", ""))  # Note: column name in file has this spelling.
            if ret is None or sup is None:
                continue

            valid_rows += 1
            if sup < ret:
                supply_lt_return += 1
            else:
                supply_ge_return += 1
            if sup == ret:
                supply_eq_return += 1

    if valid_rows == 0:
        raise SystemExit("No valid rows found for TemperatureReturn/RemperatureSupply.")

    pct_lt = 100.0 * supply_lt_return / valid_rows
    pct_ge = 100.0 * supply_ge_return / valid_rows
    pct_eq = 100.0 * supply_eq_return / valid_rows

    print(f"File: {csv_path}")
    print(f"Rows total: {total_rows}")
    print(f"Rows with both temperatures present: {valid_rows}")
    print(f"Supply < Return: {supply_lt_return} ({pct_lt:.4f}%)")
    print(f"Supply >= Return: {supply_ge_return} ({pct_ge:.4f}%)")
    print(f"Supply == Return: {supply_eq_return} ({pct_eq:.4f}%)")

    print("\nInterpretation:")
    if abs(pct_lt - 67.2) <= 0.5:
        print("- The numeric claim (~67.2% Supply < Return) is TRUE for this dataset.")
    else:
        print("- The numeric claim (~67.2% Supply < Return) is FALSE for this dataset.")
    print(
        "- Thermodynamically, for air temperatures in a reefer, Supply being lower than Return is often expected; "
        "so this pattern alone is not sufficient to declare a data-quality error."
    )


if __name__ == "__main__":
    main()

