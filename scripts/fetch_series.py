"""Download the expanded predictors into data/raw/. Run on your own computer.

Usage: python scripts/fetch_series.py
Failed downloads are listed at the end, so one bad series ID does not stop the rest.
"""

from __future__ import annotations

import sys
import urllib.request

from fuelsignal import config
from fuelsignal.expanded import NEW_SERIES

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
EIA = "https://www.eia.gov/dnav/pet/hist_xls/{}"


def url_for(file: str, name: str) -> str:
    return FRED.format(name) if file.endswith(".csv") else EIA.format(file)


def looks_valid(file: str, content: bytes) -> bool:
    if file.endswith(".csv"):
        return content.startswith(b"observation_date") or content.startswith(b"DATE")
    return content.startswith(b"\xd0\xcf\x11\xe0")


def main() -> int:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for spec in NEW_SERIES:
        try:
            request = urllib.request.Request(
                url_for(spec.file, spec.name), headers={"User-Agent": "fuelsignal-data-fetch"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            if not looks_valid(spec.file, content):
                raise ValueError("response was not the expected file type")
            (config.RAW_DIR / spec.file).write_bytes(content)
            print(f"ok      {spec.file}")
        except Exception as error:  # noqa: BLE001
            failed.append(spec.file)
            print(f"FAILED  {spec.file}: {error}")
    print(f"\n{len(NEW_SERIES) - len(failed)} downloaded, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
