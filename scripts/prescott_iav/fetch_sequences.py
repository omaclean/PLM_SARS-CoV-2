#!/usr/bin/env python3
"""Download nucleotide sequences for a list of accessions, resumably.

Companion to ``fetch_collection_dates.py``. The population panel is assembled from two
sources -- the pre-aligned local pool and whatever it is missing -- and this fetches the
missing part so the panel is the complete date window rather than the subset that happened
to survive an earlier pipeline's filters.

Resumable by design: it reads back whatever is already in the output FASTA and only requests
the remainder, so a dropped connection costs one batch rather than the whole run.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Set

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def chunked(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def request(url: str, data: Optional[bytes], retries: int, pause: float) -> str:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, data=data, timeout=180) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            last = error
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


def existing_ids(fasta: Path) -> Set[str]:
    if not fasta.exists():
        return set()
    found: Set[str] = set()
    with fasta.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                found.add(line[1:].split()[0].split(".")[0].strip())
    return found


def fetch_batch(accessions: Sequence[str], api_key: Optional[str],
                retries: int, pause: float) -> str:
    common = {"db": "nuccore", "tool": "plm_sars_prescott", "email": "noreply@example.org"}
    if api_key:
        common["api_key"] = api_key
    body = urllib.parse.urlencode({**common, "id": ",".join(accessions)}).encode()
    posted = request(f"{EUTILS}/epost.fcgi", body, retries, pause)
    try:
        key = posted.split("<QueryKey>")[1].split("</QueryKey>")[0]
        web = posted.split("<WebEnv>")[1].split("</WebEnv>")[0]
    except IndexError as error:
        raise RuntimeError(f"epost returned no history handle: {posted[:300]}") from error
    query = urllib.parse.urlencode({**common, "query_key": key, "WebEnv": web,
                                    "rettype": "fasta", "retmode": "text"})
    return request(f"{EUTILS}/efetch.fcgi?{query}", None, retries, pause)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accessions", type=Path, required=True,
                        help="one accession per line")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--sleep", type=float, default=None)
    parser.add_argument("--retries", type=int, default=4)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    pause = args.sleep if args.sleep is not None else (0.15 if args.api_key else 0.40)

    wanted = [line.strip() for line in
              args.accessions.read_text(encoding="utf-8").splitlines() if line.strip()]
    have = existing_ids(args.out)
    pending = [accession for accession in wanted if accession not in have]
    print(f"@> {len(wanted):,} wanted, {len(have):,} already present, "
          f"{len(pending):,} to fetch", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = (len(pending) + args.batch_size - 1) // args.batch_size
    with args.out.open("a", encoding="utf-8") as handle:
        for index, batch in enumerate(chunked(pending, args.batch_size), start=1):
            try:
                handle.write(fetch_batch(batch, args.api_key, args.retries, pause))
                handle.flush()
            except RuntimeError as error:
                print(f"@> batch {index}/{total} FAILED: {error}", flush=True)
                continue
            print(f"@> batch {index}/{total}", flush=True)
            time.sleep(pause)

    final = existing_ids(args.out)
    print(f"@> {args.out}: {len(final):,} sequences "
          f"({len(set(wanted) - final):,} still missing)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
