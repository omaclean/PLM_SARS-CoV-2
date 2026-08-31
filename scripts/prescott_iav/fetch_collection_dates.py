#!/usr/bin/env python3
"""Resolve NCBI collection dates for a FASTA's accessions, so a population can be date-windowed.

``Sequences/gisaid_and_genbank_data/unmatched_accessions.fasta`` is 343,569 pre-aligned H3N2
HA nucleotide sequences keyed by bare GenBank accession. Nothing local carries a collection
date, so a "sequences from 2021-2023" population cannot be built without asking NCBI.

esummary returns the date inside a packed pair of fields::

    "subtype": "strain|serotype|host|country|segment|collection_date|note"
    "subname": "A/Georgia/32/2022|H3N2|Homo sapiens|USA: Georgia|4|24-Oct-2022|passage..."

so the position of ``collection_date`` in ``subtype`` is the index to take from ``subname``.
It is NOT a fixed column -- records vary in which qualifiers they carry -- which is why this
zips the two rather than slicing a constant offset.

Uses epost + esummary against the History server: a 500-accession GET would blow the URL
length limit, and epost is the documented route for large id sets. Results are appended to a
TSV as they arrive and the file is resumable, because a 690-batch run should not have to
start over because batch 600 timed out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Month abbreviations as they appear in GenBank collection_date qualifiers.
MONTHS = {name: index for index, name in enumerate(
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), start=1)}


def entrez_accessions(query: str, api_key: Optional[str], retries: int,
                      pause: float) -> List[str]:
    """Every accession matching an Entrez query, via the History server.

    Deliberately NOT filtered by date here: ``[Collection Date]`` is an NCBI Virus field and
    does not exist on nuccore, so an esearch date filter would silently fall back to PDAT
    (deposit date) and quietly select the wrong sequences. The real collection date comes
    from esummary per record and the window is applied downstream.
    """
    common = {"db": "nuccore", "tool": "plm_sars_prescott", "email": "noreply@example.org"}
    if api_key:
        common["api_key"] = api_key

    search = urllib.parse.urlencode({**common, "term": query, "retmode": "json",
                                     "retmax": 0, "usehistory": "y"})
    payload = json.loads(request(f"{EUTILS}/esearch.fcgi?{search}", None, retries, pause))
    result = payload["esearchresult"]
    total = int(result["count"])
    key, web = result["querykey"], result["webenv"]
    print(f"@> Entrez query matches {total:,} records", flush=True)

    accessions: List[str] = []
    step = 10000
    for start in range(0, total, step):
        fetch = urllib.parse.urlencode({**common, "query_key": key, "WebEnv": web,
                                        "rettype": "acc", "retmode": "text",
                                        "retstart": start, "retmax": step})
        text = request(f"{EUTILS}/efetch.fcgi?{fetch}", None, retries, pause)
        accessions.extend(line.split(".")[0].strip()
                          for line in text.splitlines() if line.strip())
        print(f"@> collected {len(accessions):,}/{total:,} accessions", flush=True)
        time.sleep(pause)
    return accessions


def read_accessions(fasta: Path) -> List[str]:
    accessions: List[str] = []
    with fasta.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                accessions.append(line[1:].split()[0].strip())
    return accessions


def chunked(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def request(url: str, data: Optional[bytes], retries: int, pause: float) -> str:
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, data=data, timeout=120) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
            last = error
            # Exponential backoff: NCBI throttles hard and a tight retry makes it worse.
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


def epost_esummary(accessions: Sequence[str], api_key: Optional[str],
                   retries: int, pause: float) -> List[Dict[str, object]]:
    common = {"db": "nuccore", "tool": "plm_sars_prescott", "email": "noreply@example.org"}
    if api_key:
        common["api_key"] = api_key

    post_body = urllib.parse.urlencode({**common, "id": ",".join(accessions)}).encode()
    posted = request(f"{EUTILS}/epost.fcgi", post_body, retries, pause)
    try:
        key = posted.split("<QueryKey>")[1].split("</QueryKey>")[0]
        web = posted.split("<WebEnv>")[1].split("</WebEnv>")[0]
    except IndexError as error:
        raise RuntimeError(f"epost returned no history handle: {posted[:300]}") from error

    query = urllib.parse.urlencode({**common, "query_key": key, "WebEnv": web,
                                    "retmode": "json", "retmax": len(accessions)})
    payload = json.loads(request(f"{EUTILS}/esummary.fcgi?{query}", None, retries, pause))
    result = payload.get("result", {})
    return [result[uid] for uid in result.get("uids", []) if uid in result]


def extract(record: Dict[str, object]) -> Dict[str, str]:
    """Pull accession, strain, collection date and host out of one esummary record."""
    fields = str(record.get("subtype", "")).split("|")
    values = str(record.get("subname", "")).split("|")
    packed = dict(zip(fields, values))
    return {
        "accession": str(record.get("caption", "")),
        "strain": str(record.get("strain", "") or packed.get("strain", "")),
        "collection_date": packed.get("collection_date", ""),
        "host": packed.get("host", ""),
        "country": packed.get("country", ""),
        "serotype": packed.get("serotype", ""),
        "slen": str(record.get("slen", "")),
    }


def normalise_date(raw: str) -> Tuple[str, str]:
    """GenBank collection_date -> (ISO-ish date, year). Handles the three common shapes.

    '24-Oct-2022' -> ('2022-10-24', '2022');  'Oct-2022' -> ('2022-10', '2022');
    '2022' -> ('2022', '2022').  Anything else returns ('', '') rather than guessing --
    a mis-parsed date would silently move a sequence into or out of the window.
    """
    text = raw.strip()
    if not text:
        return "", ""
    parts = text.split("-")
    try:
        if len(parts) == 3 and parts[1] in MONTHS:
            return f"{int(parts[2]):04d}-{MONTHS[parts[1]]:02d}-{int(parts[0]):02d}", parts[2]
        if len(parts) == 2 and parts[0] in MONTHS:
            return f"{int(parts[1]):04d}-{MONTHS[parts[0]]:02d}", parts[1]
        if len(parts) == 1 and len(text) == 4 and text.isdigit():
            return text, text
        # Already ISO (some records carry 2022-10-24 directly).
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}", parts[0]
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            return f"{int(parts[0]):04d}-{int(parts[1]):02d}", parts[0]
    except (ValueError, IndexError):
        return "", ""
    return "", ""


COLUMNS = ("accession", "collection_date", "date_iso", "year", "strain", "host",
           "country", "serotype", "slen")


def already_done(out_path: Path) -> Set[str]:
    if not out_path.exists():
        return set()
    seen: Set[str] = set()
    with out_path.open(encoding="utf-8") as handle:
        header = handle.readline()
        if not header.startswith("accession"):
            return set()
        for line in handle:
            accession = line.split("\t", 1)[0].strip()
            if accession:
                seen.add(accession)
    return seen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fasta", type=Path, help="take accessions from this FASTA's headers")
    source.add_argument("--entrez-query", help="take accessions from an Entrez nuccore query")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accession-cache", type=Path, default=None,
                        help="with --entrez-query, cache the accession list here so a rerun "
                             "does not repeat the search")
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--api-key", default=None,
                        help="NCBI API key; raises the rate limit from 3/s to 10/s")
    parser.add_argument("--sleep", type=float, default=None,
                        help="seconds between batches; default 0.15 with a key, 0.40 without")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="stop after N accessions (testing)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sleep = args.sleep if args.sleep is not None else (0.15 if args.api_key else 0.40)

    if args.fasta:
        accessions = read_accessions(args.fasta)
    elif args.accession_cache and args.accession_cache.exists():
        accessions = [line.strip() for line in
                      args.accession_cache.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"@> reusing {len(accessions):,} cached accessions from {args.accession_cache}",
              flush=True)
    else:
        accessions = entrez_accessions(args.entrez_query, args.api_key, args.retries, sleep)
        if args.accession_cache:
            args.accession_cache.parent.mkdir(parents=True, exist_ok=True)
            args.accession_cache.write_text("\n".join(accessions) + "\n", encoding="utf-8")
    if args.limit:
        accessions = accessions[:args.limit]
    done = already_done(args.out)
    pending = [accession for accession in accessions if accession not in done]
    print(f"@> {len(accessions):,} accessions, {len(done):,} already resolved, "
          f"{len(pending):,} to fetch", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fresh = not args.out.exists() or not done
    with args.out.open("a" if done else "w", encoding="utf-8") as handle:
        if fresh:
            handle.write("\t".join(COLUMNS) + "\n")
            handle.flush()
        total_batches = (len(pending) + args.batch_size - 1) // args.batch_size
        resolved = 0
        for index, batch in enumerate(chunked(pending, args.batch_size), start=1):
            try:
                records = epost_esummary(batch, args.api_key, args.retries, sleep)
            except RuntimeError as error:
                print(f"@> batch {index}/{total_batches} FAILED: {error}", flush=True)
                continue
            for record in records:
                row = extract(record)
                iso, year = normalise_date(row["collection_date"])
                row["date_iso"], row["year"] = iso, year
                handle.write("\t".join(row.get(column, "").replace("\t", " ")
                                       for column in COLUMNS) + "\n")
                resolved += 1
            handle.flush()
            if index % 20 == 0 or index == total_batches:
                print(f"@> batch {index}/{total_batches}  resolved {resolved:,}", flush=True)
            time.sleep(sleep)
    print(f"@> wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
