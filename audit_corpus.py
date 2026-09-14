"""
Day 1, Step 1: AUDIT the corpus before doing anything else.

Why this exists:
    Many older FDA filings are SCANNED PAPER -- the page is a photograph of
    text, not text. Extraction returns nothing for those. If you don't check,
    you end up with a corpus that's silently one-third empty and you won't
    find out until your eval scores are bad for reasons you can't explain.

    Always look at your data before you process it.

Run:  python audit_corpus.py
Out:  data/audit.csv  + a printed summary
"""

import csv
import sys
from pathlib import Path

import fitz  # this is PyMuPDF

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = Path("data/raw")

# A page with fewer characters than this is almost certainly a scan
# (a normal text page of a 510k summary has 1500-3000 characters).
CHARS_PER_PAGE_THRESHOLD = 100


def audit_pdf(path: Path) -> dict:
    """Open one PDF and measure how much real text it contains."""
    try:
        doc = fitz.open(path)
    except Exception as e:
        return {"file": path.name, "status": "CORRUPT", "error": str(e),
                "pages": 0, "chars": 0, "chars_per_page": 0}

    total_chars = 0
    pages = doc.page_count
    for page in doc:
        total_chars += len(page.get_text().strip())
    doc.close()

    cpp = total_chars / pages if pages else 0

    if cpp < CHARS_PER_PAGE_THRESHOLD:
        status = "SCANNED"      # image-only, needs OCR or exclusion
    elif cpp < 400:
        status = "SPARSE"       # partly scanned, or mostly tables/figures
    else:
        status = "TEXT"         # good, use as-is

    return {
        "file": path.name,
        "folder": path.parent.name,
        "status": status,
        "pages": pages,
        "chars": total_chars,
        "chars_per_page": round(cpp),
        "error": "",
    }


def main() -> None:
    pdfs = sorted(RAW.rglob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under {RAW.resolve()}")
        return

    print(f"Auditing {len(pdfs)} PDFs...\n")
    rows = [audit_pdf(p) for p in pdfs]

    # Write the full report
    out = Path("data/audit.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "folder", "status", "pages", "chars",
                        "chars_per_page", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Summarize
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print("=" * 55)
    print("RESULTS")
    print("=" * 55)
    for status in ("TEXT", "SPARSE", "SCANNED", "CORRUPT"):
        n = counts.get(status, 0)
        if n:
            pct = 100 * n / len(rows)
            print(f"  {status:<9} {n:>4} files  ({pct:.0f}%)")

    usable = counts.get("TEXT", 0) + counts.get("SPARSE", 0)
    total_pages = sum(r["pages"] for r in rows if r["status"] != "SCANNED")

    print(f"\n  Usable documents: {usable} / {len(rows)}")
    print(f"  Usable pages:     {total_pages}")

    scanned = [r["file"] for r in rows if r["status"] == "SCANNED"]
    if scanned:
        print(f"\n  Scanned (no extractable text) -- first 15:")
        for name in scanned[:15]:
            print(f"    {name}")
        if len(scanned) > 15:
            print(f"    ... and {len(scanned) - 15} more")

    print(f"\nFull report: {out}")
    print("\nWHAT TO DO NEXT")
    print("-" * 55)
    if usable >= 90:
        print("  Enough usable documents. Exclude the scanned ones and move on.")
        print("  Note the exclusion in your README -- it's a real finding.")
    else:
        print("  Below ~90 usable docs. Two options:")
        print("   (a) OCR the scanned ones (slower, but recovers them)")
        print("   (b) Download more recent clearances, which are digital-native")
        print("  Option (b) is faster and I'd take it on a one-week timeline.")


if __name__ == "__main__":
    main()
