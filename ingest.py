"""
Day 1: Turn PDFs into labeled chunks.  (v2 -- fixed heading detection)

WHAT WENT WRONG IN v1
    Cleaning collapsed single newlines into spaces BEFORE headings were
    detected. Headings are recognised by sitting alone on their own line --
    so once the line breaks were gone, nothing matched, and every chunk was
    labelled "(document start)" on page 1.

    The fix: keep line structure all the way through detection, and flatten
    the text into flowing prose only at the very end, per chunk.

    Worth remembering: the bug wasn't visible in the counts. 1,468 chunks
    looked like success. Only reading the actual output revealed it.

Run:  python ingest.py
Out:  data/chunks.json  +  data/chunk_stats.csv
"""

import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import pymupdf  # the modern name for fitz

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW = Path("data/raw")
OUT_CHUNKS = Path("data/chunks.json")
OUT_STATS = Path("data/chunk_stats.csv")

TARGET_WORDS = 300
MAX_WORDS = 450
MIN_WORDS = 40
OVERLAP_WORDS = 50


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    doc_type: str
    doc_title: str
    section: str
    page_start: int
    page_end: int
    chunk_index: int
    word_count: int
    source_file: str


# ---------------------------------------------------------------------------
# Step 1: PDF -> text, KEEPING line structure
# ---------------------------------------------------------------------------

def extract_pages(path: Path) -> list[str]:
    doc = pymupdf.open(path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def clean_preserving_lines(text: str) -> str:
    """
    Light cleanup that KEEPS newlines, because heading detection depends on
    them. We only fix things that are unambiguously artifacts.
    """
    # Re-join words hyphenated across a line break: "steriliza-\ntion"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Normalise line endings and trim trailing spaces on each line
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse runs of spaces (but not newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Drop page furniture that appears on its own line
    text = re.sub(r"^\s*Page \d+ of \d+\s*$", "", text, flags=re.I | re.M)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*Contains Nonbinding Recommendations\s*$", "",
                  text, flags=re.I | re.M)

    # Collapse 3+ blank lines to a paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def flatten(text: str) -> str:
    """
    Final pass, applied per chunk: turn line-broken text into flowing prose.
    Single newlines become spaces; blank lines stay as paragraph breaks.
    """
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 2: Heading detection (now run on line-structured text)
# ---------------------------------------------------------------------------

SECTION_510K = re.compile(
    r"^[ \t]*(?:[IVX]+\.|\d+\.|\(\w\))?[ \t]*"
    r"(submitter|submitted by|applicant|contact person|date prepared"
    r"|device name|name of device|trade name|proprietary name|common name"
    r"|classification name|classification|regulation number|product code"
    r"|predicate device[s]?|legally marketed predicate"
    r"|device description|description of the device|description"
    r"|intended use|indications for use|statement of intended use"
    r"|technological characteristics|comparison to predicate"
    r"|performance data|performance testing|non-?clinical testing"
    r"|bench testing|clinical testing|clinical data"
    r"|biocompatibility|sterilization|sterility|shelf life|packaging"
    r"|substantial equivalence|conclusion[s]?|summary)"
    r"[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)

SECTION_GUIDANCE = re.compile(
    r"^[ \t]*("
    r"(?:[IVXLC]{1,6}\.)[ \t]+[A-Z][^\n]{2,80}"          # "III. Scope"
    r"|(?:\d{1,2}\.(?:\d{1,2}){0,3})[ \t]+[A-Z][^\n]{2,80}"     # "3. Scope"
    r"|(?:\d{1,2}(?:\.\d{1,2}){1,3})[ \t]+[A-Z][^\n]{2,80}"     # "4.2 Biocompatibility"
    r"|(?:[A-Z]\.)[ \t]+[A-Z][^\n]{2,80}"                # "B. Background"
    r"|(?:APPENDIX|ANNEX|ATTACHMENT|SECTION)[ \t]+[A-Z0-9][^\n]{0,60}"
    r"|[A-Z][A-Z0-9\s,&/()\-']{7,70}"                    # ALL CAPS HEADING
    r")[ \t]*$",
    re.MULTILINE,
)


def looks_like_heading(candidate: str) -> bool:
    """Reject false positives that the regex lets through."""
    c = candidate.strip()
    if len(c) < 3 or len(c) > 90:
        return False
    if c.endswith((".", ",", ";", ":")) and not re.match(r"^[IVXLC\d]+\.", c):
        # A sentence that happens to be short -- not a heading
        if c.endswith((".", ",", ";")):
            return False
    if c.count(" ") > 12:          # headings are short
        return False

    # Footnotes masquerade as numbered headings in FDA guidance
    if re.match(r"^\d{1,3}\s+(See|see|Id\.|Ibid|Section|21 CFR)", c):
        return False
    if "http" in c.lower() or "www." in c.lower():
        return False
    if re.search(r"\b(21|42)\s+CFR\b", c) and len(c.split()) < 8:
        return False
    if c.endswith(("(See", "(see", "e.g.", "i.e.")):
        return False

    return True


def find_sections(text: str, doc_type: str) -> list[tuple[int, str]]:
    pattern = SECTION_510K if doc_type == "510k" else SECTION_GUIDANCE

    found: list[tuple[int, str]] = []
    for m in pattern.finditer(text):
        heading = m.group(1).strip()
        if looks_like_heading(heading):
            found.append((m.start(), heading))

    # Drop consecutive duplicates (headers repeated on every page)
    deduped: list[tuple[int, str]] = []
    for pos, head in found:
        if not deduped or deduped[-1][1].lower() != head.lower():
            deduped.append((pos, head))

    if not deduped or deduped[0][0] > 0:
        deduped.insert(0, (0, "(preamble)"))

    return deduped


# ---------------------------------------------------------------------------
# Step 3: Split, tracking character offsets so page numbers stay correct
# ---------------------------------------------------------------------------

def split_long_text(text: str, target: int, maximum: int,
                    overlap: int) -> list[str]:
    if len(text.split()) <= maximum:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current: list[str] = []

    for para in paragraphs:
        para_words = para.split()

        if len(para_words) > maximum:
            if current:
                pieces.append(" ".join(current))
                current = []
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf: list[str] = []
            for sent in sentences:
                sw = sent.split()
                if len(buf) + len(sw) > target and buf:
                    pieces.append(" ".join(buf))
                    buf = buf[-overlap:] if overlap else []
                buf.extend(sw)
            if buf:
                pieces.append(" ".join(buf))
            continue

        if len(current) + len(para_words) > target and current:
            pieces.append(" ".join(current))
            current = current[-overlap:] if overlap else []

        current.extend(para_words)

    if current:
        pieces.append(" ".join(current))

    return [p for p in pieces if p.strip()]


def page_for_offset(offset: int, page_offsets: list[int]) -> int:
    page = 1
    for i, start in enumerate(page_offsets):
        if offset >= start:
            page = i + 1
        else:
            break
    return page


def derive_title(text: str, doc_id: str, doc_type: str) -> str:
    if doc_type == "510k":
        m = re.search(
            r"(?:trade name|device name|proprietary name)\s*[:\-]?\s*\n?\s*"
            r"([^\n]{3,90})",
            text, re.IGNORECASE)
        if m:
            title = m.group(1).strip(" :.-")
            if 3 < len(title) < 90:
                return title
        return f"510(k) Summary {doc_id}"
    return doc_id.replace("-", " ").replace("_", " ").title()


def chunk_document(path: Path, doc_type: str) -> list[Chunk]:
    pages = extract_pages(path)
    if not pages:
        return []

    page_offsets: list[int] = []
    parts: list[str] = []
    cursor = 0
    for raw_page in pages:
        cleaned = clean_preserving_lines(raw_page)
        page_offsets.append(cursor)
        parts.append(cleaned)
        cursor += len(cleaned) + 2

    full_text = "\n\n".join(parts)
    if len(full_text.split()) < MIN_WORDS:
        return []

    doc_id = path.stem
    doc_title = derive_title(full_text, doc_id, doc_type)
    sections = find_sections(full_text, doc_type)

    chunks: list[Chunk] = []
    idx = 0

    for i, (start, heading) in enumerate(sections):
        end = sections[i + 1][0] if i + 1 < len(sections) else len(full_text)
        body = full_text[start:end].strip()

        if len(body.split()) < MIN_WORDS:
            continue

        # Track where each piece sits, so page numbers are per-chunk
        search_from = start
        for piece in split_long_text(body, TARGET_WORDS, MAX_WORDS, OVERLAP_WORDS):
            wc = len(piece.split())
            if wc < MIN_WORDS:
                continue

            probe = " ".join(piece.split()[:6])
            loc = full_text.find(probe, search_from, end + 200)
            piece_start = loc if loc != -1 else search_from
            piece_end = piece_start + len(piece)
            search_from = max(search_from, piece_start + 1)

            chunks.append(Chunk(
                chunk_id=f"{doc_id}_c{idx:03d}",
                text=flatten(piece),
                doc_id=doc_id,
                doc_type=doc_type,
                doc_title=doc_title,
                section=heading,
                page_start=page_for_offset(piece_start, page_offsets),
                page_end=page_for_offset(min(piece_end, len(full_text) - 1),
                                         page_offsets),
                chunk_index=idx,
                word_count=wc,
                source_file=str(path),
            ))
            idx += 1

    return chunks


# ---------------------------------------------------------------------------

def main() -> None:
    jobs: list[tuple[Path, str]] = []
    for folder, doc_type in (("510k", "510k"), ("guidance", "guidance")):
        d = RAW / folder
        if d.exists():
            jobs.extend((p, doc_type) for p in sorted(d.glob("*.pdf")))

    if not jobs:
        print(f"No PDFs found under {RAW.resolve()}")
        return

    print(f"Ingesting {len(jobs)} PDFs...\n")

    all_chunks: list[Chunk] = []
    stats: list[dict] = []
    skipped: list[str] = []

    for path, doc_type in jobs:
        try:
            chunks = chunk_document(path, doc_type)
        except Exception as e:
            print(f"  ERROR {path.name}: {e}")
            skipped.append(path.name)
            continue

        if not chunks:
            skipped.append(path.name)
            continue

        all_chunks.extend(chunks)
        stats.append({
            "file": path.name,
            "doc_type": doc_type,
            "chunks": len(chunks),
            "words": sum(c.word_count for c in chunks),
            "sections": len({c.section for c in chunks}),
        })

    OUT_CHUNKS.write_text(
        json.dumps([asdict(c) for c in all_chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with OUT_STATS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "doc_type", "chunks",
                                          "words", "sections"])
        w.writeheader()
        w.writerows(stats)

    by_type: dict[str, list[Chunk]] = {}
    for c in all_chunks:
        by_type.setdefault(c.doc_type, []).append(c)

    print("=" * 58)
    print("INGESTION COMPLETE")
    print("=" * 58)
    for dt, cs in sorted(by_type.items()):
        docs = len({c.doc_id for c in cs})
        avg = sum(c.word_count for c in cs) / len(cs)
        print(f"  {dt:<10} {docs:>4} docs  {len(cs):>5} chunks  avg {avg:.0f} words")
    print(f"\n  TOTAL      {len({c.doc_id for c in all_chunks}):>4} docs  "
          f"{len(all_chunks):>5} chunks")

    # --- Heading health: the thing that was broken -------------------------
    print("\n" + "=" * 58)
    print("HEADING DETECTION HEALTH")
    print("=" * 58)
    for dt, cs in sorted(by_type.items()):
        generic = sum(1 for c in cs if c.section == "(preamble)")
        pct = 100 * generic / len(cs)
        print(f"  {dt:<10} {pct:5.1f}% of chunks have no real heading "
              f"({generic}/{len(cs)})")
        counts: dict[str, int] = {}
        for c in cs:
            counts[c.section] = counts.get(c.section, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
        print(f"    most common sections:")
        for name, n in top:
            print(f"      {n:>4}  {name[:60]}")
    print("\n  Target: under ~25% preamble. Above that, the regexes need work.")

    pages = [c.page_start for c in all_chunks]
    print(f"\n  Page numbers: min {min(pages)}, max {max(pages)}, "
          f"{len(set(pages))} distinct  (all 1 = still broken)")

    if skipped:
        print(f"\n  Skipped {len(skipped)} files (no extractable text)")

    print(f"\n  Written: {OUT_CHUNKS}")

    print("\n" + "=" * 58)
    print("SPOT CHECK")
    print("=" * 58)
    import random
    random.seed(1)
    pool = [c for c in all_chunks if c.section != "(preamble)"] or all_chunks
    for c in random.sample(pool, min(4, len(pool))):
        print(f"\n[{c.chunk_id}]  {c.doc_type}  p.{c.page_start}-{c.page_end}")
        print(f"  section: {c.section}")
        print(f"  {c.text[:240]}...")


if __name__ == "__main__":
    main()
