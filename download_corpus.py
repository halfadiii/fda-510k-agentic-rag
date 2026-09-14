"""
Day 0/1: Download the FDA corpus.

Two sources, two different methods:

  1. 510(k) Summary PDFs (~120 docs) -- found via the openFDA API, then the PDF
     is fetched from a predictable URL pattern.
  2. FDA Guidance PDFs (~30 docs) -- hand-listed below, because there's no clean
     API for them and you only need a few good ones.

Run:  python download_corpus.py
Out:  data/raw/510k/*.pdf, data/raw/guidance/*.pdf, data/manifest.json
"""

import json
import time
from pathlib import Path

import requests

RAW = Path("data/raw")
(RAW / "510k").mkdir(parents=True, exist_ok=True)
(RAW / "guidance").mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "student-rag-project/1.0"}

# ---------------------------------------------------------------------------
# PART 1: 510(k) summaries
# ---------------------------------------------------------------------------
#
# Pick ONE device area and stay in it. A focused corpus makes your eval set
# coherent and your demo answerable. Broad + shallow is the classic mistake.
#
# Good choices given your Nissha background (wound care / medical consumables):
#   "wound dressing", "surgical drape", "electrode", "catheter"
# Or lean AI/ML, which is more interesting to talk about in interviews:
#   "computer assisted detection", "radiological software"

SEARCH_TERM = "wound dressing"
TARGET_COUNT = 120


def find_510k_numbers(term: str, limit: int = 600) -> list[dict]:
    """Query openFDA for cleared devices matching a search term."""
    url = "https://api.fda.gov/device/510k.json"
    params = {
        "search": f'device_name:"{term}" AND decision_code:"SESE" '
                  f'AND decision_date:[1995-01-01 TO 2026-12-31]',
        "limit": limit,
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    return [
        {
            "k_number": item["k_number"],
            "device_name": item.get("device_name", ""),
            "applicant": item.get("applicant", ""),
            "decision_date": item.get("decision_date", ""),
        }
        for item in results
    ]


def pdf_url_for(k_number: str) -> str:
    """
    Summary PDFs live at a predictable path keyed by the year in the K-number.
      K112345 -> .../cdrh_docs/pdf11/K112345.pdf
      K233456 -> .../cdrh_docs/pdf23/K233456.pdf
    Pre-2000 numbers use the un-suffixed 'pdf' folder.
    """
    yy = k_number[1:3]
    folder = "pdf" if yy >= "76" and yy <= "99" else f"pdf{yy}"
    return f"https://www.accessdata.fda.gov/cdrh_docs/{folder}/{k_number}.pdf"


def download_510k_summaries() -> list[dict]:
    candidates = find_510k_numbers(SEARCH_TERM)
    print(f"openFDA returned {len(candidates)} candidate clearances")

    saved = []
    for item in candidates:
        if len(saved) >= TARGET_COUNT:
            break

        k = item["k_number"]
        dest = RAW / "510k" / f"{k}.pdf"
        if dest.exists():
            saved.append({**item, "path": str(dest)})
            continue

        url = pdf_url_for(k)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            print(f"  {k}: request failed ({e})")
            continue

        # Not every clearance has a public summary -- some filed a "statement"
        # instead. Those 404, which is expected. Just skip them.
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            print(f"  {k}: no summary PDF (status {r.status_code})")
            continue

        dest.write_bytes(r.content)
        saved.append({**item, "path": str(dest), "source_url": url})
        safe_name = item["device_name"][:45].encode("ascii", "replace").decode("ascii")
        print(f"  [{len(saved):>3}/{TARGET_COUNT}] {k}  {safe_name}")

        time.sleep(0.5)  # be polite to a government server

    return saved


# ---------------------------------------------------------------------------
# PART 2: Guidance documents
# ---------------------------------------------------------------------------
#
# These are the prose-heavy, section-numbered documents that make good
# questions. There's no bulk API, so: go to
#   https://www.fda.gov/medical-devices/device-advice-comprehensive-regulatory-assistance/guidance-documents-medical-devices-and-radiation-emitting-products
# filter to Final guidance, and right-click-copy the PDF links for ~30 that
# look relevant. Paste them below.
#
# Start with these themes: quality systems, software, cybersecurity, labeling,
# biocompatibility, sterilization, human factors, AI/ML-enabled devices.

GUIDANCE_URLS = [
    # Example (the one from your screenshot):
    # "https://www.fda.gov/media/71794/download",
    #
    # ---> Paste ~30 URLs here. This is 20 minutes of clicking. <---
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-international-standard-iso-10993-1-biological-evaluation-medical-devices-part-1-evaluation-and",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/biocompatibility-testing-medical-devices-standards-specific-information-accreditation-scheme",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/submission-and-review-sterility-information-premarket-notification-510k-submissions-devices-labeled",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/sterilized-convenience-kits-clinical-and-surgical-use",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/510k-program-evaluating-substantial-equivalence-premarket-notifications-510k",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/benefit-risk-factors-consider-when-determining-substantial-equivalence-premarket-notifications-510k",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/special-510k-program",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/abbreviated-510k-program",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/special-510k-program",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/abbreviated-510k-program",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/refuse-accept-policy-510ks",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/requests-feedback-and-meetings-medical-device-submissions-q-submission-program",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/medical-devices-containing-materials-derived-animal-sources-except-in-vitro-diagnostic-devices",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/labeling-regulatory-requirements-medical-devices-fda-89-4203",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/cpg-sec-300300-ineffective-devices-502fi-labeling-requirements",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/recognition-and-withdrawal-voluntary-consensus-standards",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/reprocessing-medical-devices-health-care-settings-validation-methods-and-labeling",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/deciding-when-submit-510k-software-change-existing-device",
]


def download_guidance() -> list[dict]:
    saved = []
    for i, url in enumerate(GUIDANCE_URLS, 1):
        dest = RAW / "guidance" / f"guidance_{i:03d}.pdf"
        if dest.exists():
            saved.append({"path": str(dest), "source_url": url})
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            if r.status_code == 200 and r.content.startswith(b"%PDF"):
                dest.write_bytes(r.content)
                saved.append({"path": str(dest), "source_url": url})
                print(f"  [{i:>3}] saved {dest.name}  ({len(r.content)//1024} KB)")
            else:
                print(f"  [{i:>3}] FAILED {url} (status {r.status_code})")
        except requests.RequestException as e:
            print(f"  [{i:>3}] FAILED {url} ({e})")
        time.sleep(0.5)
    return saved


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== 510(k) summaries ===")
    summaries = download_510k_summaries()

    print("\n=== Guidance documents ===")
    if not GUIDANCE_URLS:
        print("  (none listed yet -- add URLs to GUIDANCE_URLS and re-run)")
    guidance = download_guidance()

    manifest = {
        "search_term": SEARCH_TERM,
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {"510k": len(summaries), "guidance": len(guidance)},
        "510k": summaries,
        "guidance": guidance,
    }
    Path("data/manifest.json").write_text(json.dumps(manifest, indent=2))

    total = len(summaries) + len(guidance)
    print(f"\nDone. {total} PDFs in data/raw/")
    print("Manifest written to data/manifest.json -- COMMIT THIS FILE.")
    print("It's your proof of exactly which corpus your eval scores describe.")
