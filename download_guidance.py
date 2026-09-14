"""
Download FDA guidance PDFs from their landing-page URLs.

Why a separate script:
    The URLs you collected point at HTML pages ABOUT each guidance, not at the
    PDFs. The actual file sits behind a download button, at an address like
    fda.gov/media/71794/download. So for each page we:
        1. fetch the HTML
        2. find the PDF link inside it
        3. download that

Run:  python download_guidance.py
Out:  data/raw/guidance/*.pdf  +  data/guidance_manifest.json
"""

import json
import re
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path("data/raw/guidance")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "student-rag-project/1.0"}

# Duplicates removed (special-510k and abbreviated-510k appeared twice).
LANDING_URLS = [
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-international-standard-iso-10993-1-biological-evaluation-medical-devices-part-1-evaluation-and",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/biocompatibility-testing-medical-devices-standards-specific-information-accreditation-scheme",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/submission-and-review-sterility-information-premarket-notification-510k-submissions-devices-labeled",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/sterilized-convenience-kits-clinical-and-surgical-use",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/510k-program-evaluating-substantial-equivalence-premarket-notifications-510k",
    "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/benefit-risk-factors-consider-when-determining-substantial-equivalence-premarket-notifications-510k",
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


def find_pdf_links(html: str) -> list[str]:
    """
    Find candidate PDF links on an FDA guidance landing page.

    FDA pages vary in markup, so we try several patterns and return everything
    found, most-likely first. Some pages link several PDFs (the guidance plus
    attachments); we take the first that actually downloads as a PDF.
    """
    candidates: list[str] = []

    patterns = [
        r'href="(/media/\d+/download[^"]*)"',
        r'href="(https://www\.fda\.gov/media/\d+/download[^"]*)"',
        r'href="([^"]*\.pdf)"',
        r'href="(/media/\d+/[^"]*)"',
    ]
    for pat in patterns:
        for m in re.findall(pat, html, flags=re.IGNORECASE):
            url = m if m.startswith("http") else "https://www.fda.gov" + m
            if url not in candidates:
                candidates.append(url)

    return candidates


def fetch_one(landing: str, index: int) -> dict | None:
    name = landing.rstrip("/").split("/")[-1][:70]
    dest = OUT / f"{name}.pdf"

    if dest.exists():
        print(f"  [{index:>2}] SKIP (already have)  {name[:50]}")
        return {"path": str(dest), "landing_url": landing, "pdf_url": None}

    try:
        r = requests.get(landing, headers=HEADERS, timeout=45)
    except requests.RequestException as e:
        print(f"  [{index:>2}] NETWORK FAIL  {name[:45]}  ({e})")
        return None

    if r.status_code != 200:
        print(f"  [{index:>2}] PAGE {r.status_code}  {name[:50]}")
        return None

    candidates = find_pdf_links(r.text)
    if not candidates:
        print(f"  [{index:>2}] NO PDF LINK FOUND  {name[:45]}")
        print(f"       open manually: {landing}")
        return None

    # Try each candidate until one is a real PDF
    for pdf_url in candidates[:5]:
        try:
            p = requests.get(pdf_url, headers=HEADERS, timeout=120)
        except requests.RequestException:
            continue

        if p.status_code == 200 and p.content.startswith(b"%PDF"):
            dest.write_bytes(p.content)
            kb = len(p.content) // 1024
            print(f"  [{index:>2}] OK  {kb:>6} KB  {name[:48]}")
            return {"path": str(dest), "landing_url": landing,
                    "pdf_url": pdf_url, "size_kb": kb}

    print(f"  [{index:>2}] NO WORKING PDF  {name[:45]}")
    print(f"       tried {len(candidates[:5])} link(s); open manually: {landing}")
    return None


def main() -> None:
    print(f"Fetching {len(LANDING_URLS)} guidance documents...\n")

    saved, failed = [], []
    for i, url in enumerate(LANDING_URLS, 1):
        result = fetch_one(url, i)
        if result:
            saved.append(result)
        else:
            failed.append(url)
        time.sleep(1.0)

    Path("data/guidance_manifest.json").write_text(
        json.dumps(saved, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print(f"Downloaded {len(saved)} / {len(LANDING_URLS)}")
    print(f"Saved to: {OUT.resolve()}")

    if failed:
        print(f"\n{len(failed)} failed -- grab these by hand:")
        for u in failed:
            print(f"  {u}")
        print("\n  Open the page, right-click the 'Download the Final Guidance")
        print("  Document' button, Save Link As, into data/raw/guidance/")


if __name__ == "__main__":
    main()
