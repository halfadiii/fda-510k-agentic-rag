"""
Day 4: Build the exam.

WHY THIS IS THE MOST IMPORTANT DAY
    Anyone can build a RAG demo that looks good on the questions its author
    tried. This file builds ~50 questions where YOU know the right answer and
    the exact page it lives on. Day 5 scores the system against them.

THREE STEPS
    python make_eval_set.py generate   # model drafts ~66 candidates  (~$0.05)
    python make_eval_set.py review     # YOU keep / edit / drop each one
    python make_eval_set.py build      # merge with hand-written ones -> eval_set.json

WHAT KINDS OF QUESTIONS
    single_510k      one fact from one device summary
    single_guidance  one fact or definition from one guidance document
    cross_doc        needs a device summary AND a guidance document together
    unanswerable     the system SHOULD refuse (written by hand, below)
    known_gap        info exists in a PDF but was lost in extraction (tables)

WHY A MODEL DRAFTS AND A HUMAN DECIDES
    Writing 50 questions from scratch takes a day. Letting a model write them
    and trusting it blindly makes the exam worthless: it will write questions
    the passage doesn't actually answer. So the model drafts and you approve.
    The review step is the part that makes the scores mean something.

GOLD LABELS ARE PAGES, NOT JUST CHUNKS
    Each question records the doc_id + page range its answer lives on. Day 5
    counts retrieval as a hit if it returns ANY chunk from that document that
    overlaps those pages. Matching one exact chunk_id would be too strict,
    because chunks overlap by 50 words, and the same fact can sit in two
    neighbouring chunks.
"""

import json
import os
import random
import re
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

CHUNKS_FILE = Path("data/chunks.json")
EVAL_DIR = Path("data/eval")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATES = EVAL_DIR / "candidates.json"
REVIEWED = EVAL_DIR / "reviewed.json"
EVAL_SET = EVAL_DIR / "eval_set.json"

N_510K = 32        # generate extra: you'll drop roughly a third in review
N_GUIDANCE = 24
N_CROSS = 10
SEED = 7           # fixed, so the same chunks get sampled every run

MODEL = "deepseek-chat"
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com")

# 510(k) sections that are just labels (names, addresses, codes).
# Questions from these are trivia, not tests of understanding.
SKIP_SECTIONS = {
    "device name", "name of device", "trade name", "proprietary name",
    "common name", "submitter", "submitted by", "applicant",
    "contact person", "date prepared", "classification name",
    "product code", "regulation number",
}

# 510(k) sections with substance that guidance documents also talk about.
# These make good cross-document questions.
CROSS_SECTIONS = {
    "biocompatibility", "sterilization", "sterility", "shelf life",
    "packaging", "device description", "description of the device",
    "technological characteristics", "comparison to predicate",
    "performance data", "performance testing", "non-clinical testing",
    "bench testing", "substantial equivalence",
}


# ---------------------------------------------------------------------------
# Hand-written questions the system SHOULD refuse.
# Before trusting these, run `python search.py "<question>"` on each and
# confirm by eye that the answer really isn't in the corpus.
# ---------------------------------------------------------------------------

HANDWRITTEN = [
    {"id": "hw-01", "category": "unanswerable",
     "question": "Is wool a good wound dressing material?",
     "notes": "absent topic: wool appears nowhere in the corpus"},
    {"id": "hw-02", "category": "unanswerable",
     "question": "What is the retail price of the Algidex Ag silver alginate wound dressing?",
     "notes": "absent field: device is in the corpus, prices never are"},
    {"id": "hw-03", "category": "unanswerable",
     "question": "What does the EU Medical Device Regulation require for silver-containing wound dressings?",
     "notes": "wrong jurisdiction: corpus is FDA only"},
    {"id": "hw-04", "category": "unanswerable",
     "question": "What materials is the Medtronic Micra leadless pacemaker made of?",
     "notes": "absent device: not a wound dressing"},
    {"id": "hw-05", "category": "unanswerable",
     "question": "Why did FDA reject the 510(k) submission for the Algidex Ag wound dressing?",
     "notes": "false premise: K173072 was cleared, not rejected"},
    {"id": "hw-06", "category": "unanswerable",
     "question": "Which wound dressing in the FDA database heals diabetic foot ulcers the fastest?",
     "notes": "summaries don't rank clinical efficacy; verify with search.py"},
    {"id": "hw-07", "category": "unanswerable",
     "question": "What did FDA's 2027 revision of the ISO 10993-1 biocompatibility guidance change?",
     "notes": "future event: no such document in the corpus"},
    {"id": "hw-08", "category": "unanswerable",
     "question": "What was the 510(k) number of the first silver wound dressing ever cleared by FDA?",
     "notes": "corpus starts in 1995 and excludes scanned filings"},
    {"id": "hw-09", "category": "known_gap",
     "question": "According to the ISO 10993-1 guidance, which biological endpoints should be evaluated for a device with breached-surface contact?",
     "notes": "in the PDF, but the endpoint matrix is a table that extraction "
              "flattened into bare X/O marks. Scored separately from true "
              "unanswerables."},
]


# ---------------------------------------------------------------------------
# Prompts. Placeholders look like <<TEXT>> so JSON braces need no escaping.
# ---------------------------------------------------------------------------

SINGLE_PROMPT = """You are writing one exam question to test a search system over FDA documents.

PASSAGE (from <<LABEL>>, page <<PAGE>>, section "<<SECTION>>"):
\"\"\"<<TEXT>>\"\"\"

Write ONE question that:
- can be answered completely from this passage alone
- is self-contained: name the device (trade name or K-number) or the guidance topic.
  Never say "this device", "the passage", or "the document"
- asks for a specific fact: a material, a method, a test, a definition, a condition, a number
- is phrased the way a regulatory analyst would ask it, in YOUR OWN WORDS.
  Do not copy distinctive phrases from the passage

Then give the answer in one or two sentences, and 1-3 KEY FACTS: short phrases
that any correct answer must contain.

Reply in JSON only:
{"question": "...", "answer": "...", "key_facts": ["...", "..."]}

If the passage has no clear, specific fact worth asking about, reply {"skip": true}."""

CROSS_PROMPT = """You are writing one exam question that needs TWO documents to answer.

PASSAGE A (510(k) summary for <<DEVICE>>, K-number <<KNUM>>, page <<PAGE_A>>):
\"\"\"<<TEXT_A>>\"\"\"

PASSAGE B (FDA guidance "<<GUIDE>>", page <<PAGE_B>>):
\"\"\"<<TEXT_B>>\"\"\"

Write ONE question that genuinely needs a fact from Passage A AND a fact from
Passage B. For example: what a device did (from A), and what the guidance says
about that kind of thing (from B).

Rules:
- name the device by trade name or K-number
- phrase it in your own words, no copied phrases
- the answer must be fully supported by the two passages together

Reply in JSON only:
{"question": "...", "answer": "...", "key_facts": ["...", "..."]}

If the passages have no genuine connection, reply {"skip": true}. Do not force one."""


def fill(template: str, **values: str) -> str:
    for key, val in values.items():
        template = template.replace(f"<<{key}>>", val)
    return template


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOKENS = {"in": 0, "out": 0}


def ask_json(prompt: str) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        print(f"    API error: {e}")
        return None
    TOKENS["in"] += resp.usage.prompt_tokens
    TOKENS["out"] += resp.usage.completion_tokens
    try:
        return json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        return None


def usable(c: dict) -> bool:
    """Filter out chunks that make bad exam material."""
    if c["section"] == "(preamble)":
        return False
    if c["section"].strip().lower().rstrip(":") in SKIP_SECTIONS:
        return False
    if c["word_count"] < 80:
        return False
    words = c["text"].split()
    tiny = sum(1 for w in words if len(w) <= 2)
    if tiny / len(words) > 0.35:        # table debris like "X O X X O"
        return False
    return True


def pick_spread(pool: list[dict], n: int, per_doc: int,
                rng: random.Random) -> list[dict]:
    """Random sample, capped per document so one PDF can't dominate."""
    pool = pool[:]
    rng.shuffle(pool)
    taken, count = [], {}
    for c in pool:
        if count.get(c["doc_id"], 0) >= per_doc:
            continue
        taken.append(c)
        count[c["doc_id"]] = count.get(c["doc_id"], 0) + 1
        if len(taken) >= n:
            break
    return taken


def gold_of(c) -> dict:
    get = c.get if isinstance(c, dict) else lambda k: getattr(c, k)
    return {"doc_id": get("doc_id"), "chunk_id": get("chunk_id"),
            "page_start": get("page_start"), "page_end": get("page_end")}


def doc_label(c: dict) -> str:
    if c["doc_type"] == "510k":
        return f"510(k) summary {c['doc_id']} for {c['doc_title']}"
    return f"FDA guidance \"{c['doc_title']}\""


# ---------------------------------------------------------------------------
# Step 1: generate
# ---------------------------------------------------------------------------

def single(c: dict, category: str, idx: int) -> dict | None:
    data = ask_json(fill(SINGLE_PROMPT, LABEL=doc_label(c),
                         PAGE=str(c["page_start"]), SECTION=c["section"],
                         TEXT=c["text"]))
    if not data or data.get("skip") or not data.get("question"):
        return None
    return {
        "id": f"gen-{category}-{idx:03d}",
        "category": category,
        "question": data["question"].strip(),
        "answer": data.get("answer", "").strip(),
        "key_facts": data.get("key_facts", []),
        "gold": [gold_of(c)],
        "source_text": [c["text"]],
    }


def generate() -> None:
    if CANDIDATES.exists() and "--force" not in sys.argv:
        print(f"{CANDIDATES} already exists. Add --force to regenerate "
              f"(this throws away the current candidates).")
        return

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    good = [c for c in chunks if usable(c)]
    print(f"{len(good)} of {len(chunks)} chunks are usable exam material\n")

    out: list[dict] = []

    print(f"--- single-document questions from 510(k) summaries ---")
    for i, c in enumerate(pick_spread(
            [c for c in good if c["doc_type"] == "510k"], N_510K, 1, rng), 1):
        q = single(c, "single_510k", i)
        print(f"  [{i:>2}] {'ok  ' if q else 'skip'} {c['chunk_id']}")
        if q:
            out.append(q)

    print(f"\n--- single-document questions from guidance ---")
    for i, c in enumerate(pick_spread(
            [c for c in good if c["doc_type"] == "guidance"], N_GUIDANCE, 2, rng), 1):
        q = single(c, "single_guidance", i)
        print(f"  [{i:>2}] {'ok  ' if q else 'skip'} {c['chunk_id']}")
        if q:
            out.append(q)

    print(f"\n--- cross-document questions ---")
    print("  (loading the retriever to find a matching guidance passage)")
    from search import Retriever
    r = Retriever()
    cross_pool = [c for c in good if c["doc_type"] == "510k"
                  and c["section"].strip().lower().rstrip(":") in CROSS_SECTIONS]
    kept = 0
    for i, a in enumerate(pick_spread(cross_pool, N_CROSS * 2, 1, rng), 1):
        if kept >= N_CROSS:
            break
        probe = " ".join(a["text"].split()[:60])
        hits = r.search(probe, k=1, mode="dense", doc_type="guidance")
        if not hits:
            continue
        b = hits[0]
        data = ask_json(fill(CROSS_PROMPT,
                             DEVICE=a["doc_title"], KNUM=a["doc_id"],
                             PAGE_A=str(a["page_start"]), TEXT_A=a["text"],
                             GUIDE=b.doc_title, PAGE_B=str(b.page_start),
                             TEXT_B=b.text))
        if not data or data.get("skip") or not data.get("question"):
            print(f"  [{i:>2}] skip {a['chunk_id']} + {b.chunk_id}")
            continue
        kept += 1
        out.append({
            "id": f"gen-cross_doc-{kept:03d}",
            "category": "cross_doc",
            "question": data["question"].strip(),
            "answer": data.get("answer", "").strip(),
            "key_facts": data.get("key_facts", []),
            "gold": [gold_of(a), gold_of(b)],
            "source_text": [a["text"], b.text],
        })
        print(f"  [{i:>2}] ok   {a['chunk_id']} + {b.chunk_id}")
    r.close()

    CANDIDATES.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    cost = (TOKENS["in"] * 0.27 + TOKENS["out"] * 1.10) / 1_000_000
    print(f"\n{len(out)} candidates written to {CANDIDATES}")
    print(f"{TOKENS['in']}+{TOKENS['out']} tokens, about ${cost:.3f}")
    print("\nNext:  python make_eval_set.py review")


# ---------------------------------------------------------------------------
# Step 2: review  (this is YOUR job, not the model's)
# ---------------------------------------------------------------------------

CHECKLIST = """  Before keeping, check:
    1. Is the answer ACTUALLY in the passage? (read it, don't trust the model)
    2. Would a real regulatory analyst ask this?
    3. Can someone understand the question without seeing the passage?
    4. Does the question just copy the passage's wording? (makes it too easy)"""


def wrap(text: str, width: int = 90, indent: str = "    ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def review() -> None:
    if not CANDIDATES.exists():
        print("No candidates yet. Run:  python make_eval_set.py generate")
        return

    candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    reviewed = (json.loads(REVIEWED.read_text(encoding="utf-8"))
                if REVIEWED.exists() else {})

    todo = [c for c in candidates if c["id"] not in reviewed]
    print(f"{len(reviewed)} already reviewed, {len(todo)} to go.")
    print("Your progress saves after every decision; quit any time and resume.\n")

    for n, c in enumerate(todo, 1):
        print("\n" + "=" * 90)
        print(f"[{n}/{len(todo)}]  {c['id']}   ({c['category']})")
        print("=" * 90)
        for i, (g, txt) in enumerate(zip(c["gold"], c["source_text"]), 1):
            print(f"\n  PASSAGE {i}: {g['doc_id']}  p.{g['page_start']}-{g['page_end']}")
            print(wrap(txt))
        print(f"\n  QUESTION:  {c['question']}")
        print(f"  ANSWER:    {c['answer']}")
        print(f"  KEY FACTS: {c['key_facts']}")
        print("\n" + CHECKLIST)

        while True:
            choice = input("\n  [k]eep  edit [q]uestion  edit [a]nswer  "
                           "edit [f]acts  [d]rop  [x] save & exit > ").strip().lower()
            if choice == "q":
                c["question"] = input("  new question: ").strip() or c["question"]
            elif choice == "a":
                c["answer"] = input("  new answer: ").strip() or c["answer"]
            elif choice == "f":
                raw = input("  key facts, comma-separated: ").strip()
                if raw:
                    c["key_facts"] = [p.strip() for p in raw.split(",") if p.strip()]
            elif choice in ("k", "d"):
                c["decision"] = "keep" if choice == "k" else "drop"
                if choice == "d":
                    c["drop_reason"] = input("  why? (optional) ").strip()
                reviewed[c["id"]] = c
                REVIEWED.write_text(json.dumps(reviewed, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
                break
            elif choice == "x":
                print(f"\nSaved. {len(reviewed)} reviewed so far.")
                return
            else:
                continue
            print(f"\n  QUESTION:  {c['question']}")
            print(f"  ANSWER:    {c['answer']}")
            print(f"  KEY FACTS: {c['key_facts']}")

    kept = sum(1 for r in reviewed.values() if r["decision"] == "keep")
    print(f"\nAll reviewed. Kept {kept} of {len(reviewed)}.")
    print("Next:  python make_eval_set.py build")


# ---------------------------------------------------------------------------
# Step 3: build + sanity checks
# ---------------------------------------------------------------------------

STOP = set("""the a an of to in for and or is are was were be been by on with
what which how does do did that this from as at it its under when should
can could would their there has have""".split())


def content_words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-\.]*", s.lower())
            if w not in STOP and len(w) > 2}


def overlap(question: str, passage: str) -> float:
    """Share of the question's meaningful words that also appear in the passage."""
    qw = content_words(question)
    return len(qw & content_words(passage)) / len(qw) if qw else 0.0


def build() -> None:
    if not REVIEWED.exists():
        print("Nothing reviewed yet. Run:  python make_eval_set.py review")
        return

    chunks = {c["chunk_id"]: c for c in
              json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))}
    reviewed = json.loads(REVIEWED.read_text(encoding="utf-8"))

    items: list[dict] = []
    for r in reviewed.values():
        if r["decision"] != "keep":
            continue
        items.append({
            "id": r["id"],
            "category": r["category"],
            "question": r["question"],
            "expected_behavior": "answer",
            "reference_answer": r["answer"],
            "key_facts": r["key_facts"],
            "gold": r["gold"],
            "notes": "",
        })
    for h in HANDWRITTEN:
        items.append({
            "id": h["id"],
            "category": h["category"],
            "question": h["question"],
            "expected_behavior": "refuse",
            "reference_answer": "",
            "key_facts": [],
            "gold": [],
            "notes": h["notes"],
        })

    # --- checks -----------------------------------------------------------
    problems = []
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    for i in items:
        if i["expected_behavior"] == "answer":
            if not i["gold"]:
                problems.append(f"{i['id']}: answerable but no gold page")
            if not i["key_facts"]:
                problems.append(f"{i['id']}: answerable but no key facts")
            for g in i["gold"]:
                if g["chunk_id"] not in chunks:
                    problems.append(f"{i['id']}: gold chunk {g['chunk_id']} "
                                    f"not in chunks.json (re-ingested?)")

    EVAL_SET.write_text(json.dumps(items, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    # --- report -----------------------------------------------------------
    print("=" * 60)
    print(f"EVAL SET: {len(items)} questions  ->  {EVAL_SET}")
    print("=" * 60)
    by_cat: dict[str, int] = {}
    for i in items:
        by_cat[i["category"]] = by_cat.get(i["category"], 0) + 1
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<16} {n:>3}")

    refuse = sum(1 for i in items if i["category"] == "unanswerable")
    print(f"\n  should-refuse share: {100 * refuse / len(items):.0f}%  "
          f"(aim for 15-20%)")
    if len(items) < 45:
        print(f"  WARNING: only {len(items)} questions. Aim for 45+; "
              f"re-run generate with --force and higher counts if needed.")

    # Leakage check: questions that copy the passage are unfairly easy for BM25
    print("\n  WORDING LEAK CHECK (question words found in its own passage)")
    leaky = []
    for i in items:
        if not i["gold"]:
            continue
        text = " ".join(chunks[g["chunk_id"]]["text"]
                        for g in i["gold"] if g["chunk_id"] in chunks)
        ov = overlap(i["question"], text)
        if ov > 0.8:
            leaky.append((i["id"], ov, i["question"]))
    if leaky:
        print(f"  {len(leaky)} question(s) reuse over 80% of their passage's words.")
        print("  Keyword search will find these trivially. Consider rewording:")
        for qid, ov, q in leaky:
            print(f"    {qid}  {ov:.0%}  {q[:70]}")
    else:
        print("  none above 80%, good")

    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("\n  No structural problems. Commit data/eval/ -- it's your exam.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"generate": generate, "review": review, "build": build}.get(
        step, lambda: print(__doc__))()
