# Agentic RAG over FDA filings

**Ask the FDA's own documents a regulatory question. It answers with the page it
came from, or it refuses.**

Regulatory questions are the kind where a confident wrong answer is worse than no
answer. Most retrieval-augmented generation is: search, paste whatever came back
into a prompt, trust the model. This project is about everything around the model
instead: deciding when *not* to answer, and making a made-up citation impossible
to serve.

**Live demo:** [adityaaryan.in/demo/rag](https://adityaaryan.in/demo/rag), which
shows every step for a question, including the drafts it throws away.

---

## What it knows

| | |
| --- | --- |
| FDA 510(k) summaries | **97** wound dressing clearances, found through the openFDA API |
| FDA guidance documents | **19**: biocompatibility, sterility, the 510(k) program, labeling, shelf life, software changes |
| Searchable chunks | **1,752**, each carrying its document, section heading and page |

One device area on purpose. A narrow corpus makes the questions coherent and a
wrong answer recognisable; broad and shallow is the easy mistake.

### The audit came first

Before indexing anything, `audit_corpus.py` measured how much real text each PDF
had. **42 of the 140 clearances downloaded were scanned paper**, photographs of
pages with nothing to extract. Indexed as-is they would have been empty chunks
that silently drag retrieval down, and nobody would find out until the answers
were bad for reasons nobody could explain. They were excluded instead, and that
is a real bias worth stating: most of them are older filings, so the corpus leans
recent.

---

## What happens to a question

```
question
  -> retrieve 5 chunks, two ways at once, fused ............. search.py
  -> RELEVANCE GATE: do these actually answer it? ........... one word from the model
  -> DRAFT an answer that cites [S1]..[S5]
  -> CITATION CHECK: is every tag a source it was given? ..... plain Python, no model
  -> failed? say what was wrong, draft once more; failed again? refuse
```

### Retrieval, two ways, because they fail in opposite directions

**By meaning:** `BAAI/bge-small-en-v1.5` embeddings, run locally, stored in Qdrant.
It finds "gamma irradiation" when you ask about sterilisation. It is bad at exact
identifiers.

**By keyword:** BM25. It finds "ISO 10993-1" and "K212521" exactly. It is blind to
synonyms.

The two lists are combined with **Reciprocal Rank Fusion**: every chunk scores
`1 / (60 + its rank)` in each list, summed. It uses ranks only, because a cosine
similarity and a BM25 score live on scales that cannot be added.

That it matters is visible in real results. Asking **"What is a predicate
device?"**, the passage holding the legal definition ranked **1st by meaning and
did not appear in the keyword top 30 at all**, while keyword search's first pick
was absent from the meaning top 30. The answer cited the meaning-only passages.
Either search alone would have handed the model a worse five.

### The three guards

**1. The relevance gate.** Search always returns something. Ask "Is wool a good
wound dressing material?" and it dutifully returns five passages about wound
dressings. So before drafting, the model is asked one question with a one-word
answer: do these passages help? For wool it said **NO**, and the system refused in
1.6 seconds without writing a word.

**2. The citation check.** The model must tag every factual sentence `[S1]` to
`[S5]`. Numbered tags rather than document names, because a model can typo or
invent a plausible-looking document ID, while a number can only be right or wrong.
Then plain code checks every tag points at a passage the model was actually given.
An answer citing `[S7]` out of five sources, or citing nothing, fails. **This
system cannot serve a citation to a source it did not read**, and that holds
whatever the model does, because the check is not a model.

**3. Retry once, then refuse.** A failed draft goes back with what was wrong. If
the second draft fails too, the answer is a refusal, not the better of two bad
drafts.

There is also a fourth way to end: the model can reply `INSUFFICIENT:` and say
what is missing, which becomes a refusal that names the gap.

---

## Measured

Six questions, 2026-09-14, through the live demo's pipeline (same prompts, same
model, `deepseek-chat` at temperature 0):

| Question | Outcome | Time | Tokens in + out | Cost |
| --- | --- | --- | --- | --- |
| How are silver dressings sterilized? | answered, citing all five | 3.4 s | 3,567 + 77 | $0.0011 |
| What is a predicate device? | answered | 2.9 s | 4,519 + 100 | $0.0013 |
| When should a manufacturer submit a new 510(k) for a software change? | answered | 3.2 s | 5,137 + 169 | $0.0016 |
| What does ISO 10993-1 require? | answered | 3.2 s | 3,391 + 102 | $0.0010 |
| What biocompatibility testing does FDA expect for a dressing that contacts an open wound? | declined: the passages don't say | 3.3 s | 3,544 + 87 | $0.0011 |
| Is wool a good wound dressing material? | refused at the gate | 1.6 s | 1,566 + 1 | $0.0004 |

Cost is the code's own estimate at DeepSeek V3 list prices ($0.27 in, $1.10 out,
per million tokens), counting every model call including the gate.

The biocompatibility one is the most interesting row. It is the question this
project was built around, and it declines, because four of the five passages
retrieved are **manufacturers describing tests on their own devices**, not FDA
saying what it expects. Declining is right with those passages. Retrieving better
ones is the fix, and `search.py` already supports filtering by document type,
which nothing uses yet.

---

## The exam

Anyone can build a RAG demo that looks good on the questions its author tried.
So before scoring anything, `make_eval_set.py` builds an exam where the right
answer, and the page it lives on, is known in advance.

A model drafts, and a person decides. The model wrote **66** candidate questions
from real passages; every one was reviewed by hand against its passage. **40 were
kept** (6 of them edited) and **26 dropped**. The drop reasons are a finding in
themselves: most were questions built on **comparison tables that PDF extraction
had mangled** (which column was the device and which the predicate could not be
recovered) or on **boilerplate cover letters** that say nothing about the device.

Add 9 questions written by hand that the system should refuse (wool, a retail
price, EU rules, a pacemaker, a false premise, a future guidance revision), and the
exam is **49 questions**:

| Category | Questions | What it tests |
| --- | --- | --- |
| One 510(k) summary | 22 | a single fact from one device |
| One guidance document | 17 | a fact or definition from guidance |
| Two documents together | 1 | a device fact plus what guidance says about it |
| Should refuse | 8 | the answer is not in the corpus |
| Known gap | 1 | the answer is in a PDF table that extraction flattened |

40 of the 49 have a known answer, spread across 36 documents, with the page
range recorded. Retrieval counts as a hit if it returns any chunk from that
document overlapping those pages, because chunks overlap and the same fact can sit
in two neighbours. 9 of 49 (18%) should be refused, inside the 15-20% target.

```bash
python make_eval_set.py generate   # model drafts candidates
python make_eval_set.py review     # a person keeps, edits or drops each one
python make_eval_set.py build      # merge with the hand-written ones -> data/eval/eval_set.json
```

## The score

Scored on 19 October 2026 with `score.py`, which runs every question through the
full system and keeps three results apart: was the gold page retrieved, did it
answer or refuse as it should, and did the answer hold the reviewed facts. A
second model call (DeepSeek, temperature 0) marks each answer against the
reference answer and its key facts. It never sees the passages.

| | Result |
| --- | --- |
| Answerable questions answered correctly | **34 of 40** |
| Questions it should refuse, refused | **9 of 9** |
| Answers that contradict the reference | **0** |
| Gold page in the top 5 | 35 of 40 |
| Answers holding every reviewed key fact | 31 of 40 |
| Cost of the whole run | about $0.03 |

The judge passed 35. Reading its four lenient passes by hand, one left out part
of what was asked (the sheet form of K231152's dressing), so the table says 34.

Every failure was a refusal or an incomplete answer, never a wrong one. Three
were search misses, where the gold page was not in the top 5. Two had the page
and the model still answered INSUFFICIENT, which is the gate's lean towards
caution showing from the other side. One answered only part of the question.

Search alone, gold page in the top 5 of the 40: meaning search **35**, keyword
search **34**, hybrid **35**. On this exam the hybrid ties meaning search rather
than beating it. Its case is exact identifiers such as K212521, which the exam
barely asks about.

```bash
python score.py              # all 49, about $0.03
python score.py --limit 3    # a quick check first
```

Results, every answer and every verdict, are in `data/eval/results.json`.

---

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

`data/chunks.json` is committed, so you can skip the 117 MB of PDFs and build the
indexes straight from it:

```bash
python embed.py                                        # vectors + keyword index, a few minutes on a CPU
python search.py "how are silver dressings sterilized?" # compare dense, BM25 and hybrid side by side
```

Answering needs a DeepSeek key. Put it in `.env` (it is gitignored):

```
DEEPSEEK_API_KEY=your-key
```

```bash
python answer.py "how are silver dressings sterilized?"
python answer.py            # runs four questions; the last one must refuse
```

To rebuild the corpus from scratch instead:

```bash
python download_corpus.py      # 510(k) summaries via openFDA
python download_guidance.py    # guidance PDFs, found through their landing pages
python audit_corpus.py         # which PDFs actually have text
python ingest.py               # heading-aware chunks with page numbers
python embed.py
```

Four of the guidance documents were added by hand and are not in either download
script. They are in `chunks.json`, so nothing is lost unless you re-ingest.

---

## Files

```
download_corpus.py     openFDA search -> 510(k) summary PDFs, and data/manifest.json
download_guidance.py   guidance landing pages -> the PDF behind each one
test.py                the probe used to choose a device area (clearance counts per term)
audit_corpus.py        text per page, per PDF: TEXT, SPARSE or SCANNED  -> data/audit.csv
ingest.py              PDFs -> chunks with section and page             -> data/chunks.json
embed.py               chunks -> Qdrant (dense) + BM25 (keyword)
search.py              dense, BM25, and hybrid with Reciprocal Rank Fusion
answer.py              gate, draft, citation check, retry, refuse
make_eval_set.py       the exam: generate, review by hand, build     -> data/eval/
score.py               runs the exam and grades every answer        -> data/eval/results.json
data/manifest.json     exactly which clearances the corpus holds
data/audit.csv         the audit, per file
data/chunk_stats.csv   chunks, words and sections per document
data/eval/             candidates.json (drafts), reviewed.json (every keep/drop
                       decision and its reason), eval_set.json (the exam),
                       results.json (the score)
```

---

## State of things, honestly

**The score is one small exam.** 34 of 40 and 9 of 9 refusals (see "The score")
is measured on 49 questions, mostly answered from a single document, and graded
by a model checked by hand only where it was lenient. Say "on its exam", not
"85% accurate".

**Cross-document questions are thin.** Only one of the ten drafted survived
review, so the exam barely tests questions that need a device summary and a
guidance document together.

**The gate is a judgement.** It is told to lean towards YES, so weak passages
sometimes get through. The drafting step catches some of those by declining.

**Heading detection is uneven.** 31.7% of 510(k) chunks have no detected section
heading, against a target of under 25%. Guidance documents, which are numbered
properly, sit at 10.0%.

**Older filings are under-represented**, for the scanned-paper reason above. OCR
would recover them.

**The live demo is a port.** The Python here runs PyTorch and an embedded Qdrant,
neither of which fits in a serverless function, so the website runs a TypeScript
version of the query path. It was checked against this code's own `Retriever` on
24 questions: identical tokens, embeddings at a lowest cosine of 0.99999997, and
the same top 30 by meaning and by keyword and the same top five in every mode.
The embedding model ships to the site at half precision, which kept every ranking
identical; 8-bit changed the top 30 on 23 of the 24 questions and was rejected.
