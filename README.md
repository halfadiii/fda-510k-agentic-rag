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
data/manifest.json     exactly which clearances the corpus holds
data/audit.csv         the audit, per file
data/chunk_stats.csv   chunks, words and sections per document
```

---

## State of things, honestly

**There is no evaluation set yet.** The citation check proves every citation
points at a passage the model was given. It does not prove the answer reads that
passage correctly. That needs questions with known answers, and it is the next
piece of work: retrieval recall for dense against BM25 against hybrid, then answer
faithfulness.

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
