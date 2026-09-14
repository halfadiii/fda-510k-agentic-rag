"""
Day 3: Turn retrieved chunks into a grounded, cited answer.

THE PIPELINE
    question
      -> retrieve 5 chunks (Day 2)
      -> RELEVANCE GATE: do these chunks actually answer this? ......... [1]
      -> generate an answer that cites sources as [S1], [S2] ........... [2]
      -> CITATION CHECK: is every cited source one we actually gave? ... [3]
      -> if bad, regenerate once; if still bad, refuse
      -> answer + real citations

[1] solves the "wool" problem from Day 2: retrieval always returns something,
    so something else has to decide whether it's relevant.

[3] is the guarantee. It's plain Python, no AI. If the model cites a source we
    never showed it, the answer is thrown away. That lets you make a hard claim
    in your README: "this system cannot cite a source it did not read."

WHY NUMBERED CITATIONS [S1] AND NOT "K212521 p.5"
    Free-text citations can't be verified -- the model can typo, paraphrase, or
    invent a plausible-looking document ID, and you can't reliably tell which.
    Numbered slots can only be right or wrong. You map them back to real
    metadata yourself, so the citation the user sees is always accurate.

Run:  python answer.py "how are silver dressings sterilized?"
"""

import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from dotenv import load_dotenv
from openai import OpenAI

from search import Retriever, Result

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

MODEL = "deepseek-chat"          # V3. Fast and cheap. R1 is for grading, Day 5.
GRADER_MODEL = "deepseek-chat"   # relevance gate -- keep it cheap
TOP_K = 5
MAX_REGENERATIONS = 1

REFUSAL = ("I don't have that in my documents. "
           "My corpus covers FDA 510(k) wound dressing clearances and related "
           "guidance documents.")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


# ---------------------------------------------------------------------------

@dataclass
class Answer:
    text: str
    citations: list[dict] = field(default_factory=list)
    sources_retrieved: list[Result] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    attempts: int = 1
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def cost_usd(self) -> float:
        # DeepSeek V3 pricing, approximate. Update if it changes.
        return (self.tokens_in * 0.27 + self.tokens_out * 1.10) / 1_000_000


# ---------------------------------------------------------------------------
# Step 1: The relevance gate  (the fix for the "wool" problem)
# ---------------------------------------------------------------------------

GATE_PROMPT = """You are checking whether retrieved passages are relevant to a question.

QUESTION: {question}

PASSAGES:
{passages}

Do these passages contain information that helps answer the question?

Answer with one word:
YES  - the passages contain relevant information, even if partial or spread
       across several passages
NO   - the passages are about a different subject entirely, or the specific
       thing asked about is never mentioned

Default to YES when the passages are on-topic. Only answer NO when the question
asks about something genuinely absent from the passages."""


def passages_are_relevant(question: str, results: list[Result]) -> bool:
    """
    Retrieval always returns its top-k, relevant or not. This asks a model to
    make the judgement retrieval can't.

    Deliberately strict: over-refusing is a much cheaper error than
    confidently answering from irrelevant text.
    """
    passages = "\n\n".join(
        f"[S{i}] {r.text}" for i, r in enumerate(results, 1)
    )
    resp = client.chat.completions.create(
        model=GRADER_MODEL,
        messages=[{"role": "user",
                   "content": GATE_PROMPT.format(question=question,
                                                 passages=passages)}],
        temperature=0,
        max_tokens=5,
    )
    verdict = resp.choices[0].message.content.strip().upper()
    return verdict.startswith("YES")


# ---------------------------------------------------------------------------
# Step 2: Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You answer questions about FDA medical device regulations \
and 510(k) wound dressing clearances, using ONLY the sources provided.

RULES
1. Use only information in the numbered sources. Never use outside knowledge.
2. Cite every factual sentence with the source tag, like [S1] or [S2, S4].
3. Cite ONLY sources that appear in the list. Never invent a source number.
4. If the sources don't contain the answer, reply exactly:
   "INSUFFICIENT: " followed by what specific information is missing.
5. Never state a regulatory requirement you cannot cite.
6. Be concise. Two to four sentences unless the question needs more.
7. Do not repeat the question or add a preamble. Answer directly."""

USER_PROMPT = """SOURCES:
{sources}

QUESTION: {question}

Answer using only the sources above, citing each factual claim."""


def format_sources(results: list[Result]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[S{i}] (document: {r.doc_id}, page {r.page_start}, "
            f"section: {r.section})\n{r.text}"
        )
    return "\n\n".join(blocks)


def generate(question: str, results: list[Result],
             feedback: str = "") -> tuple[str, int, int]:
    user = USER_PROMPT.format(sources=format_sources(results),
                              question=question)
    if feedback:
        user += f"\n\nIMPORTANT: {feedback}"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        temperature=0,          # deterministic, so evals are reproducible
        max_tokens=600,
    )
    return (resp.choices[0].message.content.strip(),
            resp.usage.prompt_tokens,
            resp.usage.completion_tokens)


# ---------------------------------------------------------------------------
# Step 3: The citation checker  (plain code -- no AI involved)
# ---------------------------------------------------------------------------

CITE_RE = re.compile(r"\[(?:S|Source\s*)?(\d+(?:\s*,\s*(?:S|Source\s*)?\d+)*)\]",
                     re.IGNORECASE)


def extract_citations(text: str) -> list[int]:
    """Pull every source number out of an answer. Handles [S1] and [S2, S4]."""
    found: list[int] = []
    for match in CITE_RE.finditer(text):
        for part in match.group(1).split(","):
            num = re.sub(r"[^\d]", "", part)
            if num:
                found.append(int(num))
    return found


def validate_citations(text: str, n_sources: int) -> tuple[bool, list[int], str]:
    """
    The guarantee. Returns (is_valid, invalid_numbers, reason).

    Three ways an answer fails:
      - cites a source number that doesn't exist  -> hallucinated citation
      - cites nothing at all                      -> ungrounded claim
      - is empty                                  -> malformed
    """
    cited = extract_citations(text)

    invalid = [n for n in cited if n < 1 or n > n_sources]
    if invalid:
        return False, invalid, (
            f"cited nonexistent source(s) {invalid}; only S1-S{n_sources} exist")

    if not cited and not text.startswith("INSUFFICIENT"):
        return False, [], "no citations at all"

    if not text.strip():
        return False, [], "empty answer"

    return True, [], ""


def resolve_citations(text: str, results: list[Result]) -> list[dict]:
    """Turn validated [S1] tags into real, human-checkable source references."""
    out = []
    for n in sorted(set(extract_citations(text))):
        r = results[n - 1]
        out.append({
            "tag": f"S{n}",
            "doc_id": r.doc_id,
            "doc_title": r.doc_title,
            "doc_type": r.doc_type,
            "page": r.page_start,
            "section": r.section,
            "chunk_id": r.chunk_id,
        })
    return out


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

class Assistant:
    def __init__(self, mode: str = "hybrid", use_gate: bool = True) -> None:
        self.retriever = Retriever()
        self.mode = mode
        self.use_gate = use_gate

    def ask(self, question: str, k: int = TOP_K) -> Answer:
        start = time.time()

        results = self.retriever.search(question, k=k, mode=self.mode)

        if not results:
            return Answer(text=REFUSAL, refused=True,
                          refusal_reason="no results retrieved",
                          latency_s=time.time() - start)

        # --- gate ----------------------------------------------------------
        if self.use_gate and not passages_are_relevant(question, results):
            return Answer(text=REFUSAL, sources_retrieved=results,
                          refused=True, refusal_reason="passages not relevant",
                          latency_s=time.time() - start)

        # --- generate, validate, maybe retry -------------------------------
        feedback = ""
        tin = tout = 0

        for attempt in range(1, MAX_REGENERATIONS + 2):
            text, a, b = generate(question, results, feedback)
            tin += a
            tout += b

            ok, invalid, reason = validate_citations(text, len(results))

            if ok:
                if text.startswith("INSUFFICIENT"):
                    missing = text[len("INSUFFICIENT:"):].strip()
                    return Answer(
                        text=f"{REFUSAL}\n\nSpecifically missing: {missing}",
                        sources_retrieved=results, refused=True,
                        refusal_reason="model reported insufficient context",
                        attempts=attempt, latency_s=time.time() - start,
                        tokens_in=tin, tokens_out=tout)

                return Answer(
                    text=text,
                    citations=resolve_citations(text, results),
                    sources_retrieved=results,
                    attempts=attempt,
                    latency_s=time.time() - start,
                    tokens_in=tin, tokens_out=tout)

            # invalid -- tell it what went wrong and try once more
            feedback = (f"Your previous answer {reason}. "
                        f"Use only [S1] through [S{len(results)}], and cite "
                        f"every factual sentence.")

        # exhausted retries -> refuse rather than serve an unverified answer
        return Answer(
            text=REFUSAL, sources_retrieved=results, refused=True,
            refusal_reason=f"citation validation failed after "
                           f"{MAX_REGENERATIONS + 1} attempts: {reason}",
            attempts=MAX_REGENERATIONS + 1, latency_s=time.time() - start,
            tokens_in=tin, tokens_out=tout)

    def close(self) -> None:
        self.retriever.close()


# ---------------------------------------------------------------------------

def show(question: str, ans: Answer) -> None:
    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("=" * 70)
    print(f"\n{ans.text}\n")

    if ans.citations:
        print("SOURCES")
        for c in ans.citations:
            print(f"  [{c['tag']}] {c['doc_id']}  p.{c['page']}  "
                  f"({c['section'][:45]})")
            print(f"       {c['doc_title'][:60]}")

    if ans.refused:
        print(f"  [refused: {ans.refusal_reason}]")

    print(f"\n  {ans.latency_s:.1f}s | {ans.attempts} attempt(s) | "
          f"{ans.tokens_in}+{ans.tokens_out} tokens | ${ans.cost_usd:.5f}")


def main() -> None:
    args = sys.argv[1:]
    use_gate = "--no-gate" not in args
    q = " ".join(sys.argv[1:])
    a = Assistant()

    if q:
        show(q, a.ask(q))
    else:
        # The last one is the wool question from Day 2 -- it MUST refuse.
        for question in [
            "What biocompatibility testing does FDA expect for a dressing that "
            "contacts an open wound?",
            "What is a predicate device?",
            "How is the Algidex Ag dressing sterilized?",
            "Is wool a good wound dressing material?",
        ]:
            show(question, a.ask(question))

    a.close()


if __name__ == "__main__":
    main()
