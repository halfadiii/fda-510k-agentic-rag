"""
Day 5: Score the system against the exam.

THREE SCORES, KEPT APART
    retrieval   did the top 5 passages include the gold page?        (no AI)
    behaviour   did it answer when it should and refuse when it should? (no AI)
    facts       did the answer contain the key facts written in review? (AI judge)

Keeping them apart says where a failure came from: a wrong answer with the
gold page retrieved is a generation problem; without it, a search problem.

THE JUDGE
    The judge sees the question, the reviewed reference answer and its key
    facts, and the system's answer, and marks each key fact present or absent,
    plus whether the answer says anything that contradicts the reference.
    It does not see the passages, so it cannot be talked round by them.

    Two verdicts are reported. "Correct" means it fully answers the question
    as asked and contradicts nothing in the reference. "All key facts" is
    stricter: some reviewed key facts go beyond what the question asks.

Run:  python score.py              # all 49, ~$0.10 of DeepSeek calls
      python score.py --limit 3    # a quick check first
Writes data/eval/results.json and prints the summary.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from answer import Assistant, client, GRADER_MODEL

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_SET = Path("data/eval/eval_set.json")
RESULTS = Path("data/eval/results.json")

JUDGE_PROMPT = """You are grading an answer against a reviewed reference.

QUESTION: {question}

REFERENCE ANSWER: {reference}

KEY FACTS (each must appear in the answer, in any wording):
{facts}

ANSWER TO GRADE:
{answer}

For each key fact, say whether the answer states it (same meaning counts;
a vaguer or partial version does not). Then say whether the answer states
anything that contradicts the reference. Finally, say whether the answer
fully answers the question as asked, judged against the reference (some key
facts may go beyond what the question asks; missing those alone does not make
the answer incomplete).

Reply with JSON only:
{{"facts": [true or false, one per key fact, in order], "contradicts": true or false, "answers_question": true or false, "note": "one short sentence"}}"""


def gold_hit(q: dict, results) -> bool:
    for g in q["gold"]:
        for r in results:
            if (r.doc_id == g["doc_id"] and r.page_start <= g["page_end"]
                    and r.page_end >= g["page_start"]):
                return True
    return False


def judge(q: dict, answer_text: str) -> tuple[dict, int, int]:
    facts = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(q["key_facts"]))
    prompt = JUDGE_PROMPT.format(question=q["question"],
                                 reference=q["reference_answer"],
                                 facts=facts, answer=answer_text)
    for _ in range(3):
        resp = client.chat.completions.create(
            model=GRADER_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])
        u = resp.usage
        try:
            out = json.loads(resp.choices[0].message.content)
            if len(out.get("facts", [])) == len(q["key_facts"]):
                return out, u.prompt_tokens, u.completion_tokens
        except json.JSONDecodeError:
            pass
    return {"facts": [False] * len(q["key_facts"]), "contradicts": False,
            "note": "judge failed to return valid JSON"}, 0, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    exam = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if args.limit:
        exam = exam[:args.limit]

    bot = Assistant()
    rows, tin, tout = [], 0, 0
    try:
        for i, q in enumerate(exam, 1):
            ans = bot.ask(q["question"])
            tin += ans.tokens_in
            tout += ans.tokens_out
            row = {
                "id": q["id"], "category": q["category"],
                "expected": q["expected_behavior"],
                "question": q["question"],
                "refused": ans.refused, "refusal_reason": ans.refusal_reason,
                "answer": ans.text,
                "retrieved": [f"{r.doc_id} p.{r.page_start}-{r.page_end}"
                              for r in ans.sources_retrieved],
                "latency_s": round(ans.latency_s, 2),
            }
            if q["expected_behavior"] == "answer":
                row["gold_retrieved"] = gold_hit(q, ans.sources_retrieved)
                if not ans.refused:
                    verdict, a, b = judge(q, ans.text)
                    tin += a
                    tout += b
                    row["facts"] = verdict["facts"]
                    row["contradicts"] = verdict["contradicts"]
                    row["judge_note"] = verdict.get("note", "")
                    row["all_facts"] = all(verdict["facts"]) and not verdict["contradicts"]
                    row["correct"] = bool(verdict.get("answers_question")) and not verdict["contradicts"]
                else:
                    row["facts"] = [False] * len(q["key_facts"])
                    row["all_facts"] = False
                    row["correct"] = False
                mark = "correct" if row["correct"] else (
                    "refused" if ans.refused else
                    f"{sum(row['facts'])}/{len(row['facts'])} facts")
            else:
                row["correct"] = ans.refused
                mark = "refused" if ans.refused else "ANSWERED"
            rows.append(row)
            print(f"{i:>2}/{len(exam)}  {q['id']:<24} {mark}")
    finally:
        bot.close()

    ans_rows = [r for r in rows if r["expected"] == "answer"]
    ref_rows = [r for r in rows if r["expected"] == "refuse"]
    n_facts = sum(len(r["facts"]) for r in ans_rows)
    summary = {
        "questions": len(rows),
        "answerable": len(ans_rows),
        "answered_correctly": sum(r["correct"] for r in ans_rows),
        "all_key_facts": sum(r["all_facts"] for r in ans_rows),
        "wrongly_refused": sum(r["refused"] for r in ans_rows),
        "answered_with_contradiction": sum(bool(r.get("contradicts")) for r in ans_rows),
        "key_facts_found": sum(sum(r["facts"]) for r in ans_rows),
        "key_facts_total": n_facts,
        "gold_page_in_top5": sum(r["gold_retrieved"] for r in ans_rows),
        "should_refuse": len(ref_rows),
        "refused_correctly": sum(r["correct"] for r in ref_rows),
        "cost_usd_approx": round((tin * 0.27 + tout * 1.10) / 1_000_000, 4),
        "scored_at": time.strftime("%Y-%m-%d %H:%M"),
        "model": GRADER_MODEL,
    }
    RESULTS.write_text(json.dumps({"summary": summary, "rows": rows},
                                  indent=1, ensure_ascii=False), encoding="utf-8")

    s = summary
    print(f"\nAnswerable questions: {s['answered_correctly']}/{s['answerable']} fully correct, "
          f"{s['all_key_facts']} with every key fact, "
          f"{s['key_facts_found']}/{s['key_facts_total']} key facts, "
          f"{s['wrongly_refused']} wrongly refused, "
          f"{s['answered_with_contradiction']} with a contradiction")
    print(f"Gold page in the top 5: {s['gold_page_in_top5']}/{s['answerable']}")
    print(f"Should refuse: {s['refused_correctly']}/{s['should_refuse']} refused")
    print(f"Cost: about ${s['cost_usd_approx']}   -> {RESULTS}")


if __name__ == "__main__":
    main()
