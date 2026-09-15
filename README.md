# Amazon Twitter Support Agent

An AI customer-support agent for **Amazon**, built on real customer↔brand
Twitter conversations. Given an incoming customer message, the agent:

1. **Detects language** — non-English messages are escalated immediately
   (the retrieval index and embedding model are English-only).
2. **Classifies intent** into one of 7 categories, defined from the data.
3. **Routes** the message — auto-reply, clarify, escalate, or
   reply-then-escalate — based on intent and an explicit per-intent
   policy, not just a confidence threshold.
4. **Drafts a reply grounded in historical resolutions**, retrieved via
   semantic search (RAG) over real past Amazon support replies.
5. **Escalates to a human with a stated reason** whenever the situation
   requires account-level action, is ambiguous after 2 clarifications,
   or can't be safely auto-resolved.

> Per assignment rules ("we will not run your code on the full dataset —
> a subsample is expected"): the RAG index is built from a 3,000-message
> subsample, with the golden evaluation set programmatically excluded to
> prevent test-data leakage (see `ingest.py`).

---

## Architecture

```
customer message
      │
      ▼
detect_language ──(non-English)──► escalate ──► END
      │ (English)
      ▼
classify_intent
      │
      ▼
route_after_classification
 ├─ ACCOUNT_ACCESS ─────────────► escalate ──► END
 ├─ UNCLEAR_VAGUE / low conf ──► clarify (up to 2x) ──► escalate ──► END
 └─ everything else ───────────► generate_reply
                                       │
                                       ▼
                              route_after_reply
                               ├─ REFUND_AND_RETURN /
                               │  WRONG_DAMAGED_ITEM ──► escalate ──► END
                               └─ else ──────────────────► END
```

Built with **LangGraph** (routing/orchestration), **Groq**
(`openai/gpt-oss-120b` via `langchain_groq`) for classification and reply
generation, **ChromaDB** + `all-MiniLM-L6-v2` (HuggingFace, local) for
semantic retrieval.

---

## Prerequisites

- Python 3.13 (see `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) (or plain `pip`)
- A free [Groq API key](https://console.groq.com/keys)
- (Optional) a free [Hugging Face token](https://huggingface.co/settings/tokens) — avoids a download rate-limit warning, not required to run

---

## Setup (~3 min)

```bash
git clone <this-repo-url>
cd Hiver-Project

uv sync                       # or: pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_hf_token_here   # optional
```

---

## Reproduce results (~10 min total)

### 1. Build the RAG index (~2-3 min)
```bash
python ingest.py
```
Embeds a 3,000-message subsample of `Dataset/amazon_pairs.csv` into
`chroma_db/`. Automatically filters out every message present in
`Dataset/golden_dataset.csv` first, so the evaluation set never leaks
into the retrieval corpus (console output confirms how many rows were
filtered).

### 2. Live demo (~2 min)
```bash
python main.py
```
Try a few messages, e.g.:
```
Customer: my order still hasn't shipped, it's been 5 days
Customer: I ordered a book and got a damaged copy
Customer: I can't log into my account, OTP not arriving
Customer: mi paquete llega tarde
```
Each should be classified, routed, and either replied to or escalated
with a stated reason — the last one (Spanish) should escalate
immediately without attempting a reply.

### 3. Reproduce the headline evaluation numbers (~3-5 min)
```bash
python eval.py
```
This runs the pipeline against a 50-example sample of the golden set
(`Dataset/golden_dataset.csv`), scores every reply with an LLM judge,
and prints:
- Intent classification accuracy
- Escalation routing accuracy
- Mean LLM-as-judge score
- Judge-vs-human agreement (Pearson r, exact match %, within-1 %)

Full per-example results are written to `Dataset/eval_results.csv`.

> Sample size is intentionally 50 (not the full 210-row golden set) to
> stay within the 15-minute reproducibility budget given Groq free-tier
> rate limits. See `REPORT.md` → "What's misleading about my headline
> number" for why this matters.

---

## Headline results

| Metric | Value |
|---|---|
| Intent accuracy | 70.0% |
| Escalation accuracy | 72.0% |
| Mean LLM-judge score | 3.78 / 5.0 |
| Judge–human agreement (Pearson r) | 0.123 |

These numbers are **not** production-ready as-is — see `REPORT.md` for
why (small sample size, low judge-human correlation, judge rewards tone
over correctness).

---

## Repo structure

```
main.py                interactive CLI entry point
graph.py                 LangGraph pipeline definition + routing logic
nodes.py                   node implementations (detect_language, classify,
                              clarify, escalate, generate_reply)
state.py                     shared agent state schema
schemas.py                     structured-output schema for intent classification
ingest.py                       builds chroma_db from Dataset/amazon_pairs.csv,
                                   excludes golden set rows automatically
eval.py                           evaluation harness: runs pipeline + LLM judge
                                     + judge-human agreement stats on golden set
Dataset/
  twcs.csv                          raw Kaggle Customer Support on Twitter dataset (gitignored — see below)
  amazon_pairs.csv                    cleaned Amazon customer<->agent pairs
  golden_dataset.csv                    210 hand-labelled golden evaluation examples
  eval_results.csv                        per-example output from eval.py
  dataset_analysis.ipynb                    data cleaning / EDA notebook
REPORT.md                                     problem framing, baselines, failure analysis
DECISIONS.md                                    decision log
```

`Dataset/twcs.csv` (the raw ~2.8M-row Kaggle dataset) is gitignored due
to size. Download it from
[Kaggle: Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
if you want to re-run `dataset_analysis.ipynb` from scratch — it is
**not** required to run `ingest.py`, `main.py`, or `eval.py`, since
`amazon_pairs.csv` (the cleaned output) is already committed.

## Known limitations

- Language detection (`langdetect`) is unreliable on very short messages
  (<10 chars); these are treated as English by default to avoid
  over-escalating short replies like "no" or "ok".
- Retrieval and reply generation are English-only; non-English messages
  are escalated rather than answered.
- See `REPORT.md` for the full failure analysis and 5 concrete examples.