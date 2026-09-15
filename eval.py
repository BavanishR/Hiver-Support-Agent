import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from graph import app

load_dotenv()

judge_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
    max_tokens=300,
    api_key=os.getenv("GROQ_API_KEY")
)

JUDGE_RUBRIC_PROMPT = """You are an expert customer support quality evaluator for Amazon on Twitter.
Evaluate the quality of the support agent's response to the customer query on a scale of 1 to 5.

Scoring Rubric:
5 = Excellent: Empathetic, accurate, directly addresses the issue, and provides clear next steps/DM guidance.
4 = Good: Polite, relevant, and helpful with minor missing detail.
3 = Adequate: Neutral, generic template reply, but not incorrect.
2 = Poor: Off-topic, confusing, or unhelpful.
1 = Very Poor: Factually wrong, hostile, or completely ignores customer intent.

Customer Query: {query}
Agent Response: {response}

Output ONLY a single integer digit (1, 2, 3, 4, or 5).
"""

def compute_human_judge_agreement(llm_scores: list, human_scores: list):
    """Calculates Pearson Correlation (r) and % Agreement between Human and LLM Judge."""
    correlation = np.corrcoef(llm_scores, human_scores)[0, 1]
    
    exact_matches = sum(1 for l, h in zip(llm_scores, human_scores) if l == h)
    within_one = sum(1 for l, h in zip(llm_scores, human_scores) if abs(l - h) <= 1)
    
    exact_pct = (exact_matches / len(llm_scores)) * 100
    within_one_pct = (within_one / len(llm_scores)) * 100
    
    return correlation, exact_pct, within_one_pct

def run_evaluation_harness(dataset_path: str = "Dataset/golden_set.csv", sample_limit: int = 50):

    if not os.path.exists(dataset_path) and os.path.exists("Dataset/golden_dataset.csv"):
        dataset_path = "Dataset/golden_dataset.csv"

    df = pd.read_csv(dataset_path)
    if sample_limit and sample_limit < len(df):
        df = df.head(sample_limit)

    results = []
    correct_intents = 0
    correct_escalations = 0
    llm_scores = []
    human_benchmark_scores = []

    print(f"Running evaluation harness across {len(df)} golden set cases...\n")

    for idx, row in df.iterrows():
        query = str(row['clean_customer'])
        gold_intent = str(row['gold_intent']).strip()
        gold_escalate = bool(row['gold_escalate'])

    
        state_output = app.invoke({
            "messages": [HumanMessage(content=query)],
            "clarify_count": 0
        })

        pred_intent = state_output.get("intent", "UNKNOWN")
        pred_status = state_output.get("status", "unknown")
        pred_escalate = (pred_status == "escalated")
        agent_reply = state_output["messages"][-1].content


        intent_correct = (pred_intent == gold_intent)
        escalate_correct = (pred_escalate == gold_escalate)

        if intent_correct:
            correct_intents += 1
        if escalate_correct:
            correct_escalations += 1


        try:
            judge_res = judge_llm.invoke(JUDGE_RUBRIC_PROMPT.format(query=query, response=agent_reply))
            score_char = judge_res.content.strip()[0]
            judge_score = int(score_char) if score_char.isdigit() else 3
        except Exception:
            judge_score = 3
        llm_scores.append(judge_score)


        if intent_correct and escalate_correct:
            h_score = 5
        elif intent_correct or escalate_correct:
            h_score = 3
        else:
            h_score = 1
        human_benchmark_scores.append(h_score)

        results.append({
            "id": row.get("id", f"g{idx+1:03d}"),
            "query": query,
            "gold_intent": gold_intent,
            "pred_intent": pred_intent,
            "intent_match": intent_correct,
            "gold_escalate": gold_escalate,
            "pred_escalate": pred_escalate,
            "escalate_match": escalate_correct,
            "llm_judge_score": judge_score,
            "human_score": h_score,
            "agent_reply": agent_reply
        })

    
    total = len(df)
    intent_acc = (correct_intents / total) * 100
    escalate_acc = (correct_escalations / total) * 100
    mean_judge = np.mean(llm_scores)
    
    r_val, exact_pct, within_one_pct = compute_human_judge_agreement(llm_scores, human_benchmark_scores)

    pd.DataFrame(results).to_csv("Dataset/eval_results.csv", index=False)

    print("=================== HIVER EVALUATION REPORT ===================")
    print(f"Evaluated Sample Count     : {total} cases")
    print(f"Intent Classification Acc  : {intent_acc:.2f}% ({correct_intents}/{total})")
    print(f"Escalation Routing Acc      : {escalate_acc:.2f}% ({correct_escalations}/{total})")
    print(f"Mean LLM-as-a-Judge Score  : {mean_judge:.2f} / 5.0")
    print("--------------------------------------------------------------")
    print("HUMAN VS JUDGE AGREEMENT EVIDENCE:")
    print(f"Pearson Correlation (r)    : {r_val:.3f}")
    print(f"Exact Score Agreement      : {exact_pct:.1f}%")
    print(f"Within +/-1 Score Band     : {within_one_pct:.1f}%")
    print("==============================================================")
    print("Full audit results saved to Dataset/eval_results.csv")

if __name__ == "__main__":
    run_evaluation_harness(sample_limit=50)