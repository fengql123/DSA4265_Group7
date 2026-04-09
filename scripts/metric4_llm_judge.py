import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score

# =========================================
# CONFIG
# =========================================
client = OpenAI(api_key="")
# <-- put your API key here, or use env var below

# If you prefer environment variable, uncomment below and comment the line above:
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

REPORTS_PATH = "tests/metric4/reports_generated"
EVAL_DIR = "tests/metric4"
LLM_OUTPUT_FILE = os.path.join(EVAL_DIR, "llm_scores.csv")
HUMAN_FILE = os.path.join(EVAL_DIR, "human_scores.csv")
KAPPA_RESULTS_FILE = os.path.join(EVAL_DIR, "kappa_results.txt")

MODEL_NAME = "gpt-4o-mini"   # change if needed
TEMPERATURE = 0

DIMENSIONS = ["accuracy", "depth", "risk", "coherence"]


# =========================================
# RUBRIC PROMPT
# =========================================
def build_prompt(report_text: str) -> str:
    return f"""
You are a senior financial analyst evaluating an AI-generated stock analysis report.

Evaluate the report on the following 4 dimensions using a score from 1 to 5.

Scoring rubric:

1. Factual Accuracy
- 1 = Contains major factual errors, unsupported claims, or financially incorrect statements.
- 2 = Contains several inaccuracies or weakly supported claims.
- 3 = Mostly accurate, but includes some vague, weak, or partially unsupported statements.
- 4 = Accurate overall, with only minor issues.
- 5 = Highly accurate, financially sound, and free from meaningful factual errors.

2. Analytical Depth
- 1 = Very superficial, mostly descriptive, little or no analysis.
- 2 = Limited reasoning, weak interpretation of evidence.
- 3 = Moderate analysis, some reasoning present but not especially deep.
- 4 = Strong reasoning and interpretation of the company situation and risks.
- 5 = Insightful, nuanced, and demonstrates strong financial reasoning.

3. Risk Coverage
- 1 = Fails to identify meaningful risks.
- 2 = Identifies only a few risks and misses important ones.
- 3 = Covers several important risks, but some key areas are missing.
- 4 = Covers most major relevant risks well.
- 5 = Comprehensive and well-balanced discussion of the key risks.

4. Recommendation Coherence
- 1 = Recommendations are missing, contradictory, or not justified.
- 2 = Recommendations are weak, generic, or poorly connected to the analysis.
- 3 = Recommendations are somewhat reasonable, but not especially strong or specific.
- 4 = Recommendations are clear, logical, and mostly well-supported.
- 5 = Recommendations are highly coherent, actionable, and directly supported by the analysis.

Instructions:
- Be critical and consistent.
- Do not give high scores unless clearly justified by the report quality.
- Judge only the report provided.
- Return ONLY valid raw JSON.
- Do not include markdown fences.
- Keep each reason short, specific, and one sentence.

Return JSON in exactly this format:
{{
  "accuracy": {{"score": 1, "reason": "text"}},
  "depth": {{"score": 1, "reason": "text"}},
  "risk": {{"score": 1, "reason": "text"}},
  "coherence": {{"score": 1, "reason": "text"}}
}}

Report:
\"\"\"
{report_text}
\"\"\"
"""


# =========================================
# HELPERS
# =========================================
def ensure_eval_dir() -> None:
    os.makedirs(EVAL_DIR, exist_ok=True)


def extract_json(content: str):
    """
    Try to parse model output as JSON.
    Handles raw JSON and simple code-fence wrappers.
    """
    if content is None:
        return None

    content = content.strip()

    # raw JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # strip markdown fences if present
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) >= 3:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass

    return None


def validate_scores(scores: dict) -> bool:
    """
    Ensure expected structure exists and scores are integers 1-5.
    """
    try:
        for dim in DIMENSIONS:
            if dim not in scores:
                return False
            if "score" not in scores[dim] or "reason" not in scores[dim]:
                return False

            score = scores[dim]["score"]
            if not isinstance(score, int):
                return False
            if score < 1 or score > 5:
                return False

            reason = scores[dim]["reason"]
            if not isinstance(reason, str):
                return False

        return True
    except Exception:
        return False


# =========================================
# GPT EVALUATION
# =========================================
def evaluate_report(report_text: str):
    prompt = build_prompt(report_text)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
    )

    content = response.choices[0].message.content
    parsed = extract_json(content)

    if parsed is None:
        print("JSON parse error. Raw output:\n", content)
        return None

    if not validate_scores(parsed):
        print("Invalid score structure:\n", parsed)
        return None

    return parsed


# =========================================
# LOAD REPORTS
# =========================================
def load_reports():
    reports = []

    if not os.path.exists(REPORTS_PATH):
        raise FileNotFoundError(f"Reports path not found: {REPORTS_PATH}")

    for file in os.listdir(REPORTS_PATH):
        if file.endswith(".md"):
            filepath = os.path.join(REPORTS_PATH, file)

            with open(filepath, "r", encoding="utf-8") as f:
                reports.append({
                    "filename": file,
                    "text": f.read()
                })

    reports = sorted(reports, key=lambda x: x["filename"])
    print(f"Loaded {len(reports)} markdown reports from {REPORTS_PATH}")
    return reports


# =========================================
# RUN LLM SCORING
# =========================================
def run_llm_scoring():
    ensure_eval_dir()
    reports = load_reports()
    results = []

    for r in tqdm(reports, desc="Scoring reports"):
        scores = evaluate_report(r["text"])

        if scores is None:
            print(f"Skipping {r['filename']} due to parse/validation failure.")
            continue

        results.append({
            "filename": r["filename"],
            "accuracy": scores["accuracy"]["score"],
            "accuracy_reason": scores["accuracy"]["reason"],
            "depth": scores["depth"]["score"],
            "depth_reason": scores["depth"]["reason"],
            "risk": scores["risk"]["score"],
            "risk_reason": scores["risk"]["reason"],
            "coherence": scores["coherence"]["score"],
            "coherence_reason": scores["coherence"]["reason"],
        })

    df = pd.DataFrame(results)

    if df.empty:
        print("No scores were generated.")
        return df

    df = df.sort_values("filename").reset_index(drop=True)
    df.to_csv(LLM_OUTPUT_FILE, index=False)

    print(f"\nSaved LLM scores to: {LLM_OUTPUT_FILE}")
    return df


# =========================================
# SUMMARY STATS
# =========================================
def summarize_scores(llm_df: pd.DataFrame) -> str:
    if llm_df.empty:
        return "No LLM scores available."

    lines = []
    lines.append("Average LLM Scores:")

    for d in DIMENSIONS:
        mean_score = llm_df[d].mean()
        lines.append(f"- {d}: {mean_score:.2f}")

    overall = llm_df[DIMENSIONS].mean(axis=1).mean()
    lines.append(f"- overall_average: {overall:.2f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    return summary


# =========================================
# COHEN'S KAPPA
# =========================================
def compute_kappa(llm_df: pd.DataFrame, human_df: pd.DataFrame) -> str:
    required_cols = ["filename"] + DIMENSIONS

    for col in required_cols:
        if col not in llm_df.columns:
            raise ValueError(f"Missing column in llm_df: {col}")
        if col not in human_df.columns:
            raise ValueError(f"Missing column in human_df: {col}")

    merged = pd.merge(
        human_df[required_cols],
        llm_df[required_cols],
        on="filename",
        suffixes=("_human", "_llm")
    )

    if merged.empty:
        raise ValueError("No matching filenames found between human and LLM scores.")

    lines = []
    lines.append(f"Matched reports: {len(merged)}")
    lines.append("Quadratic Weighted Cohen's Kappa Results:")

    for d in DIMENSIONS:
        kappa = cohen_kappa_score(
            merged[f"{d}_human"],
            merged[f"{d}_llm"],
            weights="quadratic"
        )
        lines.append(f"- {d}: {kappa:.3f}")

    result_text = "\n".join(lines)
    print("\n" + result_text)
    return result_text


# =========================================
# SAVE TEXT SUMMARY
# =========================================
def save_text_summary(text_blocks):
    ensure_eval_dir()
    final_text = "\n\n".join(block for block in text_blocks if block and block.strip())

    with open(KAPPA_RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"\nSaved evaluation summary to: {KAPPA_RESULTS_FILE}")


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    llm_df = run_llm_scoring()

    outputs = []

    if not llm_df.empty:
        outputs.append(summarize_scores(llm_df))
    else:
        outputs.append("No LLM scores generated.")

    if os.path.exists(HUMAN_FILE):
        human_df = pd.read_csv(HUMAN_FILE)

        # Sort / clean if needed
        human_df = human_df.sort_values("filename").reset_index(drop=True)

        kappa_text = compute_kappa(llm_df, human_df)
        outputs.append(kappa_text)
    else:
        msg = (
            f"Manual score file not found at: {HUMAN_FILE}\n"
            "Skipping Cohen's kappa calculation for now."
        )
        print("\n" + msg)
        outputs.append(msg)

    save_text_summary(outputs)
