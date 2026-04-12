import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, APIError, RateLimitError
from sklearn.metrics import cohen_kappa_score

# =========================================
# CONFIG
# =========================================
client = OpenAI(api_key="")  # <-- paste your API key here

REPORTS_PATH = "tests/metric4/reports_generated"
EVAL_DIR = "tests/metric4"
LLM_OUTPUT_FILE = os.path.join(EVAL_DIR, "llm_scores.csv")
HUMAN_FILE = os.path.join(EVAL_DIR, "human_scores.csv")
KAPPA_RESULTS_FILE = os.path.join(EVAL_DIR, "kappa_results.txt")

MODEL_NAME = "gpt-4o"
TEMPERATURE = 0
MAX_RETRIES = 3
RETRY_DELAY = 5
DIMENSIONS = ["accuracy", "depth", "risk", "coherence"]


# =========================================
# RUBRIC PROMPT
# =========================================
def build_prompt(report_text: str) -> str:
    return f"""You are a strict senior financial analyst grading AI-generated stock reports.

SCALE CALIBRATION:
  Score 3 is the DEFAULT for a competent but unremarkable report. Start here.
  Score 4 requires clear, specific evidence of quality ABOVE what an average analyst would produce.
  Score 5 is rare — reserve it only for genuinely exceptional work.
  Score 2 means the report has real weaknesses that would concern a professional reader.
  Score 1 means the report is seriously flawed or misleading.

SCORING RUBRIC:

1. Factual Accuracy — Are figures, ratios, and claims verifiable and financially correct?
  1 = Major errors (wrong ticker data, fabricated numbers, contradictory statements).
  2 = Figures cited without sourcing, or contain a specific verifiable error (e.g. wrong P/E, wrong revenue).
  3 = Claims are plausible but vague; no clear sourcing; relies on approximate or implied figures.
  4 = Uses specific, named figures correctly (e.g. "Q3 revenue of $X billion, up Y% YoY") with no material errors.
  5 = Precise, sourced, financially rigorous throughout — every claim could be fact-checked and would pass.

2. Analytical Depth — Does the report interpret data, or just describe it?
  1 = Pure description with no interpretation. Reads like a Wikipedia summary.
  2 = Shallow reasoning only — e.g. "EPS grew, which is good for investors."
  3 = Has some interpretation, but it is generic and not specific to this company or period.
  4 = Connects specific data points to company-level implications (e.g. explains WHY margin compression matters here).
  5 = Non-obvious, expert-level insight that a junior analyst would not produce.

3. Risk Coverage — Are the company-specific risks identified and explained?
  1 = No risks, or only a single boilerplate risk like "macroeconomic uncertainty."
  2 = Two or fewer risks listed without explanation of mechanism or likelihood.
  3 = Several risks named but not explained — reads like a checklist without substance.
  4 = Most key risks covered with a brief explanation of mechanism AND likely impact on this company.
  5 = Comprehensive: macro + sector + idiosyncratic risks, each with reasoning and relative prioritisation.

4. Recommendation Coherence — Is the recommendation justified by the analysis?
  1 = No recommendation, or recommendation contradicts the body of the report.
  2 = Boilerplate recommendation with no link to the analysis (e.g. "suitable for long-term investors").
  3 = Recommendation is directionally consistent but the connection to evidence is implicit, not stated.
  4 = Recommendation explicitly references specific findings from the report (e.g. "given the margin risk above...").
  5 = Recommendation is precise (e.g. includes entry conditions, price target, or time horizon) and fully justified.

CALIBRATION EXAMPLES:

Depth score 2: "Revenue increased 12% YoY, which shows strong growth momentum for the company."
  -> No explanation of driver, sustainability, or valuation impact.

Depth score 3: "Revenue growth was driven by cloud segment expansion, a positive industry trend."
  -> Identifies a driver, but generic and not tied to this company's specific position.

Depth score 4: "Cloud revenue grew 34% but operating margin contracted 200bps due to heavy R&D investment,
   suggesting the company is prioritising market share over near-term profitability — a defensible
   strategy given AWS and Azure's dominance, but one that increases execution risk."
  -> Company-specific, connects multiple data points, explains the trade-off.

Risk score 2: "Key risks include competition, regulation, and macroeconomic headwinds."
  -> All three are generic; none explain how they affect this specific company.

Risk score 4: "Regulatory risk is elevated: the EU AI Act requires compliance by 2026, and the company's
   core product may require architectural changes that could delay its enterprise roadmap by 6-12 months."
  -> Specific regulation, specific product impact, specific timeline.

CHAIN-OF-THOUGHT REQUIREMENT:
For each dimension write a brief "thinking" field BEFORE the score:
  - Strongest evidence for a higher score.
  - Clearest weakness preventing a higher score.
  - Final score.

STRICT OUTPUT RULES:
- Return ONLY valid raw JSON. No markdown fences. No preamble.
- Scores must be integers 1-5.
- "reason" must cite a specific phrase or claim from the report.

Return JSON in exactly this format:
{{
  "accuracy": {{"thinking": "...", "score": 3, "reason": "one specific sentence"}},
  "depth": {{"thinking": "...", "score": 3, "reason": "one specific sentence"}},
  "risk": {{"thinking": "...", "score": 3, "reason": "one specific sentence"}},
  "coherence": {{"thinking": "...", "score": 3, "reason": "one specific sentence"}}
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
    if content is None:
        return None
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def coerce_and_validate_scores(scores: dict) -> bool:
    try:
        for dim in DIMENSIONS:
            if dim not in scores:
                print(f"  [validate] Missing dimension: {dim}")
                return False
            if "score" not in scores[dim] or "reason" not in scores[dim]:
                print(f"  [validate] Missing 'score' or 'reason' in dimension: {dim}")
                return False
            raw_score = scores[dim]["score"]
            try:
                score = int(raw_score)
            except (TypeError, ValueError):
                print(f"  [validate] Cannot coerce score to int: {raw_score!r} in {dim}")
                return False
            if score < 1 or score > 5:
                print(f"  [validate] Score out of range (1-5): {score} in {dim}")
                return False
            scores[dim]["score"] = score
            if not isinstance(scores[dim]["reason"], str):
                print(f"  [validate] Reason is not a string in dimension: {dim}")
                return False
            scores[dim].pop("thinking", None)
        return True
    except Exception as e:
        print(f"  [validate] Unexpected error: {e}")
        return False


# =========================================
# LOAD REPORTS
# =========================================
def load_reports():
    reports = []
    if not os.path.exists(REPORTS_PATH):
        raise FileNotFoundError(f"Reports path not found: {REPORTS_PATH}")
    for file in sorted(os.listdir(REPORTS_PATH)):
        if file.endswith(".md"):
            filepath = os.path.join(REPORTS_PATH, file)
            with open(filepath, "r", encoding="utf-8") as f:
                reports.append({"filename": file, "text": f.read()})
    if not reports:
        raise ValueError(f"No .md files found in: {REPORTS_PATH}")
    print(f"Loaded {len(reports)} markdown reports from {REPORTS_PATH}")
    return reports


# =========================================
# GPT EVALUATION
# =========================================
def evaluate_report(report_text: str, filename: str = ""):
    prompt = build_prompt(report_text)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a precise JSON-only evaluator. You always respond with valid raw JSON and nothing else."},
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            parsed = extract_json(content)
            if parsed is None:
                print(f"  [score] JSON parse error for '{filename}' (attempt {attempt}).")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue
            if not coerce_and_validate_scores(parsed):
                print(f"  [score] Validation failed for '{filename}' (attempt {attempt}).")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                continue
            return parsed
        except RateLimitError:
            wait = RETRY_DELAY * attempt
            print(f"  [score] Rate limit hit. Waiting {wait}s (attempt {attempt}/{MAX_RETRIES}).")
            time.sleep(wait)
        except APIError as e:
            print(f"  [score] API error for '{filename}' (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    print(f"  [score] All {MAX_RETRIES} attempts failed for '{filename}'. Skipping.")
    return None


def run_llm_scoring():
    ensure_eval_dir()
    reports = load_reports()
    results = []
    failed = []

    for r in tqdm(reports, desc="Scoring reports"):
        scores = evaluate_report(r["text"], filename=r["filename"])
        if scores is None:
            failed.append(r["filename"])
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

    if failed:
        print(f"\n[WARNING] {len(failed)} report(s) could not be scored: {failed}")

    df = pd.DataFrame(results)
    if df.empty:
        print("[ERROR] No scores were generated. Check API key and report files.")
        return df

    df = df.sort_values("filename").reset_index(drop=True)
    df.to_csv(LLM_OUTPUT_FILE, index=False)
    print(f"\nSaved LLM scores to: {LLM_OUTPUT_FILE}")
    print(f"Scored {len(df)}/{len(reports)} reports successfully.")
    return df


# =========================================
# SUMMARY STATS
# =========================================
def summarize_scores(llm_df: pd.DataFrame) -> str:
    if llm_df.empty:
        return "No LLM scores available."

    lines = ["Average LLM Scores (1-5 scale):"]
    for d in DIMENSIONS:
        lines.append(f"  {d}: {llm_df[d].mean():.2f}  (std: {llm_df[d].std():.2f})")
    lines.append(f"  overall_average: {llm_df[DIMENSIONS].mean(axis=1).mean():.2f}")

    lines.append("\nScore distribution (count per value 1-5):")
    for d in DIMENSIONS:
        counts = llm_df[d].value_counts().sort_index()
        dist_str = "  ".join(f"{v}:{counts.get(v, 0)}" for v in range(1, 6))
        lines.append(f"  {d}: [{dist_str}]")

    summary = "\n".join(lines)
    print("\n" + summary)
    return summary


# =========================================
# INTER-RATER AGREEMENT
# =========================================
def compute_agreement(llm_df: pd.DataFrame, human_df: pd.DataFrame) -> str:
    """Compute quadratic-weighted Cohen's kappa between LLM and human scores."""
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
        suffixes=("_human", "_llm"),
    )
    if merged.empty:
        raise ValueError("No matching filenames found between human and LLM scores.")

    lines = [f"Matched reports: {len(merged)}", ""]

    # Human score distribution
    lines.append("Human Score Distribution (count per value 1-5):")
    for d in DIMENSIONS:
        if d not in human_df.columns:
            lines.append(f"  {d}: column not found")
            continue
        counts = human_df[d].value_counts().sort_index()
        dist_str = "  ".join(f"{v}:{counts.get(v, 0)}" for v in range(1, 6))
        std = human_df[d].std()
        lines.append(f"  {d}: [{dist_str}]  mean={human_df[d].mean():.2f}  std={std:.2f}")

    lines.append("")

    # Cohen's kappa
    lines.append("Quadratic Weighted Cohen's Kappa:")
    for d in DIMENSIONS:
        h = merged[f"{d}_human"]
        l = merged[f"{d}_llm"]
        if h.std() < 0.3:
            lines.append(f"  {d}:  N/A — human scores have near-zero variance (kappa undefined)")
        else:
            try:
                kappa = cohen_kappa_score(h, l, weights="quadratic")
                lines.append(f"  {d}: {kappa:.3f}")
            except Exception as e:
                lines.append(f"  {d}:  ERROR: {e}")

    result_text = "\n".join(lines)
    print("\n" + result_text)
    return result_text


# =========================================
# SAVE TEXT SUMMARY
# =========================================
def save_text_summary(text_blocks):
    ensure_eval_dir()
    final_text = "\n\n".join(b for b in text_blocks if b and b.strip())
    with open(KAPPA_RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"\nSaved evaluation summary to: {KAPPA_RESULTS_FILE}")


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    if not client.api_key:
        raise EnvironmentError(
            "API key is empty. Paste your key into the api_key field at the top of the file."
        )

    llm_df = run_llm_scoring()
    outputs = []

    if not llm_df.empty:
        outputs.append(summarize_scores(llm_df))
    else:
        outputs.append("No LLM scores generated.")

    if os.path.exists(HUMAN_FILE):
        human_df = pd.read_csv(HUMAN_FILE)
        human_df = human_df.sort_values("filename").reset_index(drop=True)
        outputs.append(compute_agreement(llm_df, human_df))
    else:
        msg = (
            f"Human score file not found at: {HUMAN_FILE}\n"
            "Skipping inter-rater agreement — add human_scores.csv once 20 reports are manually scored."
        )
        print("\n" + msg)
        outputs.append(msg)

    save_text_summary(outputs)
