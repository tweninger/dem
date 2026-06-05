#!/usr/bin/env python3
# pip install openai>=1.40 pandas tqdm

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, json, re, os, sys, csv
from tqdm import tqdm

# ---------------- CONFIG ----------------
BASE_URL = "http://localhost:8082/v1"   # llama.cpp server
API_KEY  = "sk-no-key"                  # any non-empty string
MODEL    = "ggpl"                       # must match your server model
PARALLEL = 6                            # concurrent requests

PROMPT_TEMPLATE = """
 You are a senior political scientist analyzing social media posts. Your task is to classify the following social media post, which can be in any language (including Russian, Chinese, English, Arabic, etc.), into a specific category.
    Your response must be either one of the categories below in the format "Category Name". Do not add any explanations, introductory text, or quotation marks.
        **Context and Setting:** Consider the setting or context in which topics are discussed. Identify text that reference or discuss Western interference---either directly or indirectly---through accusations, implications, or criticism of Western involvement in political, economic, cultural, or social affairs of other countries.
    **Identify the Primary Focus:** Determine the main subject matter of the text. Choose the category that aligns with the central theme or the most prominent accusation or narrative related to Western influence or intervention. Include posts that employ explicit accusations, implied criticism, or metaphorical references to Western meddling or domination.
    **Categories Defined:**
    * Regime Change / Internal Instability: "Color Revolution", "Orange Revolution", "Euromaidan", "Maidan", "Arab Spring", "coup", "5th Column", "foreign agent", "foreign meddling"
    * Hostile Global Order: "hegemon", "hegemony", "imperialism", "colonialism", "NATO expansionism", "violations of sovereignty", "Western sanctions", "Western agenda", "Anti-China", "Anti-Russia", "Russophobia", "Sinophobia", "Cold War mentality", "unipolar"
    * Specific Adversary Framing: "collective West", "US-West", "US-led West" (???), "Western hypocrisy", "Western double-standard", "decadent West", "pretty country" (???)
    * Categorize as just "No Category" if the text does not belong to any of the mentioned categories.
    Here is the text to categorize:
    "{texts}"
"""
# ----------------------------------------

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def _make_prompt(text):
    return PROMPT_TEMPLATE.format(texts=text)

def _parse_json_array(text: str):
    """Try to extract the model's output cleanly."""
    text = text.strip()
    if not text:
        return ""
    try:
        arr = json.loads(text)
        if isinstance(arr, list) and arr:
            return arr[0]
    except Exception:
        pass
    m = re.search(r"\[(.*?)\]", text, flags=re.S)
    if m:
        inside = m.group(1).split(",")
        return inside[0].strip().strip('"')
    return text.splitlines()[0].strip()

def label_text(single_text):
    """Label a single text string."""
    prompt = _make_prompt(single_text)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=16,
        extra_body={"cache_prompt": True, "top_p": 0.0},
    )
    out = resp.choices[0].message.content.strip()
    return _parse_json_array(out)

def _determine_resume_index(in_csv, out_csv, col_name):
    if not os.path.exists(out_csv):
        return 0
    try:
        out_df = pd.read_csv(out_csv, encoding="utf-8-sig")
        if col_name in out_df.columns and "model_output" in out_df.columns:
            done = out_df["model_output"].notna() & (out_df["model_output"].astype(str) != "")
            return int(done.sum())
        return len(out_df)
    except Exception:
        with open(out_csv, "r", encoding="utf-8-sig") as f:
            return max(0, sum(1 for _ in f) - 1)

def _open_output_for_append(path, header):
    exists = os.path.exists(path)
    f = open(path, "a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(
        f,
        fieldnames=header,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )
    if not exists:
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())
    return f, writer

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with a 'text' column")
    ap.add_argument("--out", default="labeled_west.csv", help="Output CSV path")
    ap.add_argument("--col", default="text", help="Column to label")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    if args.col not in df.columns:
        print(f"ERROR: Column '{args.col}' not found in {args.csv}", file=sys.stderr)
        sys.exit(2)

    texts = df[args.col].astype(str).tolist()
    N = len(texts)
    start = _determine_resume_index(args.csv, args.out, args.col)
    start = max(0, min(start, N))
    print(f"Resuming at row {start}/{N}")

    header = list(df.columns) + ["model_output"]
    f, writer = _open_output_for_append(args.out, header)

    try:
        pbar = tqdm(total=N - start, desc="Labeling", smoothing=0.05)
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futures = {ex.submit(label_text, texts[i]): i for i in range(start, N)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    label = fut.result()
                except Exception as e:
                    print(f"[error] row {i} failed: {e}", file=sys.stderr)
                    label = ""
                row = {col: df.iloc[i][col] for col in df.columns}
                row["model_output"] = label
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                pbar.update(1)
                print(f"[{i+1}/{N}] -> {label}", flush=True)
        pbar.close()
    finally:
        f.close()

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
