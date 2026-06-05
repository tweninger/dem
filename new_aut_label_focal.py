from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import pandas as pd, json, re, os, sys, csv, socket, traceback
from tqdm import tqdm

# ---------------- CONFIG ----------------
API_KEY  = "sk-no-key"
MODEL    = "ggpl"
PARALLEL = 8

BASE_HOST = "127.0.0.1"
BASE_PORT = 8080
MAX_PORTS_TO_SCAN = 8

TEXT_COL = "text"

OUT_COLS = ["AUT_SUBCATEGORY", "FOCAL_COUNTRY"]

AUT_LABEL_COL = "AUT_LABEL"
DEM_LABEL_COL = "DEM_LABEL"
WEST_LABEL_COL = "WEST_LABEL"

# ---------------- PROMPTS ----------------
AUT_SUBCATEGORY_PROMPT = """
You are a senior political scientist analyzing social media posts. Your task is to classify the following social media post, which can be in any language (including Russian, Chinese, English, Arabic, etc.), into a specific category.
    Your response must be either one of the categories below in the format "Category Name". Do not add any explanations, introductory text, or quotation marks.

The post has already been determined to belong to the Authoritarian frame.
Choose the single best-fitting subcategory from the six categories below.
	
    **Context and Setting:** Consider the setting or context in which topics are discussed. Identify text that reference or discuss the promotion of autocracy---either directly or indirectly---through praise, justification, or support for authoritarian governance, efficiency, or economic success tied to authoritarian models. 
    **Identify the Primary Focus:** Determine the main subject matter of the text. Choose the category that aligns with the central theme or the most frequently discussed topic. Categorize texts that mention support or endorse authoritarian leaders or regimes explicitly or implicitly through related ideas or keywords.
	
    **Categories Defined:**
    * Authoritarian - Military/Security:  Post mentions military influence, military cooperation, and military strength of China or Russia, or other authoritarian regimes Keywords: "military support", "security cooperation", "bloc", "military training"; "military prowess"; "military strength", "military cooperation"
    * Authoritarian - Economic Influence: Post mentions economic influence and/or cooperation of China or Russia, or other authoritarian regimes. Keywords: "foreign aid", "Belt and Road", "BRI", "South-South", "global South", "development aid", "economic partnership", "economic cooperation", "foreign direct investment", "FDI", "trade"
    * Authoritarian - Digital: Posts mention the use of technology and/or digital tools to monitor the public. Keywords: "surveillance", "facial recognition", "censorship", "firewall", "social credit", "biometric"
    * Authoritarian - Legal Tools for Entrenchment: Post mentions legal tools or strategies used to protect authoritarian regimes or entrench their leaders. Keywords: "anti-terror law", "national security law", "emergency powers", "constitutional change"
    * Authoritarian - Alliances: Post discusses international alliances and partnerships of China or Russia, or other authoritarian regimes. Keywords: "Shanghai Cooperation Organization", "SCO", "BRICS", "EAEU", "Eurasian Economic Union", "GCC", "Gulf Cooperation Council", "strategic partnership", "multipolar", "Collective Security Treaty Organization", "CSTO", "Alliance of Sahel States", "AES", "Arab League" "Russia-[another country]", "China-[another country]", "alliance"
    * Authoritarian - Ideological Promotion: Post promotes the ideology of China or Russia, or other authoritarian regimes. Post discusses authoritarian or anti-liberal values. Keywords: "socialism with Chinese characteristics", "national rejuvenation", "Chinese model", "Russian model", "Third Rome", "Russkiy Mir", "Russian world", "Xi Jinping Thought", "Russian civilization"; "Chinese civilization"; "Russian culture"; "Chinese culture"; "Chinese communism", "order", "stability", "traditional values",  "anti-LGBTQ", "obedience", "strong leader", "hierarchy", "loyalty"                                                

Here is the text to categorize: 
"{text}"
"""

FOCAL_COUNTRY_PROMPT = """
Identify the focal country or region in the following post.

Return exactly one of these four labels:
- China
- Russia
- OECD
- other

If the post is mainly about China, return China.
If the post is mainly about Russia, return Russia.
If the post is mainly about the U.S., Europe, NATO, G7, EU, OECD countries, or the West broadly, return OECD.
Otherwise return other.

Text:
"{text}"
"""
# ----------------------------------------


# ---------- server discovery ----------
_clients = []

from openai import BadRequestError

def _call_model_with_truncation(client, prompt_template: str, text: str, max_tokens: int, cutoffs):
    last_err = None
    for max_chars in cutoffs:
        try_text = _truncate_text(text, max_chars)
        prompt = prompt_template.format(text=try_text)
        try:
            return _call_model(client, prompt, max_tokens=max_tokens)
        except BadRequestError as e:
            msg = str(e)
            if "exceeds the available context size" in msg or "exceed_context_size_error" in msg:
                last_err = e
                continue
            raise
    if last_err:
        raise last_err
    return ""

def _is_port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False

def _discover_servers(host: str, base_port: int, max_ports: int):
    ports = []
    for p in range(base_port, base_port + max_ports):
        if _is_port_open(host, p):
            ports.append(p)
    return ports

def init_clients():
    global _clients
    ports = _discover_servers(BASE_HOST, BASE_PORT, MAX_PORTS_TO_SCAN)
    if not ports:
        print(
            f"ERROR: No llama-server instances found on {BASE_HOST}:{BASE_PORT}-{BASE_PORT+MAX_PORTS_TO_SCAN-1}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Discovered servers on ports: {ports}")
    _clients = [OpenAI(base_url=f"http://{BASE_HOST}:{p}/v1", api_key=API_KEY) for p in ports]

def _get_client_for_row(idx: int):
    if not _clients:
        raise RuntimeError("Clients not initialized; call init_clients() first.")
    return _clients[idx % len(_clients)]


# ---------- helpers ----------
def _parse_json_array(text: str):
    text = text.strip()
    if not text:
        return ""
    try:
        arr = json.loads(text)
        if isinstance(arr, list) and arr:
            return str(arr[0]).strip()
    except Exception:
        pass
    m = re.search(r"\[(.*?)\]", text, flags=re.S)
    if m:
        inside = m.group(1).split(",")
        return inside[0].strip().strip('"').strip("'")
    return text.splitlines()[0].strip()

def _call_model(client, prompt: str, max_tokens: int = 24) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"cache_prompt": True, "top_p": 0.0},
    )
    out = resp.choices[0].message.content.strip()
    return _parse_json_array(out)

def _is_no_category(val) -> bool:
    s = str(val).strip().lower()
    return s.startswith("no categor")

def _needs_any_label(row) -> bool:
    return any(not _is_no_category(row[c]) for c in [AUT_LABEL_COL, DEM_LABEL_COL, WEST_LABEL_COL])

def _needs_aut_subcat(row) -> bool:
    return not _is_no_category(row[AUT_LABEL_COL])

def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."

def enrich_row(row, row_idx: int):
    client = _get_client_for_row(row_idx)
    text = str(row[TEXT_COL])

    out = {c: "" for c in OUT_COLS}

    if _needs_aut_subcat(row):
        out["AUT_SUBCATEGORY"] = _call_model_with_truncation(
            client,
            AUT_SUBCATEGORY_PROMPT,
            text,
            max_tokens=24,
            cutoffs=[2000, 1500, 1000, 700, 400],
        )

    if _needs_any_label(row):
        out["FOCAL_COUNTRY"] = _call_model_with_truncation(
            client,
            FOCAL_COUNTRY_PROMPT,
            text,
            max_tokens=8,
            cutoffs=[2500, 1800, 1200, 800, 400],
        )

    return out
def _determine_resume_index(out_csv, out_cols):
    if not os.path.exists(out_csv):
        return 0
    try:
        out_df = pd.read_csv(out_csv, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if all(col in out_df.columns for col in out_cols):
            done = pd.Series(True, index=out_df.index)
            for col in out_cols:
                # treat row as complete if both new columns are present;
                # blank is allowed for AUT_SUBCATEGORY when AUT_LABEL is No Category,
                # so resume logic just uses row count if columns exist.
                done &= out_df.index >= 0
            return len(out_df)
        return len(out_df)
    except Exception:
        with open(out_csv, "r", encoding="utf-8-sig") as f:
            return max(0, sum(1 for _ in f) - 1)




def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV, e.g. labeled_aut_dem_west.csv")
    ap.add_argument("--out", default="labeled_aut_dem_west_enriched.csv", help="Output CSV path")
    args = ap.parse_args()

    init_clients()

    df = pd.read_csv(
        args.csv,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines="skip",
    )

    required = [TEXT_COL, AUT_LABEL_COL, DEM_LABEL_COL, WEST_LABEL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(2)

    N = len(df)
    print(f"Loaded rows: {N:,}")

    start = _determine_resume_index(args.out, OUT_COLS)
    start = max(0, min(start, N))
    print(f"Resuming at row {start}/{N}")

    header = list(df.columns)
    for c in OUT_COLS:
        if c not in header:
            header.append(c)

    exists = os.path.exists(args.out)
    f = open(args.out, "a", encoding="utf-8-sig", newline="")
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

    try:
        pbar = tqdm(total=N - start, desc="Enriching", smoothing=0.05)
        flush_every = 10
        count = 0

        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            next_i = start
            in_flight = {}

            # seed initial batch
            while next_i < N and len(in_flight) < PARALLEL:
                fut = ex.submit(enrich_row, df.iloc[next_i], next_i)
                in_flight[fut] = next_i
                next_i += 1

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)

                for fut in done:
                    i = in_flight.pop(fut)
                    row = {col: df.iloc[i][col] for col in df.columns}

                    try:
                        extra = fut.result()
                    except Exception as e:
                        print(f"[error] row {i} failed: {e}", file=sys.stderr)
                        traceback.print_exc()
                        extra = {c: "" for c in OUT_COLS}

                    row.update(extra)
                    writer.writerow(row)

                    count += 1
                    if count % flush_every == 0:
                        f.flush()
                        os.fsync(f.fileno())
                        count = 0

                    pbar.update(1)
                    tqdm.write(
                        f"[{i+1}/{N}] "
                        f"AUT_LABEL={row.get('AUT_LABEL','')} | "
                        f"DEM_LABEL={row.get('DEM_LABEL','')} | "
                        f"WEST_LABEL={row.get('WEST_LABEL','')} | "
                        f"AUT_SUBCATEGORY={row.get('AUT_SUBCATEGORY','')} | "
                        f"FOCAL_COUNTRY={row.get('FOCAL_COUNTRY','')}"
                    )

                    # submit one new task for each completed one
                    if next_i < N:
                        new_fut = ex.submit(enrich_row, df.iloc[next_i], next_i)
                        in_flight[new_fut] = next_i
                        next_i += 1

        pbar.close()
        f.flush()
        os.fsync(f.fileno())

    finally:
        f.close()

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
