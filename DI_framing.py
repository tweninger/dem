#!/usr/bin/env python3
# pip install openai>=1.40 pandas tqdm

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd, json, re, os, sys, csv, socket
from tqdm import tqdm

# ---------------- CONFIG ----------------

API_KEY  = "sk-no-key"                  # any non-empty string
MODEL    = "ggpl"                       # must match your server model
PARALLEL = 8                            # concurrent rows

BASE_HOST = "localhost"
BASE_PORT = 8080                        # first llama-server port
MAX_PORTS_TO_SCAN = 8                  # 8080..8095
PROMPT_TEMPLATES = {
    "AUT": 'AUT prompt for: "{texts}"',
    "DEM": 'DEM prompt for: "{texts}"',
    "WEST": 'WEST prompt for: "{texts}"',
}
LABEL_COLS = ["AUT_LABEL", "DEM_LABEL", "WEST_LABEL"]
MAX_CHARS = 1000                        # safety truncation

# Three distinct prompt templates; customize these as needed.
PROMPT_TEMPLATES = {
    "AUT": """
You are a senior political scientist analyzing social media posts. Your task is to classify the following social media post, which can be in any language (including Russian, Chinese, English, Arabic, etc.), into a specific category.
    Your response must be either one of the categories below in the format "Category Name". Do not add any explanations, introductory text, or quotation marks.
	
    **Context and Setting:** Consider the setting or context in which topics are discussed. Identify text that reference or discuss the promotion of autocracy---either directly or indirectly---through praise, justification, or support for authoritarian governance, efficiency, or economic success tied to authoritarian models. 
    **Identify the Primary Focus:** Determine the main subject matter of the text. Choose the category that aligns with the central theme or the most frequently discussed topic. Categorize texts that mention support or endorse authoritarian leaders or regimes explicitly or implicitly through related ideas or keywords.
	
    **Categories Defined:**
    * Authoritarian - Military/Security:  Post mentions military influence, military cooperation, and military strength of China or Russia, or other authoritarian regimes. Keywords: "military support", "security cooperation", "bloc", "military training"; “military prowess”; “military strength”, “military cooperation”
    * Authoritarian - Economic Influence: Post mentions economic influence and/or cooperation of China or Russia, or other authoritarian regimes. Keywords: "foreign aid", "Belt and Road", "BRI", "South-South", "global South", "development aid", "economic partnership", "economic cooperation", "foreign direct investment", "FDI", “trade”
    * Authoritarian - Digital: Posts mention the use of technology and/or digital tools to monitor the public. Keywords: "surveillance", "facial recognition", "censorship", "firewall", "social credit", "biometric"
    * Authoritarian - Legal Tools for Entrenchment: Post mentions legal tools or strategies used to protect authoritarian regimes or entrench their leaders. Keywords: "anti-terror law", "national security law", "emergency powers", "constitutional change"
    * Authoritarian - Alliances: Post discusses international alliances and partnerships of China or Russia, or other authoritarian regimes. Keywords: "Shanghai Cooperation Organization", "SCO", "BRICS", "EAEU", "Eurasian Economic Union", "GCC", "Gulf Cooperation Council", "strategic partnership", "multipolar", "Collective Security Treaty Organization", "CSTO", "Alliance of Sahel States", "AES", "Arab League" "Russia-[another country]", "China-[another country]", "alliance"
    * Authoritarian - Ideological Promotion: Post promotes the ideology of China or Russia, or other authoritarian regimes. Post discusses authoritarian or anti-liberal values. Keywords: "socialism with Chinese characteristics", "national rejuvenation", "Chinese model", "Russian model", "Third Rome", "Russkiy Mir", "Russian world", "Xi Jinping Thought", “Russian civilization”; “Chinese civilization”; “Russian culture”; “Chinese culture”; “Chinese communism”, "order", "stability", "traditional values",  "anti-LGBTQ", "obedience", "strong leader", "hierarchy", "loyalty"                                                
    * Categorize as just "No Categor" if the text does not belong to any of the mentioned categories.

Here is the text to categorize: 
    "{texts}"
""",
    "DEM": """
You are a senior political science researcher analyzing social media posts. Your task is to classify the following social media post, which can be in any language (including Russian, Chinese, English, Arabic, etc.), into a specific category.
    Your response must be either one of the categories below in the format "Category Name". Do not add any explanations, introductory text, or quotation marks.
    **Context and Setting:** Consider the setting or context in which topics are discussed. 
    **Identify the Primary Focus:** Determine the main subject matter of the text. Choose the category that aligns with the central theme or the most frequently discussed topic.
    
**Categories Defined:**
    * Democracy - Values and Rights : Discusses democratic principles, values, or rights. Keywords: "democracy", "liberalism", "pluralism", "equality", "tolerance", "representation", "minority rights", "rule of law", "checks and balances", "freedom", "rights", "liberty", "freedom of speech", "freedom of press", "freedom of expression", "freedom of religion", "freedom of assembly", "human rights", "civil rights".
    * Democracy - Elections: Focuses on the process of voting and elections. Keywords: "elections", "vote", "voting", "ballot", "voter", "turnout".
    * Democracy - Institutions: Refers to the governmental bodies of a democracy that check executive power in a country Keywords: "parliament", "congress", "legislature", "courts".
    * Democracy - Civil Society: Mentions non-governmental organizations and citizen groups. Keywords: "civil society", "NGO", "community organizations", "social movements", "social capital".
    * Categorize as just "No Category" if the text does not belong to any of the mentioned categories. 
    
	Here is the text to categorize: 
    "{texts}"
""",
    "WEST": """
WEST prompt. Do the WEST thing for this text:
 You are a senior political scientist analyzing social media posts. Your task is to classify the following social media post, which can be in any language (including Russian, Chinese, English, Arabic, etc.), into a specific category.
    Your response must be either one of the categories below in the format "Category Name". Do not add any explanations, introductory text, or quotation marks.
    
	**Context and Setting:** Consider the setting or context in which topics are discussed. Identify text that reference or discuss Western interference---either directly or indirectly---through accusations, implications, or criticism of Western involvement in political, economic, cultural, or social affairs of other countries.
    
	**Identify the Primary Focus:** Determine the main subject matter of the text. Choose the category that aligns with the central theme or the most prominent accusation or narrative related to Western influence or intervention. Include posts that employ explicit accusations, implied criticism, or metaphorical references to Western meddling or domination.
	
    **Categories Defined:**
    * WI - Declining West: Post mentions the political, economic, social injustice, protests, or moral decline of Western countries or liberal democracies: "decadent West", "Western decline" ; "instablity in the West"' , "woke"; "cancel cultural" , "moral crisis", "decline of living standard in the West"; "gun violence"; "school shooting"; "fentanyl crisis", "opioid crisis"
    * WI - Western induced Regime Change/Internal Instability: ”Color Revolution", "Orange Revolution", "Euromaidan", "Maidan", "Arab Spring", "coup", "5th Column", "foreign agent", "foreign meddling"
    * WI - Hostile Global Order: "hegemon", "hegemony", "imperialism", "colonialism", “NATO expansionism”, “violations of sovereignty”, “Western sanctions”, “Western agenda”, "Anti-China", "Anti-Russia", "Russophobia", “Sinophobia”, “Cold War mentality”, “unipolar”
    * WI - Specific Adversary Framing: "collective West", "US-West", "US-led West" (美西方), “Western hypocrisy”, “Western double-standard”, “pretty country” (漂亮国)
    * Categorize as just "No Category" if the text does not belong to any of the mentioned categories.
    
	Here is the text to categorize:
    "{texts}"
""",
}

_clients = []  # list[OpenAI]

def _is_port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    """Return True if TCP host:port accepts a connection."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False

def _discover_servers(host: str, base_port: int, max_ports: int):
    """Find running llama-servers on consecutive ports starting at base_port."""
    ports = []
    for p in range(base_port, base_port + max_ports):
        if _is_port_open(host, p):
            ports.append(p)
    return ports

def init_clients():
    """Initialize global client list based on running servers."""
    global _clients
    ports = _discover_servers(BASE_HOST, BASE_PORT, MAX_PORTS_TO_SCAN)
    if not ports:
        print(f"ERROR: No llama-server instances found on {BASE_HOST}:{BASE_PORT}-{BASE_PORT+MAX_PORTS_TO_SCAN-1}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Discovered servers on ports: {ports}")
    _clients = [
        OpenAI(base_url=f"http://{BASE_HOST}:{p}/v1", api_key=API_KEY)
        for p in ports
    ]

def _get_client_for_row(idx: int):
    """Simple deterministic round-robin: choose client by row index."""
    if not _clients:
        raise RuntimeError("Clients not initialized; call init_clients() first.")
    return _clients[idx % len(_clients)]
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

def _truncate(text: str) -> str:
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS]
    return text

def label_texts(single_text, row_idx: int):
    """
    Call the model three times (AUT/DEM/WEST) for a single text,
    picking a server in round-robin based on row index.
    """
    client = _get_client_for_row(row_idx)
    text = _truncate(str(single_text))
    results = {}

    for key, tmpl in PROMPT_TEMPLATES.items():
        prompt = tmpl.format(texts=text, text=text)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=16,
            extra_body={"cache_prompt": True, "top_p": 0.0},
        )
        out = resp.choices[0].message.content.strip()
        label = _parse_json_array(out)
        results[f"{key}_LABEL"] = label

    return results
def _determine_resume_index(in_csv, out_csv, label_cols):
    """
    Figure out where to resume based on existing output.
    A row is considered 'done' only if all label_cols are non-empty.
    """
    if not os.path.exists(out_csv):
        return 0
    try:
        out_df = pd.read_csv(out_csv, encoding="utf-8-sig")
        if all(col in out_df.columns for col in label_cols):
            done = pd.Series(True, index=out_df.index)
            for col in label_cols:
                done &= out_df[col].notna() & (out_df[col].astype(str) != "")
            return int(done.sum())
        return len(out_df)
    except Exception:
        # Fallback: assume every existing line (minus header) is a completed row
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
    ap.add_argument("--out", default="labeled_aut_dem_west.csv", help="Output CSV path")
    ap.add_argument("--col", default="text", help="Column to label")
    args = ap.parse_args()

    # Discover llama-servers and init OpenAI clients
    init_clients()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    print("Loaded rows:", len(df), "cols:", len(df.columns))
    if args.col not in df.columns:
        print(f"ERROR: Column '{args.col}' not found in {args.csv}", file=sys.stderr)
        sys.exit(2)

    texts = df[args.col].astype(str).tolist()
    N = len(texts)
    start = _determine_resume_index(args.csv, args.out, LABEL_COLS)
    start = max(0, min(start, N))
    print(f"Resuming at row {start}/{N}")

    header = list(df.columns) + LABEL_COLS
    f, writer = _open_output_for_append(args.out, header)

    try:
        pbar = tqdm(total=N - start, desc="Labeling", smoothing=0.05)
        flush_every = 10
        count = 0

        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futures = {
                ex.submit(label_texts, texts[i], i): i
                for i in range(start, N)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    labels = fut.result()
                except Exception as e:
                    print(f"[error] row {i} failed: {e}", file=sys.stderr)
                    labels = {col: "" for col in LABEL_COLS}

                row = {col: df.iloc[i][col] for col in df.columns}
                for col in LABEL_COLS:
                    row[col] = labels.get(col, "")
                writer.writerow(row)
                count += 1
                if count % flush_every == 0:
                    f.flush()
                    os.fsync(f.fileno())
                    count = 0

                pbar.update(1)
                print(
                    f"[{i+1}/{N}] AUT={row.get('AUT_LABEL','')} | DEM={row.get('DEM_LABEL','')} | WEST={row.get('WEST_LABEL','')}",
                    flush=True,
                )

        pbar.close()
        # final flush
        f.flush()
        os.fsync(f.fileno())
    finally:
        f.close()

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
