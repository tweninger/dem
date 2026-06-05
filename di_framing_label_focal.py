#!/usr/bin/env python3

from openai import OpenAI, BadRequestError
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import pandas as pd
import json, re, os, sys, csv, socket, traceback
from tqdm import tqdm

# ---------------- CONFIG ----------------
API_KEY  = "sk-no-key"
MODEL    = "ggpl"
PARALLEL = 8

BASE_HOST = "127.0.0.1"
BASE_PORT = 8080
MAX_PORTS_TO_SCAN = 8

TEXT_COL = "text"

TASKS = ["AUT", "DEM", "WEST"]

LABEL_COLS_V2 = [
    "AUT_LABEL_V2",
    "DEM_LABEL_V2",
    "WEST_LABEL_V2",
    "FOCAL_COUNTRY_V2",
]

NO_CATEGORY = "No Category"


PROMPT_TEMPLATES_V2 = {
    "AUT": """
You are a senior political scientist coding geopolitical framing in social media posts.

The post may be in any language. 

Identify social media posts that explicitly or implicitly endorse, justify, or positively evaluate authoritarian governance systems, leaders, or non-democratic political models.
The key requirement is normative support for authoritarian rule, not mere discussion of authoritarian countries or outcomes.

General Rule

A post must demonstrate positive framing of authoritarian governance as: legitimate, effective, desirable, or superior to democratic alternatives.

Return exactly one label from this list:

Military/Security Promotion
Economic Influence
Digital Control and Surveillance
Legal Entrenchment
Alliances
Ideological Promotion
No Category

Coding rules:

1. Military/Security Promotion
Code posts in this category ONLY when the post explicitly praises, supports, promotes, or advocates the military power, military strength, or security cooperation of authoritarian states such as China or Russia.
The post must express approval of authoritarian military strength, strategic alliances, security doctrines, or geopolitical confrontation with liberal democratic states or institutions.
Include posts that:
* support military cooperation, alliances, or security partnerships involving authoritarian regimes;
* praise the military strength authoritarian states;
* glorify military modernization or coercive power;
* portray authoritarian military systems as effective, stabilizing, or necessary;
* celebrate demonstrations of military force, discipline, or national defense capability;
* frame NATO, the United States, or liberal democracies as threats requiring authoritarian military response.
Strong Indicators (when used approvingly)
* military cooperation
* security cooperation
* strategic partnership
* multipolar security order
* military strength
* military prowess
* joint military exercises
* defense sovereignty
* anti-Western alliance
* security bloc
* military modernization
* deterrence against the West

2. Economic Influence 
Code posts in this category ONLY when the post frames economic activity by China, Russia, or other authoritarian states as a tool of geopolitical influence, strategic alignment, or soft power expansion.
The key requirement is a strategic or political interpretation of economic activity, not mere discussion of trade or investment.
Include posts that:
* portray aid, investment, or trade as instruments of geopolitical influence;
* frame initiatives like Belt and Road as strategic expansion of authoritarian power;
* describe economic partnerships and cooperation as aligning countries into political blocs;
* portray China or Russia as using economic tools to expand global influence;
* frame Global South partnerships as part of a geopolitical strategy against the West;
Strong Indicators (only when used in strategic framing)
* Belt and Road / BRI
* South-South cooperation
* Global South (when used geopolitically)
* development aid (when framed as influence tool)
* economic partnership
* economic cooperation
* foreign direct investment / FDI
* infrastructure investment

3. Digital Control and Surveillance
Code posts ONLY when digital technologies are framed as tools to monitor the population, censorship, or behavioral regulation, particularly in authoritarian governance contexts.
The focus is on technology used for political control, not technology itself.
* describe surveillance technologies as tools for monitoring or controlling populations;
* portray censorship systems as protecting citizens;
* frame biometric systems, facial recognition, or data tracking as governance tools for social control;
* describe social credit systems or similar mechanisms as regulating citizen behavior;
* portray digital infrastructure (firewalls, platforms, algorithms) as tools of state information control;
* link technology explicitly to authoritarian governance, repression, or behavioral enforcement.
Strong Indicators (require control framing)
* surveillance
* facial recognition
* censorship
* firewall / Great Firewall
* social credit system
* biometric systems
* digital tracking
* data monitoring

4. Legal Entrenchment
Code posts in this category ONLY when the post explicitly supports, justifies, promotes, or positively frames the use of legal, constitutional, or emergency measures to consolidate, expand, or protect authoritarian political power.
This includes laws, legal doctrines, constitutional reforms, or emergency powers used to:
* weaken political opposition,
* restrict civil liberties,
* centralize executive authority,
* suppress dissent,
* extend leadership tenure,
* or protect regime stability at the expense of democratic accountability or political pluralism.
Include posts that:
* justify restrictions on speech, protest, media, or political opposition in the name of security, order, or stability;
* support emergency powers that expand executive authority;
* endorse constitutional changes that extend or entrench leadership power;
* defend legal crackdowns on dissent, separatism, extremism, or foreign influence;
* portray authoritarian legal controls as necessary for national unity, stability, or sovereignty;
* support political bans or legal repression as legitimate governance tools;
* frame legal restrictions on rights as necessary to preserve social order or regime security.
Strong Indicators (when used approvingly)
The following are stronger indicators when framed positively or supportively:
* national security law
* anti-terror law
* foreign agents law
* extremism law
* emergency powers
* constitutional reform
* constitutional change
* stability maintenance
* anti-separatism law
* state security
* social order
* security over chaos

5. Alliances: 
Code posts ONLY when international alliances or partnerships involving authoritarian states are framed as strategic blocs with geopolitical purpose, alignment, or power projection.
The focus is not on the existence of alliances, but on their political meaning as coordinated geopolitical structures.
Include posts that:
* frame alliances (e.g., BRICS, SCO, CSTO) as alternatives to Western-led global order;
* describe alliances as instruments of multipolarity or geopolitical restructuring;
* portray partnerships between authoritarian states as coordinated strategic blocs;
* interpret alliances as mechanisms to counterbalance Western power or institutions;
* emphasize ideological or geopolitical alignment among member states;
* describe emerging blocs as reshaping global governance
Strong indicators (only when framed geopolitically):
* BRICS
* SCO / Shanghai Cooperation Organization
* CSTO / Collective Security Treaty Organization
* EAEU / Eurasian Economic Union
* AES / Alliance of Sahel States
* Arab League 
* multipolar world
* strategic partnership
* bloc formation

6. Authoritarian - Ideological Promotion: 
Code posts in this category ONLY when the post explicitly endorses, glorifies, defends, or advocates Chinese or Russian political systems, leaders, or ideological values associated with them or similar political systems.
The post must express clear approval of Chinese or Russian governance, anti-liberal political values, or civilizational narratives that justify centralized political authority, political conformity, or restrictions on pluralism and individual rights.
Include posts that:
* praise centralized rule, or one-party systems;
* portray Russian or Chinese political systems as superior to liberal democracy;
* reject liberal democracy, pluralism, or individual rights in favor of political order, hierarchy, or collective unity;
* endorse official ideological doctrines, slogans
* support strong leaders as necessary for national survival, unity, or civilizational strength;
* frame Western liberalism as decadent, corrupt, weak, or dangerous while positively contrasting Chinese or Russian alternatives;
* advocate political conformity, obedience, hierarchy, nationalism, or civilizational exceptionalism;
* defend restrictions on dissent, opposition, or minority rights as necessary for national stability or moral order.
The following phrases are stronger indicators when used positively or aspirationally:
* socialism with Chinese characteristics
* Xi Jinping Thought
* national rejuvenation
* Russkiy Mir
* Russian civilization
* Chinese civilization
* Chinese model
* Russian model
* civilizational state
* strong leadership
* national unity over individual rights
* traditional values against Western decadence
* Chinese communism
* anti-LGBTQ

7. Return "No Category" if the text does not belong to any of the mentioned categories.


Post:
"{text}"
""",

    "DEM": """
You are a senior political scientist coding geopolitical framing in social media posts.

The post may be in any language. 

Identify social media posts that meaningfully discuss democracy, democratic values, institutions, elections, or civil society as political or normative concepts.

A post should only be coded if democracy-related concepts are central to its meaning, not merely mentioned.

Return exactly one label from this list:

Values and Rights
Elections
Institutions
Civil Society
No Category

Coding rules:

1. Values and Rights
Code posts ONLY when they mention democratic principles, rights, or civil liberties commonly associated with liberal democracies. Include posts about:
* civil liberties (speech, press, religion, assembly);
* rule of law, checks and balances
* pluralism, tolerance, minority rights, equality
* liberalism, human rights

2. Elections 
Code posts ONLY when elections or voting are discussed as mechanisms of democratic political participation, representation, or legitimacy. Include posts that:
* describe elections as a mechanism of democratic representation or accountability;
* emphasize voter participation, turnout, or civic engagement in democratic elections;
* highlight electoral processes;
* discuss voting as a fundamental democratic right or civic duty
Strong Indicators:
* elections (when linked to democracy/legitimacy)
* voting / vote / voter / turnout (when civic or democratic in meaning)
* ballot (when linked to democratic participation)
* electoral participation
* free and fair elections
* voter rights

3. Institutions 
Includes posts where institutions (parliament, courts, legislature) are discussed as mechanisms of democratic accountability, constraint, or governance balance. The focus is on institutions in a democratic political system Include posts that:
* describe parliament, congress, or legislature as constraining executive authority;
* emphasize judicial independence or courts limiting government power;
* highlight checks and balances between branches of government;
* frame institutions as safeguards of democracy or rule of law;
Strong Indicators (require functional framing)
* parliament / congress / legislature (when linked to oversight or constraint)
* courts / judiciary (when framed as independent constraint)
* checks and balances
* separation of powers
* constitutional oversight
* judicial review
* institutional accountability

4. Civil Society 
Includes posts where NGOs, social movements, or civic groups are described as independent actors contributing to democratic participation, accountability, or pluralism. 
Include posts that:
* describe NGOs, social movements, or community organizations as independent actors advocating for rights, accountability, or democratic reform;
* portray civil society as a counterbalance to government power;
* emphasize citizen participation, grassroots organizing, or civic engagement in governance;
Strong Indicators (require democratic/civic framing)
* civil society (when independent and politically engaged)
* NGOs (when autonomous and advocacy-oriented)
* social movements (when civic or political in nature)
* community organizations (when linked to participation or accountability)
* social capital (when tied to democratic participation or trust-building)

5. Return "No Category" if the text does not belong to any of the mentioned categories.

Post:
"{text}"
""",

    "WEST": """
You are a senior political scientist coding geopolitical framing in social media posts.

The post may be in any language. Identify social media posts that construct, endorse, or reproduce narratives that portray Western states, institutions, or allied actors as interfering in the political, economic, cultural, or social affairs of other countries.
The key requirement is narrative framing of Western interference, not mere mention of geopolitical terms.
General Rule
A post should be labeled ONLY when it frames Western actors as:
actively interfering in other countries' internal affairs, OR
forming a coordinated geopolitical system of influence, OR
constructing adversarial or exploitative global power relations.
Mentions alone are NOT sufficient.

Return exactly one label from this list:

Declining West
Western induced Regime Change/Internal Instability
Hostile Global Order
Specific Adversary Framing
No Category

Coding rules:

1. Declining West
Code posts in this category ONLY when the post frames Western countries or liberal democracies as being in systemic civilizational, moral, social, or political decline.
The key requirement is that the post interprets Western problems as evidence of structural or civilizational failure, not isolated policy issues.
Include posts that:
* portray the West as morally decaying or culturally degraded;
* describe Western societies as collapsing, failing, or in irreversible decline;
* interpret social problems (crime, drugs, protests, inequality) as evidence of systemic Western collapse;
* frame liberal democracy as producing disorder, chaos, or moral breakdown;
* use civilizational narratives of "decline," "decadence," or "end of the West";
* contrast Western decline with implied non-Western stability or superiority.
Strong Indicators (only when used in a decline narrative)
* decadent West
* Western decline
* moral crisis
* collapse of the West
* woke culture (when framed as civilizational decay)
* cancel culture (when framed as societal breakdown)
* decline of living standards in the West (when used as systemic failure)
* gun violence epidemic
* opioid crisis
* fentanyl crisis
* social instability in the West

2. Western induced Regime Change/Internal Instability
Code posts in this category ONLY when the post claims, implies, or endorses the idea that Western governments, institutions, intelligence services, or their associates intentionally promote regime change, political unrest, protests, coups, separatism, or internal instability in another country.
The post must frame domestic unrest or political opposition as externally orchestrated, manipulated, funded, or exploited by Western actors.
Include posts that:
* portray protests, revolutions, or opposition movements as Western-backed operations;
* claim that the United States, NATO, the EU, or foreign actors are destabilizing another country;
* frame domestic dissent as manipulated by foreign powers;
* accuse activists, journalists, opposition figures, or civil society organizations of acting as foreign agents or proxies;
* describe regime change efforts as part of Western strategy;
* characterize democratic uprisings as artificial, externally coordinated, or illegitimate;
* claim that Western influence threatens sovereignty, stability, or national unity.
Strong Indicators (when used supportively or affirmatively)
* Color Revolution
* Orange Revolution
* Euromaidan
* Maidan
* foreign agents
* foreign meddling
* Western interference
* Western-backed coup
* CIA-backed
* NGO interference
* 5th column
* external destabilization
* hybrid warfare
* manufactured protests

3. Hostile Global Order Framing
Code posts in this category ONLY when the post frames the international system as dominated by a coercive, unjust, or adversarial global order led by Western powers (or their allies), or when it portrays global politics as structured by systemic Western domination, containment, or ideological hostility toward other states.
Include posts that:
* portray the global system as dominated by Western hegemony or imperial control;
* frame NATO, the US, or Western alliances as expansionist or coercive global actors;
* describe international relations as structured by containment, suppression, or ideological hostility toward non-Western states;
* claim that sanctions, diplomacy, or institutions are tools of Western domination;
* depict international norms as imposed by a Western-led “unipolar” system;
* frame China, Russia, or other states as victims of systemic geopolitical hostility;
* interpret global conflicts as expressions of systemic Western power projection.
Strong Indicators (only when used in adversarial/systemic framing)
* hegemon / hegemony
* imperialism
* colonialism (when used for contemporary geopolitical critique)
* unipolar / unipolar world
* Cold War mentality
* NATO expansion / NATO expansionism
* Western sanctions (when framed as coercive system tool)
* Western agenda
* violations of sovereignty (when attributed to systemic Western behavior)
* Russophobia
* Sinophobia
* anti-China / anti-Russia (when framed as systemic Western hostility)

4. Specific Adversary Framing
Code posts in this category ONLY when the post constructs “the West” (including the United States and its allies) as a unified geopolitical or civilizational bloc that behaves in a coordinated, hypocritical, or adversarial manner toward other countries or civilizations.
The post must frame the West as engaged in political, moral, or geopolitical double standards, hostility, or interference.
Include posts that:
* portray "the West" or "US-led West" as a single coordinated actor;
* describe Western countries as acting in bad faith, hypocrisy, or double standards;
* frame Western institutions (NATO, EU, US alliances) as unified tools of domination or interference;
* construct civilizational opposition between "the West" and "non-West";
* present Western criticism of others as hypocritical or illegitimate due to Western behavior;
* use adversarial civilizational language such as "collective West" in a hostile framing context.
Strong Indicators (only when used in adversarial framing)
* collective West
* US-led West
* US-West
* Western hypocrisy
* Western double standards
* Western hegemony
* Western imperialism
* beautiful country" / "pretty country" (漂亮国) when used sarcastically or derogatorily

5. Return "No Category" if the text does not belong to any of the mentioned categories.

Post:
"{text}"
""",
}


FOCAL_COUNTRY_PROMPT_V2 = """
Identify the primary country or geopolitical actor that is the central focus of the post. 

The focal country is the country whose actions, values, interests, leadership, or political system are the primary subject of evaluation or discussion.

The focal country is the country:
* most directly discussed,
* evaluated,
* criticized,
* praised,
* or portrayed as the main actor in the post.

The focal country should reflect the main subject of the post, not merely countries that are mentioned in passing.

Return just the short name of a country or group of countries. For example: USA, China, Russia, OECD, NATO, Europe, G7, England, Philippines.

Post:
"{text}"
"""


# ---------------- server discovery ----------------

_clients = []

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
            f"ERROR: No llama-server instances found on "
            f"{BASE_HOST}:{BASE_PORT}-{BASE_PORT + MAX_PORTS_TO_SCAN - 1}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Discovered servers on ports: {ports}")
    _clients = [
        OpenAI(base_url=f"http://{BASE_HOST}:{p}/v1", api_key=API_KEY)
        for p in ports
    ]

def _get_client_for_row(idx: int):
    if not _clients:
        raise RuntimeError("Clients not initialized; call init_clients() first.")
    return _clients[idx % len(_clients)]


# ---------------- helpers ----------------
def _parse_model_output(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    try:
        arr = json.loads(text)
        if isinstance(arr, list) and arr:
            return str(arr[0]).strip().strip('"').strip("'")
    except Exception:
        pass

    m = re.search(r"\[(.*?)\]", text, flags=re.S)
    if m:
        inside = m.group(1).split(",")
        return inside[0].strip().strip('"').strip("'")

    return text.splitlines()[0].strip().strip('"').strip("'")


def _normalize_label(x: str) -> str:
    s = str(x).strip()

    if not s:
        return ""

    low = s.lower().strip()

    if low.startswith("no categor"):
        return "No Category"

    return s

def _is_no_category(x: str) -> bool:
    return _normalize_label(x) == "No Category"

def _truncate_text(text: str, max_chars: int) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def _call_model(client, prompt: str, max_tokens: int = 32) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"cache_prompt": True, "top_p": 0.0},
    )
    return _normalize_label(_parse_model_output(resp.choices[0].message.content))


def _call_model_with_truncation(client, prompt_template: str, text: str, max_tokens: int, cutoffs):
    last_err = None

    for max_chars in cutoffs:
        try_text = _truncate_text(text, max_chars)
        prompt = prompt_template.format(text=try_text, texts=try_text)

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


def _has_any_content_label(labels: dict) -> bool:
    return any(
        not _is_no_category(labels.get(col, "No Category"))
        for col in ["AUT_LABEL_V2", "DEM_LABEL_V2", "WEST_LABEL_V2"]
    )

def label_row(row, row_idx: int, force_focal: bool = False):
    client = _get_client_for_row(row_idx)
    text = str(row.get(TEXT_COL, ""))

    out = {
        "AUT_LABEL_V2": "",
        "DEM_LABEL_V2": "",
        "WEST_LABEL_V2": "",
        "FOCAL_COUNTRY_V2": "",
    }

    for task in TASKS:
        label = _call_model_with_truncation(
            client=client,
            prompt_template=PROMPT_TEMPLATES_V2[task],
            text=text,
            max_tokens=32,
            cutoffs=[2500, 1800, 1200, 800, 400],
        )
        out[f"{task}_LABEL_V2"] = label

    if force_focal or _has_any_content_label(out):
        out["FOCAL_COUNTRY_V2"] = _call_model_with_truncation(
            client=client,
            prompt_template=FOCAL_COUNTRY_PROMPT_V2,
            text=text,
            max_tokens=8,
            cutoffs=[2500, 1800, 1200, 800, 400],
        )
    else:
        out["FOCAL_COUNTRY_V2"] = ""

    return out



# ============================================================
# RESUME AND ORDERED WRITING
# ============================================================

def _determine_resume_index(out_csv: str, label_cols: list[str]) -> int:
    if not os.path.exists(out_csv):
        return 0

    try:
        out_df = pd.read_csv(
            out_csv,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )

        if not all(c in out_df.columns for c in label_cols):
            return len(out_df)

        done = pd.Series(True, index=out_df.index)
        for col in ["AUT_LABEL_V2", "DEM_LABEL_V2", "WEST_LABEL_V2"]:
            done &= out_df[col].astype(str).str.strip().ne("")

        # Resume at the first incomplete row. If all are complete, resume at len.
        incomplete = done[~done]
        if len(incomplete) == 0:
            return len(out_df)
        return int(incomplete.index[0])

    except Exception:
        with open(out_csv, "r", encoding="utf-8-sig") as f:
            return max(0, sum(1 for _ in f) - 1)


def _open_output_for_append(path: str, header: list[str]):
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


def run_labeling(df: pd.DataFrame, out_csv: str, force_focal: bool = False):
    N = len(df)

    header = list(df.columns)
    for c in LABEL_COLS_V2:
        if c not in header:
            header.append(c)

    start = _determine_resume_index(out_csv, LABEL_COLS_V2)
    start = max(0, min(start, N))

    print(f"Loaded rows: {N:,}")
    print(f"Resuming at row {start:,}/{N:,}")

    f, writer = _open_output_for_append(out_csv, header)

    try:
        pbar = tqdm(total=N - start, desc="Labeling V2", smoothing=0.05)

        next_submit = start
        next_write = start
        in_flight = {}
        completed = {}

        flush_every = 10
        since_flush = 0

        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            while next_submit < N and len(in_flight) < PARALLEL:
                fut = ex.submit(label_row, df.iloc[next_submit], next_submit, force_focal)
                in_flight[fut] = next_submit
                next_submit += 1

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)

                for fut in done:
                    i = in_flight.pop(fut)

                    try:
                        labels = fut.result()
                    except Exception as e:
                        print(f"[error] row {i} failed: {e}", file=sys.stderr)
                        traceback.print_exc()
                        labels = {c: "" for c in LABEL_COLS_V2}

                    completed[i] = labels

                    if next_submit < N:
                        new_fut = ex.submit(label_row, df.iloc[next_submit], next_submit, force_focal)
                        in_flight[new_fut] = next_submit
                        next_submit += 1

                # Write rows strictly in input order.
                while next_write in completed:
                    labels = completed.pop(next_write)
                    row = {col: df.iloc[next_write][col] for col in df.columns}
                    row.update(labels)

                    writer.writerow(row)

                    since_flush += 1
                    if since_flush >= flush_every:
                        f.flush()
                        os.fsync(f.fileno())
                        since_flush = 0

                    pbar.update(1)

                    tqdm.write(
                        f"[{next_write + 1}/{N}] "
                        f"AUT={row.get('AUT_LABEL_V2', '')} | "
                        f"DEM={row.get('DEM_LABEL_V2', '')} | "
                        f"WEST={row.get('WEST_LABEL_V2', '')} | "
                        f"FOCAL={row.get('FOCAL_COUNTRY_V2', '')}"
                    )

                    next_write += 1

        pbar.close()
        f.flush()
        os.fsync(f.fileno())

    finally:
        f.close()

    print(f"Wrote {out_csv}")


# ============================================================
# AUDIT SUPPORT
# ============================================================

def _first_existing_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_audit_gold_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expected audit columns can be either:

    Existing columns from your current sheet:
    - sample_from
    - label_value
    - Label_value Correct?
    - Focal Country Correct?
    - notes

    Better optional gold columns:
    - GOLD_LABEL_VALUE
    - GOLD_FOCAL_COUNTRY

    If GOLD_LABEL_VALUE is absent:
    - If Label_value Correct? == Y, use label_value.
    - If Label_value Correct? == N, use notes as the corrected label.
      This works if notes contains the corrected category.
    """

    df = df.copy()

    gold_label_col = _first_existing_col(
        df,
        ["GOLD_LABEL_VALUE", "gold_label_value", "Correct Label", "correct_label"],
    )

    gold_focal_col = _first_existing_col(
        df,
        ["GOLD_FOCAL_COUNTRY", "gold_focal_country", "Correct Focal Country", "correct_focal_country"],
    )

    label_correct_col = _first_existing_col(
        df,
        ["Label_value Correct?", "Label Value Correct?", "label_value_correct"],
    )

    focal_correct_col = _first_existing_col(
        df,
        ["Focal Country Correct?", "focal_country_correct"],
    )

    if gold_label_col:
        df["AUDIT_GOLD_LABEL_VALUE"] = df[gold_label_col].map(_normalize_label)
    elif label_correct_col and "label_value" in df.columns:
        df["AUDIT_GOLD_LABEL_VALUE"] = df.apply(
            lambda r: _normalize_label(r["label_value"])
            if str(r[label_correct_col]).strip().upper() == "Y"
            else _normalize_label(r.get("notes", "")),
            axis=1,
        )
    else:
        df["AUDIT_GOLD_LABEL_VALUE"] = ""

    if gold_focal_col:
        df["AUDIT_GOLD_FOCAL_COUNTRY"] = df[gold_focal_col].map(_normalize_label)
    elif focal_correct_col and "FOCAL_COUNTRY" in df.columns:
        df["AUDIT_GOLD_FOCAL_COUNTRY"] = df.apply(
            lambda r: _normalize_label(r["FOCAL_COUNTRY"])
            if str(r[focal_correct_col]).strip().upper() == "Y"
            else "",
            axis=1,
        )
    else:
        df["AUDIT_GOLD_FOCAL_COUNTRY"] = ""

    return df

def get_pred_for_sample(row):
    """
    The audit sheet has sample_from values like:
    - AUT_LABEL
    - DEM_LABEL
    - WEST_LABEL

    This function compares the relevant V2 prediction to the hand-coded gold.
    """

    sample_from = str(row.get("sample_from", "")).strip().upper()

    if sample_from.startswith("AUT"):
        return row.get("AUT_LABEL_V2", "")
    if sample_from.startswith("DEM"):
        return row.get("DEM_LABEL_V2", "")
    if sample_from.startswith("WEST"):
        return row.get("WEST_LABEL_V2", "")

    # Fallback: if sample_from is missing, use the first non-No Category prediction.
    for c in ["AUT_LABEL_V2", "DEM_LABEL_V2", "WEST_LABEL_V2"]:
        val = row.get(c, "")
        if val and not _is_no_category(val):
            return val

    return "No Category"


def score_audit(out_df: pd.DataFrame, audit_report_csv: Optional[str] = None):
    df = out_df.copy()

    if "AUDIT_GOLD_LABEL_VALUE" not in df.columns:
        df = add_audit_gold_columns(df)

    df["AUDIT_PRED_LABEL_VALUE"] = df.apply(get_pred_for_sample, axis=1).map(_normalize_label)

    df["AUDIT_LABEL_CORRECT_V2"] = (
        df["AUDIT_PRED_LABEL_VALUE"].map(_normalize_label)
        == df["AUDIT_GOLD_LABEL_VALUE"].map(_normalize_label)
    )

    has_focal_gold = df["AUDIT_GOLD_FOCAL_COUNTRY"].astype(str).str.strip().ne("")
    df["AUDIT_FOCAL_CORRECT_V2"] = False
    df.loc[has_focal_gold, "AUDIT_FOCAL_CORRECT_V2"] = (
        df.loc[has_focal_gold, "FOCAL_COUNTRY_V2"].map(_normalize_label)
        == df.loc[has_focal_gold, "AUDIT_GOLD_FOCAL_COUNTRY"].map(_normalize_label)
    )

    print("\n================ AUDIT SUMMARY ================")

    print(f"Rows: {len(df):,}")

    if df["AUDIT_GOLD_LABEL_VALUE"].astype(str).str.strip().ne("").any():
        print(f"Frame/subcategory accuracy: {df['AUDIT_LABEL_CORRECT_V2'].mean():.3f}")

        print("\nGold x predicted frame/subcategory:")
        print(
            pd.crosstab(
                df["AUDIT_GOLD_LABEL_VALUE"],
                df["AUDIT_PRED_LABEL_VALUE"],
                dropna=False,
            )
        )
    else:
        print("Frame/subcategory accuracy not scored: no gold label column found.")

    if has_focal_gold.any():
        print(f"\nFocal country rows scored: {has_focal_gold.sum():,}")
        print(f"Focal country accuracy: {df.loc[has_focal_gold, 'AUDIT_FOCAL_CORRECT_V2'].mean():.3f}")

        print("\nGold x predicted focal country:")
        print(
            pd.crosstab(
                df.loc[has_focal_gold, "AUDIT_GOLD_FOCAL_COUNTRY"],
                df.loc[has_focal_gold, "FOCAL_COUNTRY_V2"],
                dropna=False,
            )
        )
    else:
        print("\nFocal country accuracy not scored: no corrected focal-country gold column found.")

    print("\nMost useful error rows:")
    error_cols = [
        "sample_from",
        "label_value",
        "AUDIT_GOLD_LABEL_VALUE",
        "AUDIT_PRED_LABEL_VALUE",
        "AUT_LABEL_V2",
        "DEM_LABEL_V2",
        "WEST_LABEL_V2",
        "AUDIT_GOLD_FOCAL_COUNTRY",
        "FOCAL_COUNTRY_V2",
        "text",
    ]
    error_cols = [c for c in error_cols if c in df.columns]

    errors = df.loc[~df["AUDIT_LABEL_CORRECT_V2"], error_cols].copy()
    print(errors.head(25).to_string(index=False))

    if audit_report_csv:
        df.to_csv(audit_report_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        print(f"\nWrote audit report: {audit_report_csv}")

    return df


def main():
    import argparse

    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True, help="Input CSV")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--col", default="text", help="Text column")
    ap.add_argument("--audit", action="store_true", help="Treat input as hand-coded audit sheet")
    ap.add_argument("--audit-report", default="", help="Optional audit report CSV")
    ap.add_argument(
        "--force-focal",
        action="store_true",
        help="Run focal-country prompt even when AUT/DEM/WEST are all No Category",
    )

    args = ap.parse_args()

    global TEXT_COL
    TEXT_COL = args.col

    init_clients()

    df = pd.read_csv(
        args.csv,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        engine="python",
        on_bad_lines="skip",
    )

    if TEXT_COL not in df.columns:
        print(f"ERROR: text column '{TEXT_COL}' not found in {args.csv}", file=sys.stderr)
        sys.exit(2)

    if args.audit:
        df = add_audit_gold_columns(df)

    run_labeling(df, args.out, force_focal=args.force_focal or args.audit)

    if args.audit:
        out_df = pd.read_csv(
            args.out,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        score_audit(
            out_df,
            audit_report_csv=args.audit_report if args.audit_report else None,
        )


if __name__ == "__main__":
    main()