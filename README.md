# dem

Small Python scripts for labeling political framing in CSV data, then auditing and analyzing the results.

## What is here

- `DI_framing.py`: unified labeling entry point with `--type` and optional `--focal`.
- `di_framing_label_focal.py`: older V2 labeling and audit helper flow.
- `audit.py`: compares predictions against audit labels.
- `analysis.py`: distribution and association analysis on labeled outputs.
- `prompts.py`: shared prompt definitions for the unified labeling flow.

## Local development

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The labeling scripts read settings from environment variables:

```bash
source .env
python3 DI_framing.py --csv input.csv --type aut dem west --out labeled.csv
```

Examples:

```bash
python3 DI_framing.py --csv input.csv --type aut
python3 DI_framing.py --csv input.csv --type aut west --focal
python3 DI_framing.py --csv input.csv --type dem,west --col body_text
python3 DI_framing.py --csv input.csv --type aut dem west --prompt-version 1
```

If `LLM_BASE_URL` is set, the scripts use that hosted OpenAI-compatible endpoint.
If `LLM_BASE_URL` is empty, they fall back to probing local or remote `llama.cpp`-style servers starting at `LLM_BASE_PORT`.

## Environment variables

- `LLM_API_KEY`: placeholder is fine for local `llama.cpp`-style servers.
- `OPENAI_API_KEY` / `OPENROUTER_API_KEY`: supported as fallbacks when `LLM_API_KEY` is unset.
- `LLM_BASE_URL`: hosted OpenAI-compatible endpoint such as OpenAI or OpenRouter.
- `LLM_MODEL`: model name exposed by your server.
- `LLM_PARALLEL`: worker count for concurrent labeling requests.
- `LLM_TEMPERATURE`: generation temperature.
- `LLM_TOP_P`: generation top-p.
- `LLM_BASE_HOST`: host for model server discovery.
- `LLM_BASE_PORT`: first port to probe.
- `LLM_MAX_PORTS_TO_SCAN`: number of sequential ports to probe.

## Running analysis

```bash
python3 analysis.py --csv labeled.csv
python3 audit.py --csv audit_results.csv
```

## Develop here, run there

Recommended workflow:

1. Develop and test script changes locally on a small CSV sample.
2. Commit changes in this repo.
3. Push to your Git remote or sync the repo to the GPU server with `rsync`.
4. On the GPU server, create the venv once, pull/sync updates, load the right env vars, and run the long jobs there.

Example sync:

```bash
rsync -av --exclude '.venv' --exclude '.git' ./ "$REMOTE_HOST:$REMOTE_APP_DIR/"
```

Or use the helper script:

```bash
source .env
bash scripts/sync_to_remote.sh
```

Example remote run:

```bash
ssh "$REMOTE_HOST"
cd ~/dem
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source .env
python3 DI_framing.py --csv input.csv --type aut dem west --out labeled.csv
```
