# Gmail Scraper / Email Agent

Agent to help you classify and manage emails in your inbox using a local LLM (GPT4All). Runs on **MacBook Air M4** (Metal) and **PC with NVIDIA GPU** (e.g. RTX 3080, via CUDA).

## Setup

### 1. Clone and create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

**On MacBook (M1/M2/M3/M4):**

```bash
pip install -r requirements.txt
```

The app will use **Metal** for GPU-accelerated inference automatically.

**On PC with NVIDIA GPU (e.g. RTX 3080):**

```bash
pip install -r requirements.txt
pip install "gpt4all[cuda]"
```

Ensure [NVIDIA drivers](https://www.nvidia.com/drivers) and a compatible [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) are installed. The app will use **CUDA** for GPU inference on Windows/Linux when available.

**On PC without NVIDIA GPU (CPU only):**

```bash
pip install -r requirements.txt
```

### 3. Gmail API credentials

Place `credentials.json` (from [Google Cloud Console](https://console.cloud.google.com/) with Gmail API enabled) in the project root. On first run you’ll complete OAuth in the browser; a `token.json` will be created.

## Device override

To force a specific device (e.g. if auto-detection is wrong):

- **CPU only:** `set GPT4ALL_DEVICE=cpu` (Windows) or `export GPT4ALL_DEVICE=cpu` (macOS/Linux)
- **CUDA (NVIDIA):** `set GPT4ALL_DEVICE=cuda` / `export GPT4ALL_DEVICE=cuda`
- **Metal (Mac):** `export GPT4ALL_DEVICE=gpu`

## Usage

Use the Jupyter notebook `email_filter_agent_test.ipynb` to export emails, run the classification pipeline, and (optionally) delete emails from your inbox.

**Data folder:** Exports and generated files (CSVs, checkpoints) are stored under **`data/`**, not in the repo root. The folder is created automatically. `data/` is in `.gitignore` so generated data is not committed.

**Export defaults:** New exports use `in:inbox` (so Sent/Drafts are not included) and skip messages where the sender is your own Gmail address. To include sent mail or all mail, call `export_emails_paginated(..., inbox_only=False, exclude_self=False)`.

**Decision column:** The pipeline fills `suggested_decision` (DELETE / KEEP / REVIEW) from the classifier. Fill the `decision` column yourself (e.g. set to `DELETE` for emails to trash), then run `delete_emails_from_csv()`. You can copy `suggested_decision` into `decision` and then edit.

**Existing CSV with self-emails:** To drop rows you sent from an already-exported CSV:  
`df = df[~df["sender"].str.contains("your.email@gmail.com", case=False, na=False)]` then save.
