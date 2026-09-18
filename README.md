# RAGKit

A modular framework for building **Retrieval-Augmented Generation (RAG)**
applications on top of **local Large Language Models** (via [Ollama](https://ollama.com)),
plus a small **FastAPI web UI** (RAGKit UI) for uploading documents and chatting
with them.

---

## 1. What is this project?

RAGKit is both a learning project and a small production-quality RAG framework.
Every stage of the RAG pipeline is a replaceable interface (Dependency Inversion),
so you can swap embedders, vector stores, retrievers, rerankers, or LLM providers
without touching the rest of the code.

### Features

- **Document loading** – local files, `.txt` and `.docx` (`python-docx`)
- **Chunking** – character, sentence, token, semantic, structured-text
- **Embeddings** – Ollama (`nomic-embed-text`) or `sentence-transformers`
- **Vector store** – ChromaDB (persistent, on disk)
- **Keyword search** – BM25 (`rank-bm25`)
- **Hybrid retrieval** – vector + BM25 fused with **Reciprocal Rank Fusion (RRF)**
- **Query normalization** – optional LLM-based query rewriting
- **Reranking** – identity or LLM-based reranker
- **Prompting** – pluggable prompt builders
- **LLM** – Ollama (`qwen3:8b` by default)
- **Web UI** – FastAPI + static HTML/JS: upload `.docx`, list/delete documents,
  chat, and see the original vs. normalized query and the retrieved sources
- **Logging** – rotating file + console logs (see [Logging](#6-logging))

### Project layout

```text
RAGKit_UI/
├── ragkit/                  # framework package
│   ├── api/                 # FastAPI app + routers (chat, documents)
│   ├── services/            # RAGService, DocumentService, factories
│   ├── sources/ loaders/    # document loading
│   ├── chunkers/            # chunking strategies
│   ├── embeddings/          # embedders (Ollama, sentence-transformers)
│   ├── vectorstores/        # ChromaDB store
│   ├── keyword/             # BM25 searcher
│   ├── retrievers/          # similarity + hybrid retrievers
│   ├── ranking/             # Reciprocal Rank Fusion
│   ├── rerankers/           # reranking strategies
│   ├── prompts/ llms/       # prompt building + LLM clients
│   ├── query/               # query normalization
│   ├── config/              # dataclass configuration
│   └── logger.py            # central logger + configure_logging()
├── static/                  # web UI (index.html, app.js, style.css)
├── examples/                # numbered, ordered learning path (01 → 15)
├── documents/               # uploaded documents + Chroma DB on disk
├── tests/                   # pytest suite
└── pyproject.toml
```

---

## 2. Prerequisites (all platforms)

| Requirement | Notes |
|---|---|
| **Python 3.12+** | 3.12.11 recommended |
| **Ollama** | Runs the local embedding + chat models |
| **Ollama models** | `ollama pull nomic-embed-text` and `ollama pull qwen3:8b` |
| Git | To clone the repo |

Install and start Ollama, then pull the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

Ollama must be running (`ollama serve`, or the desktop app) whenever you index
documents or chat.

---

## 3. Installation

The steps are the same everywhere: get Python 3.12, create a virtual
environment, and `pip install -e .`. Only the shell commands differ.

### macOS

```bash
# 1. Install prerequisites
brew update
brew install python@3.12 git
brew install ollama            # or download from https://ollama.com

# 2. Clone
git clone <repo-url> RAGKit_UI
cd RAGKit_UI

# 3. Virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 4. Install RAGKit (add [dev] for tests + ruff)
pip install --upgrade pip
pip install -e ".[dev]"

# 5. Pull models (Ollama running)
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

<details>
<summary>Optional: manage Python versions with <code>pyenv</code></summary>

```bash
brew install pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
pyenv install 3.12.11
pyenv local 3.12.11
```
</details>

### Linux (Debian / Ubuntu)

```bash
# 1. Install prerequisites
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3-pip git
curl -fsSL https://ollama.com/install.sh | sh

# 2. Clone
git clone <repo-url> RAGKit_UI
cd RAGKit_UI

# 3. Virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 4. Install RAGKit
pip install --upgrade pip
pip install -e ".[dev]"

# 5. Start Ollama and pull models
ollama serve &                 # if not already running as a service
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

> On Fedora/RHEL use `sudo dnf install python3.12 git`. On Arch use `sudo pacman -S python git`.

### Windows (PowerShell)

```powershell
# 1. Install prerequisites (winget)
winget install Python.Python.3.12
winget install Git.Git
winget install Ollama.Ollama

# 2. Clone
git clone <repo-url> RAGKit_UI
cd RAGKit_UI

# 3. Virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 4. Install RAGKit
python -m pip install --upgrade pip
pip install -e ".[dev]"

# 5. Pull models (Ollama running from the Start menu / system tray)
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

> If `Activate.ps1` is blocked, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

---

## 4. Running

Always activate the virtual environment first
(`source .venv/bin/activate` / `.\.venv\Scripts\Activate.ps1`)
and make sure **Ollama is running**.

### Run the web UI (RAGKit UI)

```bash
uvicorn ragkit.api.app:app --reload --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>.

- The API app is `ragkit.api.app:app`.
- Uploaded `.docx` files are stored in `documents/` and indexed into a
  persistent Chroma DB at `documents/data/vector_db_rrf/`.
- Default models: embeddings `nomic-embed-text`, chat `qwen3:8b`
  (see `ragkit/config/server_config.py`).
- Interactive API docs: <http://127.0.0.1:8000/docs>

### Run the examples (learning path)

The `examples/` folder is numbered and meant to be run in order:

```bash
python examples/01_load_documents.py
python examples/02_chunk_documents.py
# ...
python examples/11_complete_rag.py
python examples/15_rrf_chat.py          # hybrid retrieval + RRF chat
```

### Use RAGKit as a library

```python
from ragkit import configure_logging

configure_logging(level="INFO")
# build a pipeline / RagKit facade — see examples/06 and examples/11
```

### Run the tests

```bash
pytest
```

---

## 5. Configuration

Configuration lives in `ragkit/config/` as frozen dataclasses - small,
reusable per-component configs (`ChunkerConfig`, `EmbeddingConfig`,
`LLMConfig`, `RetrievalConfig`, `RerankerConfig`, `VectorStoreConfig`), each
consumed directly by its component (e.g. `CharacterChunker(config=...)`).
The web server's `ServerConfig` (`ragkit/config/server_config.py`) composes
the same `ChunkerConfig`/`EmbeddingConfig`/`LLMConfig` instances rather than
re-declaring their fields, so there is one definition of what each setting
means - `default_server_config()` just supplies server-specific defaults and
reads environment-variable overrides:

| Setting (attribute path) | Default |
|---|---|
| `vector_db_path` | `documents/data/vector_db_rrf` |
| `collection_name` | `ragkit_rrf` |
| `embedding.model` | `nomic-embed-text` |
| `llm.model` | `qwen3:8b` |
| `llm.think` | `true` (override with `RAGKIT_LLM_THINK`) |
| `chunker.chunk_size` | `2500` (override with `RAGKIT_CHUNK_SIZE`) |
| `chunker.overlap` | `300` (override with `RAGKIT_CHUNK_OVERLAP`) |
| `retrieval_top_k` | `15` (override with `RAGKIT_TOP_K`) |

`retrieval_top_k` is the one setting that stays a plain `ServerConfig` field
rather than being folded into `RetrievalConfig`/`RerankerConfig`: despite the
shared name, it means something different (RAGService's final post-RRF
result count) from either of those, and the server doesn't use the reranker
at all - forcing it into one of those classes would imply a connection that
doesn't exist.

### Chunk size and answer length

`chunk_size`/`chunk_overlap` control how uploaded `.docx` files are split
before indexing. Smaller chunks retrieve more precisely but hand the LLM
less material per source, which shows up as short, thin, or overly hedged
answers even when a relevant document is indexed - the model can only
work with what's in the top `retrieval_top_k` retrieved chunks. Increase
`RAGKIT_CHUNK_SIZE` (and `RAGKIT_TOP_K`) if answers feel too short given
what's actually in your documents.

**These settings only affect documents indexed *after* you change them.**
Documents already indexed keep their original chunk size - delete and
re-upload them (or clear `documents/data/vector_db_rrf/`) to re-chunk with
new settings.

### Thinking mode (`RAGKIT_LLM_THINK`)

`qwen3:8b` is a reasoning model: with thinking enabled (the default) it
works through a hidden chain-of-thought before answering, which produces
more thorough, detailed answers but is slow - especially on CPU-only
machines, where it can take minutes. Query normalization always runs with
thinking disabled regardless of this setting, since it's a small task that
doesn't benefit from it.

```bash
# Trade answer quality for speed
RAGKIT_LLM_THINK=false uvicorn ragkit.api.app:app --reload
```

Logging is controlled by environment variables (see below).

---

## 6. Logging

RAGKit has a central logger (`ragkit.logger.logger`, name `"ragkit"`). As a
library it installs only a `NullHandler`; applications turn logging on by calling
`configure_logging()`.

- The **web app calls `configure_logging()` automatically** on startup
  (`ragkit/api/app.py`).
- Output goes to **the console** and to a **rotating file** at
  `logs/ragkit.log` (5 MB per file, 5 backups). The `logs/` directory is
  created automatically and is git-ignored.

Logged events include: app startup and resolved server config, each chat request
and its query normalization, retrieval counts (vector / keyword / after RRF),
LLM generation (with a `took=<seconds>` timing for every Ollama call), and
document upload / index / delete. Check `logs/ragkit.log` first when a chat
request is slow or errors out — it shows exactly how long each step took.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `RAGKIT_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `RAGKIT_LOG_FILE` | Path to the log file | `logs/ragkit.log` |

```bash
RAGKIT_LOG_LEVEL=DEBUG uvicorn ragkit.api.app:app --reload
```

Enable logging from your own scripts or examples:

```python
from ragkit import configure_logging

configure_logging(level="DEBUG")            # console + logs/ragkit.log
configure_logging(log_to_console=False)     # file only
```

---

## 7. Architecture overview

```text
Query
  └─ QueryNormalizer (optional, LLM)
      ├─ Vector retrieval (Chroma + Ollama embeddings)
      └─ BM25 keyword retrieval (rank-bm25)
          └─ Reciprocal Rank Fusion (RRF)
              └─ (optional) Reranker
                  └─ PromptBuilder
                      └─ LLM (Ollama qwen3:8b)
                          └─ RAGResponse (answer + sources + queries)
```

### Why so many interfaces?

Everything is replaceable. Swap `OllamaEmbedder` for `OpenAIEmbedder`, or
`ChromaVectorStore` for FAISS, without changing anything else — the Dependency
Inversion Principle. Rich return types (`SearchResult`, `LLMResponse`,
`RAGResponse`) let fields be added later without breaking the API.

Study order for the codebase is in [`Flow.txt`](Flow.txt).

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `Connection refused` to Ollama | Start Ollama (`ollama serve` or the app) |
| `model 'qwen3:8b' not found` | `ollama pull qwen3:8b` (and `nomic-embed-text`) |
| `Static directory does not exist` | Run uvicorn from the repo root |
| Empty answers / no sources | Upload a `.docx` in the UI first so it gets indexed |
| Slow first response | Models load into memory on first use |
| Chat takes 1–3+ minutes | `qwen3:8b` is a reasoning model that, with thinking enabled (the default), generates a long hidden "thinking" trace before its answer — expensive on CPU-only machines. Set `RAGKIT_LLM_THINK=false` to skip that trace and trade answer quality for speed; check `logs/ragkit.log` for `Ollama generate: ... took=Xs` to see where the time goes, and consider a smaller model (e.g. `qwen3:1.7b`, `llama3.2`) if you're on CPU |
| Answers are one-word / too terse | Likely running with `RAGKIT_LLM_THINK=false`. Unset it (thinking is on by default) for more thorough answers, or keep thinking off for speed — the prompt (`ragkit/prompts/default_prompt_builder.py`) requires a complete sentence rather than a bare word/number either way |
| Answer is short/thin even with sources returned | The retrieved chunks are too small/narrow — check `logs/ragkit.log` for the `Retrieval: ...` line. Increase `RAGKIT_CHUNK_SIZE`/`RAGKIT_TOP_K` and **re-upload the document** (existing chunks in the index don't change automatically) |
