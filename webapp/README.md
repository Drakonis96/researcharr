# Researcharr WebApp

AI-powered research assistant that connects directly to your local Zotero SQLite database.

## Features

- **Direct database connection**: Reads your Zotero library without needing a plugin
- **Semantic search**: Index your library with embeddings for intelligent retrieval
- **RAG chat**: Ask questions about your research with cited answers
- **Web interface**: Clean, modern UI accessible from your browser

## Setup

1. **Install dependencies** (already done):
```bash
cd webapp
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`./start.sh` will use the repo root `.venv` when present and fall back to `webapp/venv`.

2. **Start the server**:
```bash
cd /path/to/researcharr
HOST=127.0.0.1 PORT=5000 ./webapp/start.sh
```

Or manually:
```bash
cd webapp
source venv/bin/activate
HOST=127.0.0.1 PORT=5000 python app.py
```

3. **Start with Docker Compose**:
```bash
cd /path/to/researcharr
docker compose up --build
```

The compose setup exposes `http://127.0.0.1:5000`, mounts your local `${HOME}/Zotero` library read-only at `/zotero`, and persists app state plus vector indexes under `webapp/instance/`.

4. **Open in browser**:
Navigate to `http://127.0.0.1:5000`

## Usage

1. **Settings**: Save one or more provider API keys, activate the one you want to use, and then load chat/embeddings models
2. **Index**: Click "Start Indexing" to vectorize your library (this may take a while for large libraries)
3. **Chat**: Ask questions about your research and get cited answers with source fragments
4. **Library**: Browse your Zotero items and collections

## Architecture

- `app.py` - Flask backend with REST API
- `zotero_reader.py` - SQLite database reader for Zotero
- `openrouter_client.py` - OpenRouter API client
- `settings_store.py` - Persistent encrypted config and API key storage
- `vector_store.py` - Local vector storage with numpy
- `templates/index.html` - Web frontend

## Data Access

The app reads from `~/Zotero/zotero.sqlite` (Zotero's local database). It creates a temporary copy to avoid lock conflicts while Zotero is running.

In containers, the default compose file maps `${HOME}/Zotero` to `/zotero` and configures `ZOTERO_DB_PATH=/zotero/zotero.sqlite` plus `ZOTERO_STORAGE_DIR=/zotero/storage`.

## Notes

- Your Zotero library has **22,505 items** and **133 collections**
- Indexing all items may take time and consume API credits
- Consider indexing specific collections rather than the entire library
- API keys are stored only on the server side, encrypted at rest under `webapp/instance/`, and are never returned to the browser after they are saved
- If you expose the app beyond localhost, put it behind HTTPS and authentication; the app now defaults to `127.0.0.1` and no longer enables Flask debug mode by default
