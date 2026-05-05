# Researcharr — Your AI Research Assistant

Researcharr turns your Zotero library into a personal AI research assistant. Think of it like NotebookLM: you can **chat with your own documents** — ask questions, get summaries, and find things across everything you've saved in Zotero.

It works by **indexing** your documents (reading them and turning them into searchable data), then letting you ask questions in plain language. The AI finds the most relevant parts of your documents and gives you answers with citations.

---

## How it works (in plain English)

1. You add your API keys (so the app can use AI models).
2. You pick which collections in Zotero to index.
3. The app reads your documents, chops them into small pieces, and creates "embeddings" (a kind of smart fingerprint for each piece).
4. You ask questions — the app finds the most relevant pieces and uses an AI model to answer.

---

## Setup

There are two ways to run Researcharr. Pick the one that sounds easiest to you.

### Option A: Docker (Recommended — easiest)

Docker is like a "to-go box" for apps — it packages everything together so it just works, and you don't need to install anything complicated.

1. **Install Docker Desktop**
   - Go to [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
   - Download and install it (it's free)
   - Open Docker Desktop and wait for it to finish starting up

2. **Open Terminal**
   - On Mac: Press `Cmd + Space`, type "Terminal", and press Enter
   - On Windows: Press the Start button, type "Command Prompt" or "PowerShell", and open it

3. **Navigate to the Researcharr folder**
   - In Terminal, type: `cd "` then drag the Researcharr folder into the Terminal window, then type `"`
   - Press Enter
   - _(On Mac you can also type `cd ` and drag the folder, it'll fill in the path automatically)_

4. **Start the app**
   ```bash
   docker compose up --build
   ```
   - This will take a few minutes the first time (it's downloading and setting up everything)
   - You'll see a lot of text scrolling by — that's normal!

5. **Open the app**
   - Open your web browser (Chrome, Safari, Edge, or Firefox)
   - Go to: `http://127.0.0.1:5000`

6. **To stop the app**: Go back to the Terminal and press `Ctrl + C` (hold Control, press C)

> **Tip:** Docker is the simplest path because you don't need to install Python or any other tools. Everything is included in the "to-go box."

### Option B: Run with Python (if you already have Python installed)

1. Open Terminal (see steps above).
2. Navigate to the Researcharr folder (same as step 3 above).
3. Run these commands one by one (copy each, paste, press Enter):

   ```bash
   cd webapp
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cd ..
   HOST=127.0.0.1 PORT=5000 ./webapp/start.sh
   ```

4. Open your browser and go to `http://127.0.0.1:5000`

---

## First-time setup (in the app)

Once the app is running and you see it in your browser, follow these steps:

### 1. Get an API key from OpenRouter

The app needs an AI "brain" to answer your questions. OpenRouter is a service that gives you access to many AI models in one place.

**What is OpenRouter?** Think of it as a store where you buy access to AI models. You put money into your account (as little as $5), and the app spends tiny amounts per question. Most answers cost a fraction of a cent.

**How to get your API key:**

1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
2. Create an account or log in (you can sign in with Google or GitHub)
3. Click **"Create Key"**
4. Give it a name (like "Researcharr")
5. Copy the key (it looks like: `sk-or-v1-...`)

> **⚠️ Important:** Set a usage limit to avoid surprises. On OpenRouter's website, go to **Settings > Billing > Limits**. Set a monthly spending limit (e.g., $5 or $10). This way the app will stop spending once you hit that limit.

### 2. Add your API key to the app

1. In the app, click the **Settings** tab (gear icon in the sidebar).
2. Under **API Keys**, select **OpenRouter** as your provider.
3. Paste your API key into the field and click **Save**.
4. Click **"Load models"** to see available AI models.

### 3. Choose your AI models

#### Chat model (the one that answers your questions)
- Recommended: **DeepSeek V4 Flash** (or `deepseek/deepseek-v4-flash`)
- Why: It's very cheap and works great for answering research questions
- You can always change this later

#### Embeddings model (the one that reads and understands your documents)
- Recommended: **OpenAI Embeddings Small** (or `openai/text-embedding-3-small`)
- Cost: Very cheap — you pay once per document during indexing, then it's free to search

> **What are embeddings?** Imagine you have a giant pile of LEGO bricks. Embeddings are like sorting all the blue bricks together, all the red bricks together, etc. When you ask a question, the app uses embeddings to find the pieces of text that are most similar to your question. It's how the app "understands" what your documents are about.

#### Reranker (makes answers more accurate)

**What is a reranker?** After the app finds candidate pieces of text, a reranker gives each one a second, more careful look to pick the very best ones. Think of it as a first pass that grabs 20 possibly relevant passages, then a reranker carefully picks the 5 best ones.

**Options:**
- **Local (free):** Runs on your computer, no extra cost. This is the default and works well.
- **OpenRouter-hosted:** Uses a paid model through OpenRouter for even better results.

Recommended if you want to try a paid option: **Cohere Rerank V4 Pro** or **Cohere Rerank Fast** (both available through OpenRouter).

### 4. Index your documents

1. Click the **Index** tab (the 4-square grid icon in the sidebar).
2. You'll see your Zotero collections listed. Check the ones you want to index.
3. The app will **show you an estimated cost** before it starts — you can decide if you want to proceed.
4. Click **Start Indexing**. This may take a while (minutes to hours depending on how many documents you have).
5. Wait for it to finish. The app will tell you when it's done.

> **💡 Tip:** Start with just one or two collections to test things out. You can always index more later.

> **💡 Tip about costs:** The estimate is just an estimate. Indexing is a one-time cost — once your documents are indexed, asking questions is very cheap (fractions of a cent per question).

### 5. Start chatting!

1. Click the **Chat** tab (the speech bubble icon).
2. Type a question about your research — for example, "What are the main findings of these papers?"
3. The app will search your indexed documents and give you an answer with citations.
4. You can see which parts of which documents were used by clicking the citations.

---

## The final result

Once you've gone through these steps, you have your own NotebookLM-like system:

- **Ask any question** about your research
- **Get cited answers** that point back to the original sources
- **Search across everything** you've saved in Zotero
- **Choose any models** you prefer — swap the chat model, embeddings model, or reranker anytime from Settings

Everything runs locally on your computer. Your documents and API keys stay with you.

---

## Troubleshooting

**"I see a blank page when I go to `http://127.0.0.1:5000`"**
- Make sure the app is still running in Terminal
- Try refreshing the page

**"The app says it can't find my Zotero library"**
- Make sure Zotero is installed and has been opened at least once
- Researcharr reads your Zotero database directly — Zotero can be open while you use it

**"Indexing is taking forever"**
- This is normal if you have a large library. Try indexing fewer collections at a time
- Each document needs to be read, chunked, and converted into embeddings

**"API key errors"**
- Make sure you copied the full key (including `sk-or-v1-`)
- Make sure you have credits in your OpenRouter account
- Check your usage limits on OpenRouter's website

---

## Need help?

If something isn't working, open an issue at [github.com/anomalyco/researcharr/issues](https://github.com/anomalyco/researcharr/issues).
