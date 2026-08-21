Here is a complete, production-ready `README.md` for your GitHub repository.

```markdown
# 💬 Multilingual WhatsApp & Multi-Source RAG Support System

An intelligent, production-ready customer support system built with **FastAPI**, **LangChain**, **Pinecone**, and **OpenAI**. The system provides automated customer support over **WhatsApp Cloud API** and includes a **ChatGPT-style Web Playground** capable of indexing and chatting with PDFs, Excel sheets, CSVs, plain text, and live web URLs with native **English & Hinglish** (Hindi in Roman script) support.

---

## 🚀 Features

- **WhatsApp Cloud API Integration:** Webhook-based, real-time message handling with auto-reply within the 24-hour service window.
- **Multilingual & Hinglish Support:** Uses `text-embedding-3-large` and `gpt-4o` with prompt engineering to understand and mirror Hinglish queries naturally against English documentation.
- **Universal Multi-Source Ingestion:** Ingests and parses multiple document formats:
  - 📄 **PDFs** (`.pdf` up to 8 MB)
  - 📊 **Spreadsheets & Data** (`.xlsx`, `.xls`, `.csv`)
  - 📝 **Plain Text / Docs** (`.txt`, `.md`)
  - 🌐 **Live Web URLs** (automated web scraping and cleaning)
- **ChatGPT-Style Web Playground:**
  - Dark-mode chat UI with markdown rendering.
  - Interactive file upload & web URL scraper.
  - **Chunk Inspector Drawer:** Inspect token splits, character counts, and source page/row metadata.
  - Session-isolated vector namespaces in Pinecone.
- **Resilient Session Management:** Built-in Redis support for conversation history with a graceful in-memory dictionary fallback.
- **Human Escalation Ready:** Intent-based routing to seamlessly hand off conversations to live support platforms (e.g., Chatwoot, Zendesk).

---

## 🛠️ Tech Stack

- **Backend:** Python 3.12, FastAPI, Uvicorn
- **Orchestration & RAG:** LangChain, LangChain-OpenAI, LangChain-Pinecone
- **Vector Database:** Pinecone (Serverless)
- **Models:**
  - Embedding: `text-embedding-3-large` (3072 dimensions)
  - LLM: `gpt-4o`
- **Parsers:** PyPDF, Pandas, OpenPyXL, BeautifulSoup4
- **State/Cache:** Redis (with local fallback)
- **Frontend:** Tailwind CSS, Marked.js (HTML5/Vanilla JS)

---

## 📂 Project Structure

```text
whatsapp-rag-bot/
├── static/
│   ├── index.html            # ChatGPT-style web UI for default support bot
│   └── playground.html       # Multi-source document tester & chunk inspector
├── docs/                     # Default knowledge base PDFs
├── ingest.py                 # Offline batch document ingestion script
├── rag_engine.py             # Core RAG logic, multilingual prompts & parsers
├── main.py                   # FastAPI application & WhatsApp webhook endpoints
├── test_search.py            # Local vector search verification script
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions
├── .python-version           # Specifies Python 3.12 for cloud platforms
└── requirements.txt          # Python dependencies

```

---

## ⚙️ Prerequisites & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/whatsapp-rag-support.git](https://github.com/YOUR_USERNAME/whatsapp-rag-support.git)
cd whatsapp-rag-support

```

### 2. Create Virtual Environment

```bash
python -m venv venv

# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# OpenAI API
OPENAI_API_KEY=sk-proj-yourOpenAIKeyHere

# Pinecone
PINECONE_API_KEY=pcsk_yourPineconeKeyHere
PINECONE_INDEX_NAME=whatsapp-rag-index

# Meta WhatsApp Cloud API
WHATSAPP_TOKEN=EAAG...
PHONE_NUMBER_ID=109876543210000
WHATSAPP_BUSINESS_ACCOUNT_ID=105678901234000
WEBHOOK_VERIFY_TOKEN=my_secret_token_123

# Redis (Optional: defaults to local in-memory store if not available)
REDIS_HOST=localhost
REDIS_PORT=6379

```

---

## 🏃 Running the Application

### 1. Ingest Default Knowledge Base Documents

Place your support PDFs into the `docs/` folder, then run:

```bash
python ingest.py

```

### 2. Start the FastAPI Server

```bash
uvicorn main:app --reload --port 8000

```

* **Main Support Bot UI:** [http://localhost:8000](http://localhost:8000)
* **Universal Multi-Source Playground:** [http://localhost:8000/playground](http://localhost:8000/playground)

---

## 📱 WhatsApp Integration Setup

1. **Expose Local Server (for testing):**
```bash
# Using Cloudflare Tunnel
cloudflared tunnel --url http://localhost:8000

# Or using LocalTunnel
npx localtunnel --port 8000

```


2. **Meta Developer Portal:**
* Go to **WhatsApp** $\rightarrow$ **Configuration** $\rightarrow$ **Webhook**.
* **Callback URL:** `https://<YOUR_TUNNEL_URL>/webhook`
* **Verify Token:** The value set in `WEBHOOK_VERIFY_TOKEN` in your `.env`.
* Subscribe to the **`messages`** webhook field.


3. Add your phone number under **API Setup** and send a test message.

---

## ☁️ Deployment (Render / Cloud)

1. Push this repository to a **Private** GitHub repository.
2. In **[Render](https://www.google.com/search?q=https://render.com/)**, create a new **Web Service** connected to your repository.
3. Set the following build configuration:
* **Runtime:** `Python 3`
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`


4. Add your secrets under **Environment Variables** (`OPENAI_API_KEY`, `PINECONE_API_KEY`, etc.).
5. Update your Meta Webhook Callback URL to point to your live Render URL:
`https://<your-app-name>.onrender.com/webhook`

---

## 🔒 Security & Privacy Notes

* **Never commit your `.env` file.** Keep credentials secure in cloud environment secrets.
* Uploaded files in the playground are processed in session-isolated Pinecone namespaces to avoid cross-contamination of document context.

---

## 📄 License

MIT License. Feel free to use and modify for personal and commercial projects.

```

```
