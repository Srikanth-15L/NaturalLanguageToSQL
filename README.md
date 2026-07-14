# DataInsight AI — Natural Language to SQL Agent

Ask questions in plain English, get answers from your database. Powered by a custom ReAct agent loop built with LangChain and Groq.

**Repository:** https://github.com/Srikanth-15L/NaturalLanguageToSQL.git

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/Srikanth-15L/NaturalLanguageToSQL.git
cd NaturalLanguageToSQL
```

### 2. Set up API keys

```bash
cp .env.example .env
```

Open `.env` and add your keys:

```
GROQ_API_KEY=gsk_...
```

### 3. Start the backend

```bash
uv sync
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Run the CLI agents directly

```bash
# SQL agent (natural language to SQL)
uv run python core/sql_agent.py
uv run python core/sql_agent.py --verbose

# Search agent (web search, weather, calculator)
uv run python core/search_agent.py
uv run python core/search_agent.py --verbose
```
