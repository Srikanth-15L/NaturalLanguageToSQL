from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any
import sqlite3

# Import our agent function and DB connection
from agents.agent_nl_sql_demo import run_react_agent, DB_CONN

app = FastAPI(title="NL to SQL Agent API")

# Enable CORS so the React frontend can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    final_answer: Optional[str]
    steps: List[dict]

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Process a natural language question and return the final answer + reasoning steps.
    """
    result = run_react_agent(req.question, return_steps=True)
    if result is None:
        return {"final_answer": "The agent was unable to process this request.", "steps": []}
    return result

@app.get("/api/schema")
def schema_endpoint():
    """
    Returns the database schema (tables and columns) for the frontend sidebar.
    """
    cursor = DB_CONN.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema = {}
    for table_name in tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        schema[table_name] = [
            {
                "name": col[1],
                "type": col[2],
                "primary_key": bool(col[5])
            }
            for col in columns
        ]
    return schema
