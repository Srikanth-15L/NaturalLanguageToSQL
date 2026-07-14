# System Design: DataInsight AI (Natural Language to SQL)

This document describes the architecture, data flow, and components of the Natural Language to SQL interface. It uses **Mermaid.js** diagrams, which are natively rendered by GitHub and modern markdown viewers.

---

## 1. High-Level Architecture

The system consists of three main tiers:
1. **Frontend (Vite + React)**: Providing the user interface, schema browser, and reasoning visualizer.
2. **Backend (FastAPI)**: Serving API endpoints for database schema inspection and routing chat requests to the agent core.
3. **Core (ReAct Agent + SQLite)**: Running the Reason-Act-Observe loop, interacting with the LLM (Groq Llama 3.3), and executing queries on the SQLite database.

```mermaid
graph TD
    %% Define components
    subgraph Client [Client Side - React]
        UI[Chat Interface]
        SB[Schema Browser]
    end

    subgraph Server [Server Side - FastAPI]
        API[api.py Endpoint Router]
    end

    subgraph Agent [Agent Core - core/sql_agent.py]
        Loop[ReAct Agent Loop]
        Parser[Output Parser]
        DB[SQLite Database]
    end

    subgraph External [External Services]
        LLM[Groq LLM / Llama 3.3]
    end

    %% Define connections
    UI -->|POST /api/chat| API
    SB -->|GET /api/schema| API
    
    API -->|run_react_agent| Loop
    API -->|DB_CONN Schema queries| DB
    
    Loop -->|Prompt + History| LLM
    LLM -->|Formatted Thoughts/Actions| Parser
    Parser -->|Validated Actions| Loop
    
    Loop -->|run_sql / list_tables| DB
    DB -->|Query Results| Loop
```

---

## 2. ReAct Loop Request Lifecycle (Sequence Diagram)

This sequence diagram illustrates the exact sequence of actions when a user asks: *"What is the budget of the Engineering department?"*

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as React UI
    participant API as FastAPI Server
    participant Agent as SQL Agent Loop
    participant LLM as Groq LLM
    participant DB as SQLite DB

    User->>UI: Type: "Show Engineering budget"
    UI->>API: POST /api/chat { question }
    API->>Agent: run_react_agent("Show Engineering budget")
    
    Note over Agent, LLM: Iteration 1: Discover Tables
    Agent->>LLM: Prompt with query + list_tables tool description
    LLM-->>Agent: Action: list_tables
    Agent->>DB: Query tables (sqlite_master)
    DB-->>Agent: Result: ["departments", "employees", "projects"]
    
    Note over Agent, LLM: Iteration 2: Get Table Schema
    Agent->>LLM: Appended Result -> thought process
    LLM-->>Agent: Action: get_schema ("departments")
    Agent->>DB: PRAGMA table_info(departments)
    DB-->>Agent: Schema: id, name, budget, manager
    
    Note over Agent, LLM: Iteration 3: Validate SQL
    Agent->>LLM: Appended schema -> thought process
    LLM-->>Agent: Action: validate_sql ("SELECT budget FROM departments WHERE name='Engineering'")
    Agent->>DB: EXPLAIN SELECT budget FROM departments...
    DB-->>Agent: Valid: Syntax Correct
    
    Note over Agent, LLM: Iteration 4: Run SQL
    Agent->>LLM: Appended validation -> thought process
    LLM-->>Agent: Action: run_sql ("SELECT budget FROM departments WHERE name='Engineering'")
    Agent->>DB: Execute SELECT budget FROM departments...
    DB-->>Agent: Result: [1500000.0]
    
    Note over Agent, LLM: Iteration 5: Final Answer
    Agent->>LLM: Appended raw data -> thought process
    LLM-->>Agent: Final Answer: "The budget for the Engineering department is $1,500,000."
    
    Agent-->>API: Return final_answer + reasoning_steps
    API-->>UI: JSON response
    UI-->>User: Display answer bubble + collapsible reasoning steps
```

---

## 3. Database Schema Design

The SQLite database (`demo_company.db`) has three main relational tables:

```mermaid
erDiagram
    DEPARTMENTS {
        int id PK
        string name
        real budget
        string manager
    }
    EMPLOYEES {
        int id PK
        string name
        string department
        real salary
        string hire_date
    }
    PROJECTS {
        int id PK
        string name
        int department_id FK
        string status
        string deadline
    }

    DEPARTMENTS ||--o{ PROJECTS : "has"
```
