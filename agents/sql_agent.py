"""
Natural Language to SQL Agent
==============================
Converts plain English questions into SQL queries, runs them against
a local SQLite database, and returns a readable answer.

Built on top of the same ReAct (Reason + Act) loop as search_agent.py,
but all the tools here are SQL-specific.

Usage:
    python sql_agent.py
    python sql_agent.py --verbose   # prints the full prompt each iteration

Requirements:
    uv pip install langchain langchain-groq langchain-core python-dotenv rich
"""

import os
import re
import sys
import time
import sqlite3
import argparse

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule
from rich import print as rprint

load_dotenv()

console = Console()

# Only parse CLI args when the script is run directly.
# When api.py imports this module, we skip argparse entirely.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Natural Language to SQL Agent")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the full prompt sent to the LLM on every iteration",
    )
    args = parser.parse_args()
    VERBOSE = args.verbose
else:
    VERBOSE = False

if not os.getenv("GROQ_API_KEY"):
    if __name__ == "__main__":
        console.print(
            Panel(
                "[bold red]ERROR:[/bold red] GROQ_API_KEY not found in environment.\n\n"
                "Create a [cyan].env[/cyan] file with:\n"
                "  GROQ_API_KEY=gsk_...\n\n"
                "Or export it in your shell.",
                title="Missing API Key",
                border_style="red",
            )
        )
        sys.exit(1)


# -------------------------------------------------------------------
# Database setup
# -------------------------------------------------------------------
# We use a file-based SQLite DB (demo_company.db) with three tables:
#   departments, employees, projects
#
# The tables are always dropped and recreated on startup so the demo
# starts from a clean, predictable state.
# -------------------------------------------------------------------

def create_database() -> sqlite3.Connection:
    """Create and seed the demo SQLite database, return the connection."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "demo_company.db")
    print(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    # Drop and recreate so we always start fresh
    cursor.execute("DROP TABLE IF EXISTS departments")
    cursor.execute("DROP TABLE IF EXISTS employees")
    cursor.execute("DROP TABLE IF EXISTS projects")

    cursor.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            budget REAL NOT NULL,
            manager TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            salary REAL NOT NULL,
            hire_date TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            deadline TEXT NOT NULL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    """)

    departments = [
        (1, "Engineering",  1500000, "Alice Chen"),
        (2, "Marketing",     800000, "Bob Martinez"),
        (3, "Sales",        1200000, "Carol Williams"),
        (4, "HR",            500000, "David Kim"),
        (5, "Finance",       700000, "Eve Johnson"),
    ]
    cursor.executemany("INSERT INTO departments VALUES (?, ?, ?, ?)", departments)

    employees = [
        (1,  "Alice Chen",       "Engineering",  145000, "2019-03-15"),
        (2,  "Frank Lopez",      "Engineering",  125000, "2020-07-01"),
        (3,  "Grace Park",       "Engineering",  132000, "2021-01-20"),
        (4,  "Henry Nguyen",     "Engineering",  118000, "2022-06-10"),
        (5,  "Ivy Sharma",       "Engineering",  140000, "2023-02-28"),
        (6,  "Bob Martinez",     "Marketing",     98000, "2018-11-05"),
        (7,  "Jack Thompson",    "Marketing",     85000, "2022-04-15"),
        (8,  "Karen Lee",        "Marketing",     92000, "2023-08-01"),
        (9,  "Carol Williams",   "Sales",        110000, "2019-05-20"),
        (10, "Leo Brown",        "Sales",         95000, "2021-09-12"),
        (11, "Maria Garcia",     "Sales",         88000, "2024-01-15"),
        (12, "David Kim",        "HR",           105000, "2020-02-14"),
        (13, "Nina Patel",       "HR",            78000, "2023-06-01"),
        (14, "Eve Johnson",      "Finance",      115000, "2019-08-22"),
        (15, "Oscar Davis",      "Finance",       92000, "2022-11-30"),
    ]
    cursor.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", employees)

    projects = [
        (1,  "Cloud Migration",       1, "In Progress",    "2025-06-30"),
        (2,  "Mobile App v2",          1, "In Progress",    "2025-04-15"),
        (3,  "API Redesign",           1, "Completed",      "2024-12-01"),
        (4,  "Brand Refresh",          2, "In Progress",    "2025-03-31"),
        (5,  "Social Media Campaign",  2, "Completed",      "2024-11-15"),
        (6,  "Q1 Sales Push",          3, "Completed",      "2025-03-31"),
        (7,  "Enterprise Accounts",    3, "In Progress",    "2025-08-30"),
        (8,  "Partner Program",        3, "Behind Schedule","2025-02-28"),
        (9,  "Benefits Overhaul",      4, "In Progress",    "2025-05-15"),
        (10, "Hiring Pipeline",        4, "Behind Schedule","2025-01-31"),
        (11, "Budget Forecast 2026",   5, "In Progress",    "2025-09-30"),
        (12, "Audit Preparation",      5, "Behind Schedule","2025-03-15"),
    ]
    cursor.executemany("INSERT INTO projects VALUES (?, ?, ?, ?, ?)", projects)

    conn.commit()
    return conn


# Global DB connection shared by all tools and the FastAPI routes
DB_CONN = create_database()


# -------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------

@tool
def list_tables() -> str:
    """List all tables in the database. ALWAYS call this first before writing any SQL. No input needed -- just pass an empty string or 'all'. This tells you what data is available."""
    cursor = DB_CONN.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    return f"Tables in the database: {', '.join(tables)}"


@tool
def get_schema(table_name: str) -> str:
    """Get the schema (column names and types) of a specific table. ALWAYS call this before writing SQL so you know the exact column names. Input is the table name like 'employees'. Do NOT guess column names from memory."""
    table_name = table_name.strip().strip("'\"")
    cursor = DB_CONN.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        if not columns:
            return f"Error: table '{table_name}' not found. Use list_tables to see available tables."

        schema_lines = []
        for col in columns:
            cid, name, dtype, notnull, default, pk = col
            parts = [f"  {name} ({dtype})"]
            if pk:
                parts.append("PRIMARY KEY")
            if notnull:
                parts.append("NOT NULL")
            schema_lines.append(" ".join(parts))

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        return f"Table: {table_name} ({count} rows)\nColumns:\n" + "\n".join(schema_lines)
    except Exception as e:
        return f"Error getting schema: {e}"


@tool
def run_sql(query: str) -> str:
    """Execute a SQL query against the database and return the results. Input is a valid SQL SELECT query. Only SELECT queries are allowed for safety. Always get the schema first so you use correct column names. Do NOT wrap the query in quotes."""
    query = query.strip().strip("'\"").rstrip(";").strip()

    # Only allow read-only queries
    if not query.upper().startswith(("SELECT", "PRAGMA", "WITH")):
        return "Error: only SELECT queries are allowed. Do not use INSERT, UPDATE, DELETE, DROP, etc."

    cursor = DB_CONN.cursor()
    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        if not rows:
            return f"Query returned 0 rows.\nColumns: {', '.join(columns)}"

        # Simple pipe-delimited table, capped at 50 rows
        result_lines = [" | ".join(str(c) for c in columns)]
        result_lines.append("-" * len(result_lines[0]))
        for row in rows[:50]:
            result_lines.append(" | ".join(str(v) for v in row))

        if len(rows) > 50:
            result_lines.append(f"... and {len(rows) - 50} more rows")

        return f"Query returned {len(rows)} row(s):\n" + "\n".join(result_lines)
    except Exception as e:
        return f"SQL Error: {e}"


@tool
def validate_sql(query: str) -> str:
    """Check if a SQL query has valid syntax without executing it. Input is the SQL query to validate. Do NOT wrap in quotes."""
    query = query.strip().strip("'\"").rstrip(";").strip()

    if not query.upper().startswith(("SELECT", "PRAGMA", "WITH")):
        return "Invalid: only SELECT queries are allowed."

    cursor = DB_CONN.cursor()
    try:
        # EXPLAIN parses the query without actually running it
        cursor.execute(f"EXPLAIN {query}")
        return "Valid: the SQL syntax is correct and all referenced tables/columns exist."
    except Exception as e:
        return f"Invalid: {e}"


tools = [list_tables, get_schema, run_sql, validate_sql]
tool_registry = {t.name: t for t in tools}

tools_text = "\n".join(f"{t.name}: {t.description}" for t in tools)
tool_names = ", ".join(t.name for t in tools)


# -------------------------------------------------------------------
# ReAct prompt (tuned for SQL workflows)
# -------------------------------------------------------------------
# The LLM is instructed to always follow the same order:
#   list_tables -> get_schema -> validate_sql -> run_sql -> Final Answer
# This prevents it from guessing column names or skipping validation.
# -------------------------------------------------------------------

REACT_PROMPT = """You are a helpful SQL assistant. You convert natural language questions into SQL queries, execute them, and return clear answers.

You have access to the following tools:

{tools}

IMPORTANT WORKFLOW -- follow these steps IN ORDER, do NOT skip any:
1. ALWAYS start with list_tables to see what tables are available.
2. ALWAYS use get_schema on the relevant tables to learn the exact column names. NEVER guess column names.
3. ALWAYS use validate_sql to check your SQL query BEFORE running it.
4. Use run_sql to execute the validated query.
5. Interpret the results and give a clear, human-readable final answer.

RULES:
- You MUST call validate_sql before run_sql. No exceptions.
- You MUST call get_schema before writing any SQL. Do not guess column names.
- If validate_sql returns an error, fix the query and validate again before running.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{scratchpad}"""


def parse_react_output(text: str) -> dict:
    """
    Parse the LLM's ReAct-formatted output.

    Returns either:
      {"final_answer": "..."}  -- model is done
      {"action": "...", "action_input": "..."}  -- model wants to call a tool

    Raises ValueError if neither pattern is found.
    """
    final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    if final_match:
        return {"final_answer": final_match.group(1).strip()}

    action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", text)
    input_match = re.search(r"Action Input:\s*(.*)", text, re.DOTALL)
    if action_match and input_match:
        return {
            "action": action_match.group(1).strip(),
            "action_input": input_match.group(1).strip(),
        }

    raise ValueError(f"Could not parse LLM output:\n{text}")


# Stop sequences prevent the LLM from writing fake Observations
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
).bind(stop=["\nObservation:", "Observation:"])


# -------------------------------------------------------------------
# Terminal display helpers (Rich)
# -------------------------------------------------------------------

def show_banner():
    banner_text = Text()
    banner_text.append("\n  Natural Language to SQL Agent\n", style="bold bright_green")
    banner_text.append("  Ask questions in English, get SQL answers\n", style="dim green")
    banner_text.append("  Built from scratch with LangChain + SQLite\n", style="dim")

    console.print(Panel(banner_text, border_style="bright_green", padding=(1, 2)))

    tool_table = Table(
        title="Available Tools",
        border_style="blue",
        title_style="bold blue",
        show_header=True,
        header_style="bold cyan",
    )
    tool_table.add_column("Tool", style="green", min_width=16)
    tool_table.add_column("Description", style="white", min_width=45)

    for t in tools:
        tool_table.add_row(t.name, t.description.split(".")[0] + ".")

    console.print(tool_table)
    console.print()

    show_database_summary()

    if VERBOSE:
        console.print("[dim italic]  --verbose mode: full prompts will be shown[/dim italic]\n")


def show_database_summary():
    cursor = DB_CONN.cursor()

    db_table = Table(
        title="Database Contents",
        border_style="green",
        title_style="bold green",
        show_header=True,
        header_style="bold cyan",
    )
    db_table.add_column("Table", style="yellow", min_width=14)
    db_table.add_column("Columns", style="white", min_width=35)
    db_table.add_column("Rows", style="cyan", justify="center", width=6)

    for table_name in ["departments", "employees", "projects"]:
        cursor.execute(f"PRAGMA table_info({table_name})")
        cols = [row[1] for row in cursor.fetchall()]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        db_table.add_row(table_name, ", ".join(cols), str(count))

    console.print(db_table)
    console.print()


def show_question(question: str):
    console.print(
        Panel(
            f"[bold white]{question}[/bold white]",
            title="[bold blue]Your Question[/bold blue]",
            border_style="blue",
            padding=(0, 2),
        )
    )
    console.print()


def show_iteration_header(n: int):
    console.print(Rule(f"[bold bright_yellow] Iteration {n} [/bold bright_yellow]", style="yellow"))
    console.print()


def show_scratchpad(scratchpad: str):
    content = scratchpad.strip() if scratchpad.strip() else "(empty -- first iteration)"
    console.print(
        Panel(
            content,
            title="[bold blue]Scratchpad (agent memory)[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )
    )
    console.print()


def show_full_prompt(prompt: str):
    console.print(
        Panel(
            Syntax(prompt, "text", theme="monokai", word_wrap=True),
            title="[bold blue]FULL PROMPT SENT TO LLM[/bold blue]",
            border_style="bright_blue",
            padding=(0, 1),
        )
    )
    console.print()


def show_llm_output(text: str):
    console.print(
        Panel(
            Syntax(text, "text", theme="monokai", word_wrap=True),
            title="[bold magenta]LLM Output[/bold magenta]",
            border_style="bright_magenta",
            padding=(0, 1),
        )
    )
    console.print()


def show_parsed(parsed: dict):
    if "final_answer" in parsed:
        content = f"[bold green]Final Answer:[/bold green] {parsed['final_answer']}"
    else:
        content = (
            f"[bold cyan]Action:[/bold cyan]       {parsed['action']}\n"
            f"[bold cyan]Action Input:[/bold cyan] {parsed['action_input']}"
        )
    console.print(
        Panel(
            content,
            title="[bold yellow]Parsed[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )
    console.print()


def show_tool_execution(tool_name: str, tool_input: str, result: str):
    icons = {
        "list_tables":  "[bold yellow]TABLES[/bold yellow]",
        "get_schema":   "[bold blue]SCHEMA[/bold blue]",
        "run_sql":      "[bold green]SQL[/bold green]",
        "validate_sql": "[bold cyan]VALIDATE[/bold cyan]",
    }
    icon = icons.get(tool_name, "[bold white]TOOL[/bold white]")

    console.print(
        Panel(
            f"{icon} [bold]{tool_name}[/bold]([cyan]{tool_input}[/cyan])\n\n"
            f"[green]Result:[/green]\n{result}",
            title="[bold green]Tool Execution[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )
    console.print()


def show_decision(can_answer: bool, reason: str, next_iter: int = None):
    if can_answer:
        status = "[bold green]YES -- delivering final answer[/bold green]"
        border = "green"
    else:
        status = f"[bold red]NO[/bold red] -- {reason}"
        border = "red"

    content = f"Can I answer the question?  {status}"
    if not can_answer and next_iter:
        content += f"\n\n[bold cyan]>> LOOP BACK to Iteration {next_iter}[/bold cyan]"

    console.print(
        Panel(
            content,
            title="[bold yellow]Decision[/bold yellow]",
            border_style=border,
            padding=(0, 1),
        )
    )
    console.print()


def show_final_answer(answer: str):
    console.print(
        Panel(
            f"[bold white]{answer}[/bold white]",
            title="[bold green]FINAL ANSWER[/bold green]",
            border_style="bright_green",
            padding=(1, 2),
        )
    )
    console.print()


def show_recap(steps: list, total_time: float):
    table = Table(
        title="Recap -- Agent Execution Summary",
        border_style="magenta",
        title_style="bold magenta",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("#", style="yellow", justify="center", width=4)
    table.add_column("Action", style="green", min_width=16)
    table.add_column("Input", style="white", min_width=25, max_width=40)
    table.add_column("Result (preview)", style="dim white", min_width=25, max_width=45)
    table.add_column("Time", style="cyan", justify="right", width=8)

    for s in steps:
        result_preview = s.get("result_preview", "")
        if len(result_preview) > 60:
            result_preview = result_preview[:57] + "..."
        input_preview = s.get("input", "--")
        if len(input_preview) > 40:
            input_preview = input_preview[:37] + "..."
        table.add_row(
            str(s["iteration"]),
            s.get("action", "Final Answer"),
            input_preview,
            result_preview,
            f"{s['time']:.1f}s",
        )

    console.print(table)
    console.print(f"\n  [bold]Total time:[/bold] [cyan]{total_time:.2f}s[/cyan]")
    console.print(f"  [bold]Iterations:[/bold] [cyan]{len(steps)}[/cyan]")
    console.print()


def show_parse_error(text: str, error: str):
    console.print(
        Panel(
            f"[bold red]Parse Error:[/bold red] {error}\n\n"
            f"[dim]Raw LLM output:[/dim]\n{text}",
            title="[bold red]ERROR[/bold red]",
            border_style="red",
            padding=(0, 1),
        )
    )
    console.print()


# -------------------------------------------------------------------
# The agent loop
# -------------------------------------------------------------------
# Each iteration:
#   1. Build the prompt (template + growing scratchpad)
#   2. Call the LLM (stops before writing "Observation:")
#   3. Parse: Final Answer -> done; Action -> run tool, append, loop
#
# The typical SQL workflow looks like:
#   list_tables -> get_schema -> validate_sql -> run_sql -> Final Answer
# -------------------------------------------------------------------

def run_react_agent(question: str, max_iters: int = 12, return_steps: bool = False) -> str | dict | None:
    """
    Run the ReAct loop on a natural language database question.

    Args:
        question: Plain English question about the database.
        max_iters: Safety cap on the number of Thought/Action cycles.
        return_steps: When True, returns a dict with both the answer and
                      the step log (used by the API endpoint).

    Returns:
        The final answer string, a {"final_answer": ..., "steps": [...]} dict
        if return_steps is True, or None if the agent ran out of iterations.
    """
    show_question(question)

    scratchpad = ""
    step_log = []
    start_time = time.time()

    for step in range(1, max_iters + 1):
        iter_start = time.time()
        show_iteration_header(step)

        # Verbose mode shows the entire prompt; default shows just the scratchpad
        if VERBOSE:
            full_prompt = REACT_PROMPT.format(
                tools=tools_text,
                tool_names=tool_names,
                input=question,
                scratchpad=scratchpad,
            )
            show_full_prompt(full_prompt)
        else:
            show_scratchpad(scratchpad)

        prompt = REACT_PROMPT.format(
            tools=tools_text,
            tool_names=tool_names,
            input=question,
            scratchpad=scratchpad,
        )

        try:
            ai_response = llm.invoke(prompt)
            text = ai_response.content
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]LLM call failed:[/bold red] {e}",
                    title="[bold red]ERROR[/bold red]",
                    border_style="red",
                )
            )
            return None

        show_llm_output(text)

        try:
            parsed = parse_react_output(text)
        except ValueError as e:
            show_parse_error(text, str(e))
            scratchpad += text + "\nObservation: Your output was not in the correct format. Please use the format: Thought/Action/Action Input or Thought/Final Answer.\n"
            step_log.append({
                "iteration": step,
                "action": "PARSE ERROR",
                "input": "--",
                "result_preview": "Format error -- retrying",
                "time": time.time() - iter_start,
            })
            show_decision(False, "Output could not be parsed -- retrying", step + 1)
            continue

        show_parsed(parsed)

        if "final_answer" in parsed:
            iter_time = time.time() - iter_start
            step_log.append({
                "iteration": step,
                "action": "Final Answer",
                "input": "--",
                "result_preview": parsed["final_answer"],
                "time": iter_time,
            })
            show_decision(True, "")
            show_final_answer(parsed["final_answer"])
            show_recap(step_log, time.time() - start_time)
            if return_steps:
                return {"final_answer": parsed["final_answer"], "steps": step_log}
            return parsed["final_answer"]

        action = parsed["action"]
        action_input = parsed["action_input"]

        if action not in tool_registry:
            observation = f"Error: unknown tool '{action}'. Available tools: {tool_names}"
            show_tool_execution(action, action_input, observation)
        else:
            try:
                observation = str(tool_registry[action].invoke(action_input))
            except Exception as e:
                observation = f"Error running {action}: {e}"
            show_tool_execution(action, action_input, observation)

        iter_time = time.time() - iter_start
        step_log.append({
            "iteration": step,
            "action": action,
            "input": action_input,
            "result_preview": observation,
            "time": iter_time,
        })

        show_decision(False, "Need more information or need to run SQL", step + 1)

        # Grow the scratchpad with what the model said + the tool result
        scratchpad += text + f"\nObservation: {observation}\n"

    console.print(
        Panel(
            f"[bold red]Agent did not reach a final answer in {max_iters} iterations.[/bold red]\n"
            "The agent may be stuck in a loop. Try rephrasing your question.",
            title="[bold red]MAX ITERATIONS REACHED[/bold red]",
            border_style="red",
        )
    )
    show_recap(step_log, time.time() - start_time)
    if return_steps:
        return {"final_answer": None, "steps": step_log}
    return None


def main():
    show_banner()

    console.print(
        Panel(
            "[bold]Try these example questions:[/bold]\n\n"
            "  1. What is the average salary in each department?\n"
            "  2. Which employees were hired after 2023?\n"
            "  3. What is the total budget across all departments?\n"
            "  4. List all projects that are behind schedule.\n"
            "  5. Who is the highest paid employee and what department are they in?\n"
            "  6. How many employees does each department have?\n"
            "  7. What projects are in the Engineering department?\n"
            "  8. Show me departments where the average salary is above 100000.\n",
            title="[bold cyan]Example Questions[/bold cyan]",
            border_style="cyan",
            padding=(0, 2),
        )
    )

    while True:
        try:
            console.print()
            question = console.input("[bold bright_green]Ask a question about the database[/bold bright_green] (or 'quit'): ")
            question = question.strip()

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                console.print("\n[dim]Goodbye![/dim]\n")
                break

            console.print()
            run_react_agent(question)

        except KeyboardInterrupt:
            console.print("\n\n[dim]Interrupted. Goodbye![/dim]\n")
            break
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]\n")
            break


if __name__ == "__main__":
    main()
