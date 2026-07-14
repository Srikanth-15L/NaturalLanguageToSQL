"""
ReAct Agent Demo -- Built from Scratch with LangChain
=====================================================
A message-based agent loop that shows every step of the
Thought -> Action -> Observation cycle in the terminal.

Uses Groq's native Tool-Calling capabilities.

Usage:
    python core/search_agent.py
    python core/search_agent.py --verbose   # shows the full message history each iteration

Requirements:
    uv pip install langchain langchain-groq langchain-core python-dotenv rich requests
"""

import os
import re
import sys
import time
import argparse
import requests

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule

load_dotenv()

console = Console()

parser = argparse.ArgumentParser(description="ReAct Agent Demo")
parser.add_argument(
    "--verbose", "-v",
    action="store_true",
    help="Show the full message payload sent to the LLM each iteration",
)
args = parser.parse_args()
VERBOSE = args.verbose

if not os.getenv("GROQ_API_KEY"):
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
# Tools
# -------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the internet for real-time, factual information. ALWAYS use this tool when the question involves facts you are not 100% certain about, such as: population numbers, current events, rankings, statistics, records, or any data that changes over time. Do NOT rely on your training data for factual claims -- search first. Input is a plain search query string."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            f"[SIMULATED SEARCH] Results for '{query}':\n"
            f"- Wikipedia: {query} is a widely discussed topic with many sources.\n"
            f"- News article: Recent developments related to {query} show growing interest.\n"
            f"- Research paper: A 2024 study examined {query} in depth."
        )
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": 3},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        return "\n".join(
            f"- {item['title']}: {item['content'][:250]}"
            for item in data.get("results", [])
        )
    except Exception as e:
        return f"[SEARCH ERROR] {e}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the exact result. ALWAYS use this for any calculation -- never do math in your head. Input should be a valid Python math expression like '2 + 3 * 4' or '(100 / 5) ** 2'. Only arithmetic operators are supported: + - * / ** % ()."""
    sanitized = expression.strip()
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%\,]+$', sanitized):
        return "Error: expression contains disallowed characters. Only numbers and +-*/()%. are permitted."
    try:
        sanitized = sanitized.replace(",", "")
        result = eval(sanitized, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current LIVE weather and temperature for a city. Use this whenever the question asks about weather, temperature, or climate conditions. This returns real-time data -- do NOT guess weather from memory. Input is a city name like 'London' or 'Tokyo, Japan'."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        import random
        temp = random.randint(5, 35)
        conditions = random.choice(["Sunny", "Partly Cloudy", "Overcast", "Light Rain", "Clear"])
        return (
            f"[SIMULATED WEATHER] {city}: {temp}C, {conditions}, "
            f"Humidity: {random.randint(30,90)}%, Wind: {random.randint(5,30)} km/h"
        )
    try:
        r = requests.get(
            "http://api.weatherapi.com/v1/current.json",
            params={"key": api_key, "q": city, "aqi": "no"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        cur = data["current"]
        loc = data["location"]
        return (
            f"{loc['name']}, {loc['country']}: {cur['temp_c']}C, "
            f"{cur['condition']['text']}, Humidity: {cur['humidity']}%, "
            f"Wind: {cur['wind_kph']} km/h"
        )
    except Exception as e:
        return f"[WEATHER ERROR] {e}"


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up a topic on Wikipedia and return an authoritative summary. Use this for definitions, biographies, historical facts, or explanations of concepts. Input is the topic name like 'Python programming language' or 'Albert Einstein'."""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(topic)}"
        r = requests.get(url, timeout=10, headers={"User-Agent": "ReActAgentDemo/1.0"})
        if r.status_code == 404:
            return f"No Wikipedia article found for '{topic}'."
        r.raise_for_status()
        data = r.json()
        extract = data.get("extract", "No summary available.")
        if len(extract) > 500:
            extract = extract[:500] + "..."
        return f"Wikipedia -- {data.get('title', topic)}: {extract}"
    except Exception as e:
        return f"[WIKIPEDIA ERROR] {e}. Returning simulated result: {topic} is a notable subject covered in many encyclopedic sources."


tools = [web_search, calculator, get_weather, wikipedia_lookup]
tool_registry = {t.name: t for t in tools}


# -------------------------------------------------------------------
# System Prompt & LLM Setup
# -------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's questions as best you can.
You have access to tools to gather information. Follow these rules:
- ALWAYS use tools to gather information. NEVER answer from memory alone.
- If the question involves ANY factual data (populations, dates, statistics, rankings), use web_search FIRST.
- If the question involves weather or temperature, use get_weather.
- If the question involves math, use calculator. NEVER compute in your head.
- Only output the final answer when you have ALL required information.
"""

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)


# -------------------------------------------------------------------
# Terminal display helpers (Rich)
# -------------------------------------------------------------------

def show_banner():
    banner_text = Text()
    banner_text.append("\n  ReAct Agent Demo (Tool-Calling)\n", style="bold magenta")
    banner_text.append("  Reason + Act = Agent\n", style="dim magenta")
    banner_text.append("  Built using native LLM Tool-Calling bindings\n", style="dim")

    console.print(Panel(banner_text, border_style="bright_magenta", padding=(1, 2)))

    table = Table(
        title="Available Tools",
        border_style="blue",
        title_style="bold blue",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Tool", style="green", min_width=18)
    table.add_column("Description", style="white", min_width=40)

    for t in tools:
        table.add_row(t.name, t.description.split(".")[0] + ".")

    console.print(table)
    console.print()

    if VERBOSE:
        console.print("[dim italic]  --verbose mode: full message payloads will be shown[/dim italic]\n")


def show_question(question: str):
    console.print(Panel(f"[bold white]{question}[/bold white]", title="[bold blue]Your Question[/bold blue]", border_style="blue", padding=(0, 2)))
    console.print()


def show_iteration_header(n: int):
    console.print(Rule(f"[bold bright_yellow] Iteration {n} [/bold bright_yellow]", style="yellow"))
    console.print()


def show_message_payload(messages: list):
    payload = ""
    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "")
        content = msg.content
        if isinstance(msg, AIMessage) and msg.tool_calls:
            content += f"\n[Tool Calls: {msg.tool_calls}]"
        payload += f"[bold cyan]{role}:[/bold cyan] {content}\n---\n"
    console.print(Panel(payload.strip(), title="[bold blue]Message History (Agent Context)[/bold blue]", border_style="blue", padding=(0, 1)))
    console.print()


def show_llm_response(msg: AIMessage):
    content = msg.content if msg.content else "(No text output)"
    if msg.tool_calls:
        content += f"\n\n[bold magenta]Tool Calls requested:[/bold magenta]\n"
        for call in msg.tool_calls:
            content += f" - {call['name']}({call['args']})\n"
    console.print(Panel(content.strip(), title="[bold magenta]LLM Output[/bold magenta]", border_style="bright_magenta", padding=(0, 1)))
    console.print()


def show_tool_execution(tool_name: str, tool_input: dict, result: str):
    icons = {
        "web_search":       "[bold blue]SEARCH[/bold blue]",
        "calculator":       "[bold green]CALC[/bold green]",
        "get_weather":      "[bold cyan]WEATHER[/bold cyan]",
        "wikipedia_lookup": "[bold magenta]WIKI[/bold magenta]",
    }
    icon = icons.get(tool_name, "[bold white]TOOL[/bold white]")

    console.print(
        Panel(
            f"{icon} [bold]{tool_name}[/bold]({tool_input})\n\n[green]Result:[/green] {result}",
            title="[bold green]Tool Execution[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )
    console.print()


def show_final_answer(answer: str):
    console.print(Panel(f"[bold white]{answer}[/bold white]", title="[bold green]FINAL ANSWER[/bold green]", border_style="bright_green", padding=(1, 2)))
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
    table.add_column("Action", style="green", min_width=18)
    table.add_column("Input", style="white", min_width=25, max_width=40)
    table.add_column("Result (preview)", style="dim white", min_width=25, max_width=45)
    table.add_column("Time", style="cyan", justify="right", width=8)

    for s in steps:
        result_preview = s.get("result_preview", "")
        if len(result_preview) > 60:
            result_preview = result_preview[:57] + "..."
        table.add_row(
            str(s["iteration"]),
            s.get("action", "Final Answer"),
            str(s.get("input", "--")),
            result_preview,
            f"{s['time']:.1f}s",
        )

    console.print(table)
    console.print(f"\n  [bold]Total time:[/bold] [cyan]{total_time:.2f}s[/cyan]")
    console.print(f"  [bold]Iterations:[/bold] [cyan]{len(steps)}[/cyan]")
    console.print()


# -------------------------------------------------------------------
# The agent loop (Tool-Calling Implementation)
# -------------------------------------------------------------------

def run_react_agent(question: str, max_iters: int = 10) -> str | None:
    """
    Run the ReAct agent on a question using native tool-calling.

    Args:
        question: The user's natural language question.
        max_iters: Maximum number of iterations before giving up.
    """
    show_question(question)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]
    step_log = []
    start_time = time.time()

    for step in range(1, max_iters + 1):
        iter_start = time.time()
        show_iteration_header(step)

        if VERBOSE:
            show_message_payload(messages)

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            console.print(Panel(f"[bold red]LLM call failed:[/bold red] {e}", title="[bold red]ERROR[/bold red]", border_style="red"))
            return None

        show_llm_response(response)

        if response.tool_calls:
            messages.append(response)

            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                call_id = tool_call["id"]

                if name not in tool_registry:
                    result = f"Error: unknown tool '{name}'."
                else:
                    try:
                        if len(args) == 1:
                            val = list(args.values())[0]
                            result = str(tool_registry[name].invoke(val))
                        else:
                            result = str(tool_registry[name].invoke(args))
                    except Exception as e:
                        result = f"Error running {name}: {e}"

                show_tool_execution(name, args, result)

                step_log.append({
                    "iteration": step,
                    "action": name,
                    "input": str(args),
                    "result_preview": result,
                    "time": time.time() - iter_start
                })

                messages.append(ToolMessage(content=result, tool_call_id=call_id))
        else:
            final_answer = response.content
            iter_time = time.time() - iter_start
            step_log.append({
                "iteration": step,
                "action": "Final Answer",
                "input": "--",
                "result_preview": final_answer,
                "time": iter_time
            })

            show_final_answer(final_answer)
            show_recap(step_log, time.time() - start_time)
            return final_answer

    console.print(
        Panel(
            f"[bold red]Agent did not reach a final answer in {max_iters} iterations.[/bold red]",
            title="[bold red]MAX ITERATIONS REACHED[/bold red]",
            border_style="red"
        )
    )
    show_recap(step_log, time.time() - start_time)
    return None


def main():
    show_banner()

    console.print(
        Panel(
            "[bold]Try these example questions:[/bold]\n\n"
            "  1. What is the population of Tokyo multiplied by 2?\n"
            "  2. What is the weather in London and Paris right now?\n"
            "  3. Tell me about the Python programming language and compute 2**10.\n"
            "  4. Who invented the telephone? Search the web and Wikipedia.\n"
            "  5. What is (365 * 24 * 60) and what does that number represent?\n",
            title="[bold cyan]Example Questions[/bold cyan]",
            border_style="cyan",
            padding=(0, 2)
        )
    )

    while True:
        try:
            console.print()
            question = console.input("[bold bright_magenta]Enter your question[/bold bright_magenta] (or 'quit'): ")
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
