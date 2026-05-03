"""
ReAct Agent Demo -- Built from Scratch with LangChain
=====================================================
This script builds a complete ReAct agent manually, showing every step
of the Reason-Act-Observe loop in beautiful terminal output using rich.

No AgentExecutor. No LangGraph. Just the raw loop, a prompt string,
regex parsing, and a dictionary of tools.

Run:
    python agent_react_demo.py
    python agent_react_demo.py --verbose   # shows the FULL prompt sent to the LLM

Requirements:
    uv pip install langchain langchain-openai langchain-core python-dotenv rich requests
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
import sys
import time
import argparse
import requests

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.rule import Rule

# =============================================================================
# SETUP
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Create a single Rich console used throughout
console = Console()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="ReAct Agent Demo")
parser.add_argument(
    "--verbose", "-v",
    action="store_true",
    help="Show the FULL prompt sent to the LLM (not just the scratchpad)",
)
args = parser.parse_args()
VERBOSE = args.verbose

# Check for required API key
if not os.getenv("OPENAI_API_KEY"):
    console.print(
        Panel(
            "[bold red]ERROR:[/bold red] OPENAI_API_KEY not found in environment.\n\n"
            "Create a [cyan].env[/cyan] file with:\n"
            "  OPENAI_API_KEY=sk-...\n\n"
            "Or export it in your shell.",
            title="Missing API Key",
            border_style="red",
        )
    )
    sys.exit(1)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================
# Each tool is a plain Python function decorated with @tool from langchain_core.
# The docstring becomes the tool description the LLM sees in the prompt.
# If a real API key is available the tool calls the real API;
# otherwise it returns fake but plausible results.
# =============================================================================

@tool
def web_search(query: str) -> str:
    """Search the internet for real-time, factual information. ALWAYS use this tool when the question involves facts you are not 100% certain about, such as: population numbers, current events, rankings, statistics, records, or any data that changes over time. Do NOT rely on your training data for factual claims -- search first. Input is a plain search query string."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        # No Tavily key -- return plausible fake results so the demo still works
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
    # Security: only allow digits, operators, parentheses, whitespace, and decimal points
    sanitized = expression.strip()
    if not re.match(r'^[\d\s\+\-\*\/\.\(\)\%\,]+$', sanitized):
        return f"Error: expression contains disallowed characters. Only numbers and +-*/()%. are permitted."
    try:
        # Replace commas that might be thousand-separators
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
        # No weather key -- return simulated weather
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
        # Use the Wikipedia REST API for a summary
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(topic)}"
        r = requests.get(url, timeout=10, headers={"User-Agent": "ReActAgentDemo/1.0"})
        if r.status_code == 404:
            return f"No Wikipedia article found for '{topic}'."
        r.raise_for_status()
        data = r.json()
        extract = data.get("extract", "No summary available.")
        # Truncate to keep context window manageable
        if len(extract) > 500:
            extract = extract[:500] + "..."
        return f"Wikipedia -- {data.get('title', topic)}: {extract}"
    except Exception as e:
        return f"[WIKIPEDIA ERROR] {e}. Returning simulated result: {topic} is a notable subject covered in many encyclopedic sources."


# =============================================================================
# TOOL REGISTRY
# =============================================================================
# We store all tools in a list and build the text blocks the prompt needs.
# The LLM only sees the plain-text descriptions -- no JSON schema, no bind_tools().
# =============================================================================

tools = [web_search, calculator, get_weather, wikipedia_lookup]
tool_registry = {t.name: t for t in tools}

# Build the two strings the prompt template needs
tools_text = "\n".join(f"{t.name}: {t.description}" for t in tools)
tool_names = ", ".join(t.name for t in tools)


# =============================================================================
# REACT PROMPT TEMPLATE
# =============================================================================
# This is the Harrison Chase ReAct prompt (hwchase17/react) written out as a
# plain string. Four placeholders: {tools}, {tool_names}, {input}, {scratchpad}.
#
# The format tells the model:
#   1. Write Thought -> Action -> Action Input (we parse these)
#   2. We paste Observation (the tool result)
#   3. Repeat until the model writes "Final Answer"
# =============================================================================

REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

IMPORTANT RULES:
- ALWAYS use tools to gather information. NEVER answer from memory alone.
- If the question involves ANY factual data (populations, dates, statistics, rankings), use web_search FIRST.
- If the question involves weather or temperature, use get_weather.
- If the question involves math, use calculator. NEVER compute in your head.
- Use multiple tools if needed. Each tool call gives you one piece of the puzzle.
- Only write "Final Answer" when you have ALL the information from tools.

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


# =============================================================================
# OUTPUT PARSER
# =============================================================================
# We parse the LLM's text response using regex. Two possible outcomes:
#   - The model wrote "Final Answer: ..." -> we extract the answer and stop.
#   - The model wrote "Action: ..." + "Action Input: ..." -> we run the tool.
# =============================================================================

def parse_react_output(text: str) -> dict:
    """
    Parse the LLM's ReAct-formatted output.

    Returns:
        {"final_answer": "..."} if the model is done, OR
        {"action": "tool_name", "action_input": "the input"} if it wants a tool.

    Raises:
        ValueError if the text cannot be parsed into either format.
    """
    # Check for Final Answer first
    final_match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    if final_match:
        return {"final_answer": final_match.group(1).strip()}

    # Otherwise look for Action + Action Input
    action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", text)
    input_match = re.search(r"Action Input:\s*(.*)", text, re.DOTALL)
    if action_match and input_match:
        return {
            "action": action_match.group(1).strip(),
            "action_input": input_match.group(1).strip(),
        }

    raise ValueError(f"Could not parse LLM output into Action or Final Answer:\n{text}")


# =============================================================================
# LLM SETUP
# =============================================================================
# We use ChatOpenAI with stop=["\nObservation:", "Observation:"] so the model
# halts as soon as it tries to write the Observation line. The observation
# must come from the tool, not the model -- otherwise the model hallucinates.
# =============================================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
).bind(stop=["\nObservation:", "Observation:"])


# =============================================================================
# RICH DISPLAY HELPERS
# =============================================================================
# These functions build the pretty terminal output. Each one is self-contained
# and takes only the data it needs.
# =============================================================================

def show_banner():
    """Display the startup banner and tool table."""
    banner_text = Text()
    banner_text.append("\n  ReAct Agent Demo\n", style="bold magenta")
    banner_text.append("  Reason + Act = Agent\n", style="dim magenta")
    banner_text.append("  Built from scratch with LangChain\n", style="dim")

    console.print(Panel(banner_text, border_style="bright_magenta", padding=(1, 2)))

    # Tool table
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
        console.print("[dim italic]  --verbose mode: full prompts will be shown[/dim italic]\n")


def show_question(question: str):
    """Display the user's question in a styled panel."""
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
    """Display the iteration number as a rule line."""
    console.print(Rule(f"[bold bright_yellow] Iteration {n} [/bold bright_yellow]", style="yellow"))
    console.print()


def show_scratchpad(scratchpad: str):
    """Display the current scratchpad contents (the growing memory of the agent)."""
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
    """Display the full prompt sent to the LLM (only in verbose mode)."""
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
    """Display the raw LLM output, syntax-highlighted."""
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
    """Display what was extracted from the LLM output."""
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
    """Display the tool being called and its result."""
    # Map tool names to emoji-like icons
    icons = {
        "web_search": "[bold blue]SEARCH[/bold blue]",
        "calculator": "[bold green]CALC[/bold green]",
        "get_weather": "[bold cyan]WEATHER[/bold cyan]",
        "wikipedia_lookup": "[bold magenta]WIKI[/bold magenta]",
    }
    icon = icons.get(tool_name, "[bold white]TOOL[/bold white]")

    console.print(
        Panel(
            f"{icon} [bold]{tool_name}[/bold]([cyan]{tool_input}[/cyan])\n\n"
            f"[green]Result:[/green] {result}",
            title="[bold green]Tool Execution[/bold green]",
            border_style="green",
            padding=(0, 1),
        )
    )
    console.print()


def show_decision(can_answer: bool, reason: str, next_iter: int = None):
    """Display the agent's decision: can it answer yet?"""
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
    """Display the final answer in a big green panel."""
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
    """
    Display a recap table of all iterations.

    Args:
        steps: list of dicts with keys 'iteration', 'action', 'input', 'result_preview', 'time'
        total_time: total elapsed time in seconds
    """
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
            s.get("input", "--"),
            result_preview,
            f"{s['time']:.1f}s",
        )

    console.print(table)
    console.print(f"\n  [bold]Total time:[/bold] [cyan]{total_time:.2f}s[/cyan]")
    console.print(f"  [bold]Iterations:[/bold] [cyan]{len(steps)}[/cyan]")
    console.print()


def show_parse_error(text: str, error: str):
    """Display a parse error with the offending LLM output."""
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


# =============================================================================
# THE AGENT LOOP
# =============================================================================
# This is the core of the ReAct agent. It:
#   1. Builds the prompt from the template + scratchpad
#   2. Sends it to the LLM (which stops at "Observation:")
#   3. Parses the response (Action/Input or Final Answer)
#   4. If Action: runs the tool, appends result to scratchpad, loops back
#   5. If Final Answer: returns the answer
#
# The rich display functions are called at each step so you can SEE
# exactly what the agent is doing internally.
# =============================================================================

def run_react_agent(question: str, max_iters: int = 10) -> str | None:
    """
    Run the ReAct agent loop on a question.

    Args:
        question: The user's natural language question.
        max_iters: Maximum number of Thought/Action/Observation cycles.

    Returns:
        The final answer string, or None if the agent did not finish.
    """
    show_question(question)

    scratchpad = ""
    step_log = []          # For the recap table
    start_time = time.time()

    for step in range(1, max_iters + 1):
        iter_start = time.time()
        show_iteration_header(step)

        # --- 1. Show the scratchpad (or full prompt in verbose mode) ---
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

        # --- 2. Call the LLM ---
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

        # --- 3. Parse the output ---
        try:
            parsed = parse_react_output(text)
        except ValueError as e:
            show_parse_error(text, str(e))
            # Try to recover: append what the model said and loop
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

        # --- 4a. Final Answer? We are done. ---
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
            return parsed["final_answer"]

        # --- 4b. Run the tool ---
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

        # --- 5. Decision: can the agent answer yet? ---
        show_decision(False, "Need more information or computation", step + 1)

        # --- 6. Append to scratchpad and loop ---
        # The scratchpad grows with each iteration. This is the agent's memory.
        scratchpad += text + f"\nObservation: {observation}\n"

    # If we get here, the agent ran out of iterations
    console.print(
        Panel(
            f"[bold red]Agent did not reach a final answer in {max_iters} iterations.[/bold red]\n"
            "This is a common failure mode. In production, you set a budget\n"
            "(max iterations, token cost, wall time) and bail out gracefully.",
            title="[bold red]MAX ITERATIONS REACHED[/bold red]",
            border_style="red",
        )
    )
    show_recap(step_log, time.time() - start_time)
    return None


# =============================================================================
# INTERACTIVE CLI
# =============================================================================
# The main loop: show the banner, then keep asking for questions until
# the user types 'quit' or 'exit' (or presses Ctrl+C).
# =============================================================================

def main():
    """Interactive CLI for the ReAct Agent Demo."""
    show_banner()

    # Suggest some example questions
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
            padding=(0, 2),
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
