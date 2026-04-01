"""Chitti — AI personal assistant powered by Google Gemini."""

import asyncio
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv
import os

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from chitti.client import ChittiClient, Usage


BASE_DIR = Path(__file__).parent.parent.parent  # src/chitti -> src -> project root
console = Console()


def load_config() -> dict:
    """Load and validate chitti.toml."""
    config_path = BASE_DIR / "chitti.toml"
    if not config_path.exists():
        console.print("[bold red]Error:[/] chitti.toml not found.")
        sys.exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_soul(config: dict) -> str:
    """Read the SOUL.md file."""
    soul_path = BASE_DIR / config["assistant"]["soul_file"]
    if not soul_path.exists():
        console.print(f"[bold red]Error:[/] {soul_path.name} not found.")
        sys.exit(1)
    return soul_path.read_text()


def get_api_key() -> str:
    """Get GEMINI_API_KEY from environment."""
    load_dotenv(BASE_DIR / ".env")
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        console.print("[bold red]Error:[/] GEMINI_API_KEY not set.")
        console.print("Add it to .env or export it. Get one at https://aistudio.google.com/apikey")
        sys.exit(1)
    return api_key


def print_usage(usage: Usage | None) -> None:
    """Print token usage in a compact format."""
    if not usage:
        return
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column(style="dim cyan", justify="right")
    table.add_row("in", str(usage.input_tokens))
    table.add_row("out", str(usage.output_tokens))
    if usage.thought_tokens:
        table.add_row("think", str(usage.thought_tokens))
    table.add_row("total", str(usage.total_tokens))
    console.print(table)


def print_session_usage(client: ChittiClient) -> None:
    """Print cumulative session token usage."""
    su = client.session_usage
    table = Table(title="Session Usage", box=None, padding=(0, 1))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")
    table.add_row("Interactions", str(su.interaction_count))
    table.add_row("Input tokens", str(su.total_input))
    table.add_row("Output tokens", str(su.total_output))
    if su.total_thought:
        table.add_row("Thought tokens", str(su.total_thought))
    table.add_row("Total tokens", str(su.total_tokens))
    console.print(table)


def handle_command(cmd: str, client: ChittiClient, config: dict) -> bool:
    """Handle slash commands. Returns True if handled."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()

    match command:
        case "/new":
            client.new_conversation()
            console.print("[dim]Started a new conversation.[/]")
        case "/usage":
            print_session_usage(client)
        case "/model":
            console.print(f"[dim]Model:[/] {client.model}")
        case "/help":
            help_text = Text()
            help_text.append("/new", style="bold cyan")
            help_text.append("     Start a new conversation\n")
            help_text.append("/usage", style="bold cyan")
            help_text.append("   Show session token usage\n")
            help_text.append("/model", style="bold cyan")
            help_text.append("   Show current model\n")
            help_text.append("/help", style="bold cyan")
            help_text.append("    Show this help\n")
            help_text.append("quit", style="bold cyan")
            help_text.append("     Exit Chitti")
            console.print(help_text)
        case _:
            console.print(f"[dim]Unknown command: {command}. Type /help for options.[/]")
    return True


async def repl(client: ChittiClient, config: dict) -> None:
    """Main async REPL loop."""
    session: PromptSession[str] = PromptSession(history=InMemoryHistory())
    show_usage = config["display"]["show_usage"]
    show_thinking = config["display"]["show_thinking"]

    console.print(f"[bold green]Chitti v0.1[/] — model: [cyan]{client.model}[/]")
    console.print("[dim]Type /help for commands, quit to exit.[/]\n")

    while True:
        try:
            user_input = await session.prompt_async("you> ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            console.print("[dim]Bye![/]")
            break
        if user_input.startswith("/"):
            handle_command(user_input, client, config)
            continue

        # Stream the response
        console.print()
        full_text: list[str] = []
        thought_text: list[str] = []
        last_usage: Usage | None = None

        try:
            async for chunk in client.send(user_input):
                if chunk.text:
                    full_text.append(chunk.text)
                    console.print(chunk.text, end="", highlight=False)
                if chunk.thought_summary and show_thinking:
                    thought_text.append(chunk.thought_summary)
                if chunk.is_complete:
                    last_usage = chunk.usage
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/]")
            continue
        except Exception as e:
            console.print(f"\n[bold red]Error:[/] {e}")
            continue

        # Render the full response as markdown
        response = "".join(full_text)
        if response:
            # Clear the raw streamed text and re-render as markdown
            # Only do this if the response contains markdown formatting
            if any(c in response for c in ["#", "```", "**", "- ", "1. "]):
                console.print()  # newline after stream
                console.print(Markdown(response))
            else:
                console.print()  # just a newline after stream

        if thought_text and show_thinking:
            console.print(f"[dim italic]{''.join(thought_text)}[/]")

        if show_usage and last_usage:
            print_usage(last_usage)

        console.print()


def main() -> None:
    """Entry point."""
    api_key = get_api_key()
    config = load_config()
    soul = load_soul(config)
    client = ChittiClient(api_key=api_key, config=config, soul=soul)
    try:
        asyncio.run(repl(client, config))
    except KeyboardInterrupt:
        console.print("\n[dim]Bye![/]")


if __name__ == "__main__":
    main()
