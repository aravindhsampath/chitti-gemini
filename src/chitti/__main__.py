"""Chitti — AI personal assistant powered by Google Gemini."""

import asyncio
import os
import shutil
import sys
import tomllib
from pathlib import Path

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from chitti.client import ChittiClient, Usage


BASE_DIR = Path(__file__).parent.parent.parent  # src/chitti -> src -> project root
console = Console()

THINKING_ABBREV = {"minimal": "min", "low": "low", "medium": "med", "high": "high"}

PT_STYLE = PTStyle.from_dict({
    "bottom-toolbar":          "bg:#1a1a2e #e0e0e0",
    "bottom-toolbar.name":     "bg:#1a1a2e bold #00d4aa",
    "bottom-toolbar.path":     "bg:#1a1a2e #888888",
    "bottom-toolbar.sep":      "bg:#1a1a2e #444444",
    "bottom-toolbar.label":    "bg:#1a1a2e #777777",
    "bottom-toolbar.value":    "bg:#1a1a2e bold #e0e0e0",
    "bottom-toolbar.cost":     "bg:#1a1a2e bold #ffd700",
    "bottom-toolbar.model":    "bg:#1a1a2e #82aaff",
    "bottom-toolbar.thinking": "bg:#1a1a2e #c792ea",
})


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


def make_status_bar(client: ChittiClient, config: dict) -> FormattedText:
    """Build the bottom toolbar formatted text."""
    su = client.session_usage
    cwd = str(BASE_DIR)
    thinking = THINKING_ABBREV.get(config["generation"]["thinking_level"], "?")
    model_short = client.model

    # Trim path if it's too long
    cols = shutil.get_terminal_size().columns
    max_path_len = max(10, cols // 4)
    if len(cwd) > max_path_len:
        cwd = "..." + cwd[-(max_path_len - 3):]

    # Build right-side segments
    right_parts: list[tuple[str, str]] = [
        ("class:bottom-toolbar.label", " IN:"),
        ("class:bottom-toolbar.value", str(su.total_input)),
        ("class:bottom-toolbar.sep",   "  "),
        ("class:bottom-toolbar.label", "OUT:"),
        ("class:bottom-toolbar.value", str(su.total_output)),
        ("class:bottom-toolbar.sep",   "  "),
        ("class:bottom-toolbar.label", "THINK:"),
        ("class:bottom-toolbar.value", str(su.total_thought)),
        ("class:bottom-toolbar.sep",   "  "),
        ("class:bottom-toolbar.cost",  f"${su.cost:.4f}"),
        ("class:bottom-toolbar.sep",   "  "),
        ("class:bottom-toolbar.label", "Mod:"),
        ("class:bottom-toolbar.model", model_short),
        ("class:bottom-toolbar.sep",   "  "),
        ("class:bottom-toolbar.label", "Th:"),
        ("class:bottom-toolbar.thinking", thinking),
        ("class:bottom-toolbar",       " "),
    ]

    # Left side
    left_parts: list[tuple[str, str]] = [
        ("class:bottom-toolbar.name", " Chitti "),
        ("class:bottom-toolbar.path", cwd),
    ]

    # Calculate spacing
    left_len = sum(len(text) for _, text in left_parts)
    right_len = sum(len(text) for _, text in right_parts)
    spacer_len = max(1, cols - left_len - right_len)

    return FormattedText(
        left_parts
        + [("class:bottom-toolbar", " " * spacer_len)]
        + right_parts
    )


def print_session_usage(client: ChittiClient) -> None:
    """Print cumulative session token usage."""
    su = client.session_usage
    console.print(f"[dim]Interactions:[/] [cyan]{su.interaction_count}[/]  "
                  f"[dim]IN:[/] [cyan]{su.total_input}[/]  "
                  f"[dim]OUT:[/] [cyan]{su.total_output}[/]  "
                  f"[dim]THINK:[/] [cyan]{su.total_thought}[/]  "
                  f"[dim]Total:[/] [cyan]{su.total_tokens}[/]  "
                  f"[dim]Cost:[/] [yellow]${su.cost:.4f}[/]")


def handle_command(cmd: str, client: ChittiClient, config: dict) -> bool:
    """Handle slash commands. Returns True if handled."""
    command = cmd.strip().split(maxsplit=1)[0].lower()

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
    show_thinking = config["display"]["show_thinking"]

    session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(),
        bottom_toolbar=lambda: make_status_bar(client, config),
        style=PT_STYLE,
    )

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

        try:
            async for chunk in client.send(user_input):
                if chunk.text:
                    full_text.append(chunk.text)
                    console.print(chunk.text, end="", highlight=False)
                if chunk.thought_summary and show_thinking:
                    thought_text.append(chunk.thought_summary)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/]")
            continue
        except Exception as e:
            console.print(f"\n[bold red]Error:[/] {e}")
            continue

        # Render full response as markdown if it contains formatting
        response = "".join(full_text)
        if response and any(c in response for c in ["#", "```", "**", "- ", "1. "]):
            console.print()
            console.print(Markdown(response))
        else:
            console.print()

        if thought_text and show_thinking:
            console.print(f"[dim italic]{''.join(thought_text)}[/]")

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
