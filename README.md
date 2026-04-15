# Chitti Gemini

AI personal assistant powered by Google Gemini's Interactions API.

## Setup

```bash
# Install Python 3.14 via uv
uv python install 3.14

# Install dependencies
uv sync

# Add your API key
echo 'GEMINI_API_KEY=your-key-here' > .env
```

## Usage

```bash
uv run chitti
```

### REPL Commands

| Command  | Description                |
|----------|----------------------------|
| `/new`   | Start a new conversation   |
| `/usage` | Show session token usage   |
| `/model` | Show current model         |
| `/help`  | List commands              |
| `quit`   | Exit                       |

## Configuration

Edit `chitti.toml` to change model, temperature, and display settings.
Edit `SOUL.md` to change Chitti's personality and system instruction.

## Tests

```bash
uv run pytest
```
