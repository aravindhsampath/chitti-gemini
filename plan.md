# Chitti-Gemini Implementation Plan

## Architecture Overview

Chitti is an async terminal-based AI personal assistant powered by Google
Gemini's Interactions API via the official `google-genai` SDK (>= 1.68.0).

- **Python 3.14** — lazy annotations, modern typing, improved asyncio
- **Async-first** — `client.aio.interactions.create()` with `prompt_toolkit`'s
  `prompt_async()` so the event loop is never blocked
- **Streaming** — `stream=True` yields SSE chunks (`content.delta`) printed
  in real-time via `rich` Live/Console
- **Multi-turn** — server-managed via `previous_interaction_id`
- **Identity** — SOUL.md sent as `system_instruction` on every interaction
  (interaction-scoped params are NOT carried over by `previous_interaction_id`)

```
┌──────────────────────────────────────┐
│  Terminal (prompt_toolkit + rich)     │
│  async REPL loop                     │
└──────────────┬───────────────────────┘
               │
               v
┌──────────────────────────────────────┐
│  chitti.py  (entry point)            │
│  - loads .env, chitti.toml, SOUL.md  │
│  - creates client, runs REPL        │
└──────────────┬───────────────────────┘
               │
               v
┌──────────────────────────────────────┐
│  client.py  (thin async wrapper)     │
│  - wraps client.aio.interactions     │
│  - handles streaming + tool loops    │
└──────────────┬───────────────────────┘
               │
               v
┌──────────────────────────────────────┐
│  google-genai SDK (>= 1.68.0)       │
│  client.aio.interactions.create()    │
└──────────────────────────────────────┘
```

## Configuration & Identity

| File           | Purpose                                  | Git-tracked? |
|----------------|------------------------------------------|--------------|
| `.env`         | `GEMINI_API_KEY=...`                     | No           |
| `chitti.toml`  | Model, generation params, tool toggles   | Yes          |
| `SOUL.md`      | Chitti's identity / system instruction    | Yes          |

### chitti.toml schema

```toml
[model]
name = "gemini-3.1-flash-lite-preview"

[generation]
temperature = 1.0
max_output_tokens = 8192
thinking_level = "low"          # minimal | low | medium | high

[assistant]
soul_file = "SOUL.md"

[display]
show_usage = true               # print token counts after each response
show_thinking = false            # show thinking summaries
```

### SOUL.md

Markdown file defining Chitti's personality, rules, and communication style.
Read once at startup, sent as `system_instruction` string on every
`interactions.create()` call. Gemini 3.x implicit caching auto-caches
repeated prefixes — we pay full input tokens only on the first call per
cache window.

**Critical design note**: The Interactions API docs state:
> Only the conversation history (inputs and outputs) is preserved using
> `previous_interaction_id`. The other parameters are interaction-scoped:
> `tools`, `system_instruction`, `generation_config`. You must re-specify
> these parameters in each new interaction.

This means we build a dict of "interaction params" once at startup and
spread it into every `create()` call.

---

## Stage 1: Foundation + Streaming REPL (this PR)

### Files to create/modify

| File               | Action  | What                                          |
|--------------------|---------|-----------------------------------------------|
| `.python-version`  | Update  | Change to `3.14`                              |
| `pyproject.toml`   | Update  | Set python>=3.14, add google-genai, rich,     |
|                    |         | prompt-toolkit, python-dotenv                 |
| `.env`             | Create  | `GEMINI_API_KEY=your-key-here`                |
| `.gitignore`       | Update  | Ensure .env, references/, __pycache__, .venv  |
| `chitti.toml`      | Create  | Default configuration                         |
| `SOUL.md`          | Create  | Chitti's identity prompt                      |
| `client.py`        | Create  | Async Interactions API wrapper                |
| `chitti.py`        | Rewrite | Async entry point + REPL                      |

### 1.1 client.py — Async Interactions wrapper

Thin wrapper around `client.aio.interactions`. Encapsulates the
interaction-scoped params so callers just pass user input.

```python
class ChittiClient:
    """Async wrapper around Gemini Interactions API."""

    def __init__(self, api_key: str, config: dict, soul: str) -> None:
        self._genai = genai.Client(api_key=api_key)
        self._model = config["model"]["name"]
        self._system_instruction = soul
        self._generation_config = {
            "temperature": config["generation"]["temperature"],
            "max_output_tokens": config["generation"]["max_output_tokens"],
            "thinking_level": config["generation"]["thinking_level"],
        }
        self._previous_id: str | None = None

    async def send(self, user_input: str) -> AsyncIterator[StreamChunk]:
        """Send a message, stream back response chunks."""
        stream = await self._genai.aio.interactions.create(
            model=self._model,
            input=user_input,
            system_instruction=self._system_instruction,
            generation_config=self._generation_config,
            previous_interaction_id=self._previous_id,
            stream=True,
        )
        # yields text chunks, captures interaction_id and usage on complete
        ...

    def new_conversation(self) -> None:
        """Reset multi-turn state."""
        self._previous_id = None

    @property
    def previous_interaction_id(self) -> str | None:
        return self._previous_id
```

**Streaming event handling:**

```python
async for chunk in stream:
    match chunk.event_type:
        case "content.delta":
            if chunk.delta.type == "text":
                yield chunk.delta.text        # print immediately
            elif chunk.delta.type == "thought_summary":
                yield thought summary          # if show_thinking
        case "interaction.complete":
            self._previous_id = chunk.interaction.id
            self._last_usage = chunk.interaction.usage
```

### 1.2 chitti.py — Async REPL

```python
async def main() -> None:
    # 1. Load .env via dotenv
    # 2. Load chitti.toml via tomllib (stdlib in 3.14)
    # 3. Read SOUL.md
    # 4. Create ChittiClient
    # 5. Run REPL

async def repl(client: ChittiClient, config: dict) -> None:
    session = PromptSession()
    console = Console()

    while True:
        user_input = await session.prompt_async("you> ")

        if not user_input.strip():
            continue
        if user_input.strip().lower() in ("quit", "exit"):
            break

        # Handle slash commands
        if user_input.startswith("/"):
            handle_command(user_input, client, config)
            continue

        # Stream response with rich Live display
        console.print()
        full_response = []
        async for text_chunk in client.send(user_input):
            console.print(text_chunk, end="")
            full_response.append(text_chunk)
        console.print()

        # Show usage if configured
        if config["display"]["show_usage"]:
            print_usage(client.last_usage)
```

**REPL commands:**
- `quit` / `exit` / Ctrl+C / Ctrl+D — exit
- `/new` — start fresh conversation (reset `previous_interaction_id`)
- `/usage` — show cumulative token usage for the session
- `/model` — show current model name
- `/help` — list available commands

**Terminal rendering with rich:**
- Model responses rendered as markdown via `rich.markdown.Markdown`
- Usage stats in a compact `rich.table.Table`
- Errors in `[bold red]`
- Thinking summaries in `[dim italic]` (if enabled)

### 1.3 Error handling

| Scenario                    | Behavior                                  |
|-----------------------------|-------------------------------------------|
| Missing GEMINI_API_KEY      | Print error + link, exit                  |
| Missing SOUL.md / toml      | Print error, exit                         |
| API error (4xx/5xx)         | Print error, keep REPL alive, skip turn   |
| Network error               | Print error, keep REPL alive              |
| Stream interrupted          | Print partial response, continue          |
| Ctrl+C during streaming     | Cancel current stream, return to prompt   |

### 1.4 Testing approach

- **Unit tests**: Mock `genai.Client` to return fake stream chunks.
  Test that `ChittiClient.send()` yields correct text and captures
  interaction_id/usage from `interaction.complete`.
- **Config tests**: Missing files, malformed TOML, missing keys.
- **Command tests**: `/new` resets state, `/usage` formats output.
- **Integration test** (needs API key): Single round-trip, assert
  completed status and non-empty response.

---

## Stage 2: Tool-Augmented Assistant

The Interactions API natively supports tool declarations. Tools are
interaction-scoped, so we add them to our params dict.

### Built-in tools (API-executed, no local handling needed)

These are added to the `tools` array and the API handles them:

```python
tools = []
if config["tools"]["google_search"]:
    tools.append({"type": "google_search"})
if config["tools"]["code_execution"]:
    tools.append({"type": "code_execution"})
if config["tools"]["url_context"]:
    tools.append({"type": "url_context"})
```

### Custom function calling (requires local dispatch)

When the model returns `status: "requires_action"` with `function_call`
outputs, we execute locally and feed results back:

```python
async def send_with_tools(self, user_input: str) -> AsyncIterator[str]:
    interaction = await self._create(user_input)

    while interaction.status == "requires_action":
        results = []
        for output in interaction.outputs:
            if output.type == "function_call":
                result = await self._dispatch(output.name, output.arguments)
                results.append({
                    "type": "function_result",
                    "call_id": output.id,
                    "name": output.name,
                    "result": result,
                })
        interaction = await self._create(input=results)

    # yield final text outputs
```

### Config additions

```toml
[tools]
google_search = true
code_execution = true
url_context = true
```

### Built-in local functions (Stage 2b)

- `current_datetime` — returns current date/time
- `read_file` — reads a local file
- `run_command` — executes a shell command (with confirmation)
- `web_fetch` — fetches a URL and returns content

---

## Stage 3: Multimodal + Files

### `/upload <path>` command

Uses the google-genai SDK's `client.files.upload()` for media files.

```python
async def handle_upload(self, file_path: str) -> None:
    uploaded = await self._genai.aio.files.upload(file=file_path)
    # Poll until ACTIVE for video
    self._pending_files.append(uploaded)
    # Next interaction includes file reference in input
```

### Supported types
- Images (JPEG, PNG) — describe, analyze, OCR
- PDFs — summarize, Q&A (50MB limit)
- Audio (MP3, WAV) — transcribe, analyze
- Video (MP4) — describe, summarize (poll until ACTIVE)

### Config additions

```toml
[files]
max_upload_size_mb = 100
auto_cleanup = true
```

---

## Stage 4: Advanced Features

### 4a. Deep Research Agent

`/research <topic>` — uses `agent="deep-research-pro-preview-12-2025"`
with `background=True`. Poll via `client.aio.interactions.get(id)` until
status is `completed`.

### 4b. Explicit Caching for Large Contexts

For repeated queries against large uploaded documents, create an explicit
cache via `client.caches.create()` and use `generate_content` with
`cached_content=cache.name` for those specific interactions. This is a
separate code path from the Interactions API.

### 4c. Embeddings + Local RAG

- Embed documents via `gemini-embedding-001` / `gemini-embedding-2-preview`
- Store in local SQLite-vec or ChromaDB
- Retrieve top-k chunks per query
- Include as context in the interaction input

### 4d. Conversation Persistence

- Store interaction IDs + metadata in local SQLite
- `/history` — list past conversations
- `/resume <id>` — continue via `previous_interaction_id`
- `/export` — dump conversation to markdown

---

## Execution Order (Stage 1)

1. Update `.python-version` to 3.14
2. Update `pyproject.toml` — python>=3.14, add deps
3. Update `.gitignore` — ensure .env, references/ covered
4. Create `.env` with placeholder
5. Create `chitti.toml` with defaults
6. Create `SOUL.md` with Chitti's identity
7. Implement `client.py` — ChittiClient async class
8. Implement `chitti.py` — config loading + async REPL
9. `uv sync` and smoke test
10. Write tests
11. Git: commit to feature branch, push, create PR

## Dependencies (Stage 1)

```
google-genai >= 1.68.0
rich
prompt-toolkit
python-dotenv
```

All stdlib: `tomllib`, `asyncio`, `pathlib`, `typing`
