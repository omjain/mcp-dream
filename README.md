# 🌙 mcp-dream

> Let AI dream on your problems while you sleep.

`mcp-dream` is an MCP server + agent loop that runs autonomous overnight (or over-lunch) exploration sessions on whatever you point it at. You give it a goal, go to bed, and wake up to a morning report.

```
You at 11pm:  "Dream about my codebase tonight. Surprise me."
You at 7am:   "Hey Claude, what did you dream about?"
Claude:       (pulls back a markdown report with 3 ideas, 4 observations,
               and 2 questions for you)
```

It plugs into Claude Desktop, Cursor, or any MCP-compatible client, so you can start and read dreams from inside your normal AI chat.

---

## Why

Most AI tools want a **specific question**. Sometimes you don't have one. Sometimes the most useful thing an agent can do is wander — pulling threads, reading your stuff, surprising you with what it noticed.

`mcp-dream` is built around the idea that **aimless, time-boxed exploration is its own product category**. The hard parts are the guardrails: how do you let an agent loop for hours without it draining your API budget or doing something stupid?

This repo's answer:

- **Three independent budget caps** (time, dollars, iterations) — all on by default. The first to trip stops the dream.
- **Read-only by default.** Shell and code execution are off until you flip a switch.
- **File access is sandboxed** to paths you explicitly list.
- **Every dream is logged** to `~/.mcp-dream/dreams/<id>/` with the goal, notes, stats, and the final report.

---

## Quickstart (60 seconds)

```bash
pip install mcp-dream
export ANTHROPIC_API_KEY=sk-...
mcp-dream init                                # writes ~/.mcp-dream/config.toml
mcp-dream dream "surprise me with three weird side-project ideas"
```

That last command runs a dream synchronously and prints the report. Once you're convinced it's not going to set anything on fire, wire it into Claude Desktop:

```bash
mcp-dream install   # prints a config snippet to paste into claude_desktop_config.json
```

Restart Claude Desktop and you'll see `mcp-dream` in your tools. Then just say things like:

> "Dream about good benchmarks for evaluating RAG systems. Three hours, two dollars max."

---

## How it works

```
┌─────────────────────────────────────────┐
│  Your laptop                            │
│                                         │
│  ┌──────────────┐   spawns   ┌────────┐ │
│  │ MCP server   │──────────▶ │ Dream  │ │
│  │ (stdio)      │            │ loop   │ │
│  └──────▲───────┘            └────┬───┘ │
│         │ tool calls               │    │
│         │                  writes  │    │
│  ┌──────┴───────┐                  ▼    │
│  │Claude Desktop│      ~/.mcp-dream/    │
│  │/ Cursor / …  │      ├ config.toml    │
│  └──────────────┘      └ dreams/<id>/   │
│                          ├ report.md    │
│                          └ metadata.json│
└─────────────────────────────────────────┘
                    │
                    ▼
          api.anthropic.com  (or local Ollama)
```

The MCP server exposes five tools to your client:

| Tool                          | What it does                                     |
| ----------------------------- | ------------------------------------------------ |
| `dream_start(goal, …)`        | Kick off a dream in the background.              |
| `dream_status(dream_id?)`     | Check how a running dream is going.              |
| `dream_list()`                | List past dreams.                                |
| `dream_read_report(dream_id)` | Pull the morning report back into chat.          |
| `dream_stop(dream_id)`        | Politely end a dream early.                      |

Behind the scenes the dream loop is just: pick a tool → run it → save a note → repeat → write a report. The agent picks its own threads. You picked the rope.

---

## What the agent can do

The agent always has a `note` tool to record findings. Other tools are opt-in via `~/.mcp-dream/config.toml`:

| Tool              | Default | Notes                                             |
| ----------------- | ------- | ------------------------------------------------- |
| `web_search`      | on      | Free DuckDuckGo, no API key needed.               |
| `read_file`       | on*     | *Requires `allowed_read_paths` set. Sandboxed.    |
| `list_dir`        | on*     | Same restriction.                                 |
| `shell`           | **off** | Runs arbitrary shell. Opt in only if you mean it. |
| `run_python`      | **off** | Same.                                             |

To let the agent poke around your code, add to your config:

```toml
[tools]
allowed_read_paths = ["/Users/you/Projects/my-repo"]
```

That's it. The agent now has read-only access to that tree and nothing else.

---

## The morning report

Reports are markdown with five sections. The agent fills them in based on its notes:

```markdown
# 🌙 Morning report

## 🎯 What I dreamed about
(one paragraph restating your goal)

## 💡 Ideas I had
(3–5 concrete ideas, 1–3 sentences each)

## 🔍 Things I noticed
(observations and surprises from research)

## ❓ Questions for you
(things that would make the next dream better)

## 📌 What I'd do next
(the thread it would pull tomorrow)
```

The report is the product. The dream loop is the engine. If you read one section, read the questions — they're usually where the best stuff is.

---

## Configuration

`~/.mcp-dream/config.toml`:

```toml
[provider]
name = "anthropic"               # or "ollama" for local/free
anthropic_model = "claude-opus-4-7"
ollama_model = "llama3.2"
ollama_host = "http://localhost:11434"

[tools]
web_search = true
read_files = true
allowed_read_paths = []           # e.g. ["/Users/you/Projects/foo"]
shell = false                     # DANGEROUS — opt in carefully
code_execution = false            # DANGEROUS — opt in carefully

[budget]
max_hours = 2.0
max_dollars = 2.0
max_iterations = 30
```

Per-dream overrides are possible from the MCP tool (`dream_start(goal, max_hours=3)`) or the CLI (`mcp-dream dream "…" --hours 3`).

---

## Local-only (no API key)

Don't have an Anthropic API key? Use Ollama:

```bash
brew install ollama  # or whatever your platform wants
ollama pull llama3.2
pip install "mcp-dream[ollama]"
```

Then in your config:

```toml
[provider]
name = "ollama"
ollama_model = "llama3.2"
```

Local models can't do reliable tool-use, so the dream becomes an iterative reflection loop instead — slower, smaller, but free.

---

## CLI reference

```bash
mcp-dream init                 # create config
mcp-dream config               # show effective config
mcp-dream dream "GOAL"         # run a dream synchronously
mcp-dream list                 # list past dreams
mcp-dream report DREAM_ID      # print a saved report
mcp-dream install              # print Claude Desktop snippet
mcp-dream-server               # run the MCP server (used by clients, not you)
```

---

## Safety notes

This thing runs an LLM in a loop. Some honest cautions:

1. **The dollar cap is an estimate**, not a billing guarantee. Token pricing changes; check your actual API dashboard the first few times.
2. **`shell` and `run_python` are dangerous.** They run arbitrary code with your user privileges. Enable only on a machine and in a context where you'd be okay with that.
3. **`allowed_read_paths` is enforced by `Path.resolve()`** before any filesystem touch — symlinks and `..` traversal can't escape it. That's a deliberate design choice; please report any way to defeat it.
4. **The agent decides when to stop calling tools.** If you give it a vague goal and big caps, expect it to use them.

---

## Roadmap

Things this project doesn't do yet but probably should:

- [ ] Persistent background daemon (currently dreams die with the MCP server process)
- [ ] Native scheduled dreams (cron-style "every Sunday night")
- [ ] Multi-step dream sequences (one dream feeds the next)
- [ ] Pluggable tool ecosystem (community-contributed tools)
- [ ] Token-counter integration for live budget readout in `dream_status`

PRs welcome on any of these.

---

## Acknowledgements

Built on [Model Context Protocol](https://modelcontextprotocol.io/), the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python), and the patience of people who watched it sometimes wander in circles before getting the report shape right.

---

## License

MIT. See [LICENSE](LICENSE).
