# Customer Support Agent

A local demonstration of the six-module, stateful LangGraph workflow described in the accompanying paper. It routes flight, hotel, car-rental, policy, and general requests, and requires explicit approval before a cancellation is executed.

The browser demo and command-line evaluation use mock in-memory bookings and a small keyword-based policy collection. They do not contact real travel services. This checkout is configured to use the local Ollama model `llama3.1:latest`; the deterministic rule-based mode is available without any model server or API key.

The Python application is portable across macOS, Windows, and Linux with Python 3.10 or newer. Ollama is a separate application and model runtime, installed outside Python. Model performance depends on the hardware available on the computer.

## Paper architecture in this project

| Paper module | Implementation |
| --- | --- |
| 1. Conversation state and context | `state.py`; `update_context` records messages and the graph trace |
| 2. Intent and sentiment triage | `triage` in `graph_proposed.py`; model interface in `llm.py` |
| 3. Specialist router | Conditional LangGraph edges to flight, hotel, car, policy, or general specialists |
| 4. Tool execution and human approval | `tools/backend.py`; cancellation is blocked until `user_approved` is true |
| 5. Response and policy grounding | Specialist tool results and retrieved policy passages are formatted into a response |
| 6. Escalation and outcome logging | The final graph node sets `escalated` and retains resolution/trace fields |

The local implementation intentionally keeps the paper's mock/offline configuration reproducible. It does not yet include persistent checkpoints, semantic/vector policy retrieval, authenticated live service APIs, or production authorization.

## Install and start

1. Install Python 3.10 or newer and Ollama for your operating system from [ollama.com/download](https://ollama.com/download).
2. In a terminal, install the local model and check that Ollama can serve it:

```bash
ollama pull llama3.1:latest
ollama list
```

Keep the Ollama app running. If its local API is not already running, start it in a separate terminal with `ollama serve`.

3. In Terminal or PowerShell, move into the project directory and create a Python virtual environment:

```bash
cd customer-support-agent
python -m venv .venv
```

Activate it with the command for your shell:

```bash
# macOS / Linux (bash or zsh)
source .venv/bin/activate

# Windows PowerShell (run this instead of the line above)
.\.venv\Scripts\Activate.ps1
```

4. Install Python packages and copy the sample settings file:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, copy the sample settings with `cp .env.example .env`; on Windows PowerShell use `Copy-Item .env.example .env`. The checked-in `.env.example` selects Ollama and `llama3.1:latest`. `.env` is ignored by Git.

5. Start the browser demo:

```bash
python server.py --port 8001
```

Open <http://127.0.0.1:8001>. To stop the server, press `Ctrl+C` in its terminal. To leave the virtual environment, run `deactivate`.

The browser shows the six-step execution trace and shared intent, specialist, sentiment, urgency, booking, and resolution state. Try these demo inputs:

- `Check my flight BLR-DEL-01`
- `How long does a refund take?`
- `Cancel my flight BLR-DEL-01` (then click **Confirm cancellation** or **Keep booking**)
- `I need a rental car in Mumbai`

Click **New conversation** to reset the in-memory sample bookings.

## Language-model modes

### Local Ollama (no cloud key)

Ollama serves the local model at `http://127.0.0.1:11434`. The `.env` file selects it; start the Ollama app/service before starting the web server. Check installed models with `ollama list`.

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:latest
OLLAMA_TIMEOUT=180
```

The adapter calls Ollama's local `/api/chat` endpoint with streaming disabled. See the [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion). No API key or cloud access is needed for model inference.

### Gemini (optional cloud model)

Set these entries in `.env`, then restart the server:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY="your-key"
GEMINI_MODEL=gemini-3.8-flash
```

The key stays on the server and is not sent to the browser. Google documents the API-key header and `generateContent` request [here](https://ai.google.dev/gemini-api/docs/generate-content/api-key).

OpenAI mode is also available if `langchain-openai` is installed and `OPENAI_API_KEY` is set:

```bash
.venv/bin/python -m pip install langchain-openai
```

Then set these entries in `.env`:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY="paste-your-key-here"
```

Start the server:

```bash
.venv/bin/python server.py
```

Use `LLM_PROVIDER=rule` to explicitly force the offline model even when API keys are present. The `.env` file is ignored by Git. Do not paste API keys into the browser or commit them to the repository.

## Run the evaluation

```bash
.venv/bin/python evaluate.py
```

The script evaluates the proposed graph and simpler single-graph baseline over 25 synthetic requests. It writes per-case results and aggregate metrics to `results/proposed_results.csv`, `results/baseline_results.csv`, and `results/summary.json`.

The attached paper reports 76% intent accuracy and 64% completion for its proposed system. The currently checked-out repository's saved `summary.json` instead reports 100% intent accuracy and 92% completion. These figures do not match; present the output produced by the version you actually run and identify which set of results you are showing.

## Interactive Python example

```python
from graph_proposed import PROPOSED_GRAPH

result = PROPOSED_GRAPH.invoke({"user_message": "Cancel my flight BLR-DEL-01"})
print(result["response"])  # requests approval; does not cancel yet

result["user_approved"] = True
result["approval_replay"] = True
result = PROPOSED_GRAPH.invoke(result)
print(result["response"])  # invokes the mock cancellation tool
```

## Project files

- `graph_proposed.py` — six-module LangGraph agent.
- `graph_baseline.py` — single-assistant baseline used for comparison.
- `state.py` — shared typed graph state.
- `llm.py` — offline, local Ollama, Gemini REST, and optional OpenAI language-model adapters.
- `tools/backend.py` — mock booking tools and policy passages.
- `server.py` and `web/` — local browser demonstration.
- `evaluate.py` and `tests/test_conversations.py` — evaluation runner and synthetic cases.
