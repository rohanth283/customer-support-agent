# Customer Support Agent Demo

A browser demo of the six-module customer-support workflow described in the project paper. It uses LangGraph to route travel requests to flight, hotel, car-rental, policy, and general-support specialists. Cancellations require confirmation before the mock booking is changed.

The demo runs on Windows, macOS, and Linux with Python 3.10 or newer. It uses sample in-memory bookings and local policy text; it does not connect to real airlines, hotels, or car-rental systems. The checked-in configuration uses Ollama locally. A deterministic rule-based mode and optional Gemini/OpenAI providers are also available.

## Install on a new computer

### 1. Install prerequisites

- Install Python 3.10 or newer.
- Install [Ollama](https://ollama.com/download) for your operating system. Ollama is a separate application; it is not installed by `pip`.
- Have several gigabytes of free disk space for the local model. Model performance depends on the computer's available memory and processor/GPU.

### 2. Clone this repository

```bash
git clone https://github.com/rohanth283/customer-support-agent.git
cd customer-support-agent
```

### 3. Create and activate a Python environment

Create the environment from the project folder:

```bash
python -m venv .venv
```

Activate it using the command for your terminal:

```bash
# macOS / Linux (bash or zsh)
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

If Windows blocks activation, you can skip activation and use `.venv\Scripts\python.exe` in place of `python` in the commands below.

### 4. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file pins the LangGraph packages used by this project. Ollama calls use Python's standard library, so no Ollama Python package is required.

### 5. Download and start the local model

```bash
ollama pull llama3.1:latest
ollama list
```

Keep the Ollama application running. On macOS and Windows, use the Ollama app. On Linux, if the Ollama service is not already running, start it in a separate terminal with:

```bash
ollama serve
```

The sample model setting is in `.env.example`. Copy it to `.env`:

```bash
# macOS / Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

The local Ollama configuration uses `http://127.0.0.1:11434` and `llama3.1:latest`. If you choose a different Ollama model, pull it with `ollama pull <model-name>` and set `OLLAMA_MODEL` to that exact name in `.env`. No cloud API key is needed for Ollama.

### 6. Start the browser demo

From the project folder, with the Python environment active:

```bash
python server.py --port 8001
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001). The header identifies the selected model. Stop the server with `Ctrl+C`. If port 8001 is busy, choose another port, such as `8002`, and open that address instead.

## Try the demo

- `Check my flight BLR-DEL-01` — retrieves a sample flight booking.
- `How long does a refund take?` — retrieves the sample refund policy.
- `Cancel my flight BLR-DEL-01` — asks for approval before cancellation. Choose **Confirm cancellation** to execute the mock action, or **Keep booking** to decline.
- `I need a rental car in Mumbai` — creates a sample car booking.

The right panel shows the graph trace and shared state, including intent, sentiment, urgency, selected specialist, booking ID, and resolution. **New conversation** resets the mock booking database.

## Model settings

The `.env.example` file selects the local Ollama model:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:latest
OLLAMA_TIMEOUT=180
```

The app uses Ollama's local `/api/chat` endpoint with streaming disabled. See the [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion).

To use the deterministic fallback instead, set this in `.env`:

```dotenv
LLM_PROVIDER=rule
```

Gemini is optional. To enable it, set the provider and key in your local `.env`, then restart the server:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY="your-key"
GEMINI_MODEL=gemini-3.8-flash
```

The `.env` file is ignored by Git. Keep API keys there; do not put them in the browser or commit them. The Gemini adapter calls Google's `generateContent` API. See [Gemini API key documentation](https://ai.google.dev/gemini-api/docs/generate-content/api-key).

OpenAI is also supported. Install its optional package and configure `OPENAI_API_KEY` and `LLM_PROVIDER=openai` in `.env`:

```bash
python -m pip install langchain-openai
```

Restart the server after changing `.env` so the selected provider is loaded.

## Run the evaluation

```bash
python evaluate.py
```

This compares the proposed graph with a simpler baseline over 25 synthetic support requests. It writes case-level CSV files and aggregate metrics to `results/`.

The paper reports 76% intent accuracy and 64% completion for the proposed system. The repository's saved evaluation reports 100% and 92%, respectively. These figures differ; when presenting results, state which model/configuration and output you ran.

## How the project maps to the paper

| Paper module | Project implementation |
| --- | --- |
| Conversation state and context | `state.py` and `update_context` in `graph_proposed.py` |
| Intent and sentiment triage | `triage` in `graph_proposed.py`; provider adapters in `llm.py` |
| Specialist routing | Conditional LangGraph edges in `route` |
| Tool execution and human approval | Domain specialists in `graph_proposed.py`; sample tools in `tools/backend.py` |
| Response and policy grounding | Result formatting and policy retrieval |
| Escalation and logging | Final graph node sets resolution and escalation state |

This is a local demonstration, not a production support service. It does not include persistent conversation checkpoints, semantic/vector policy retrieval, authentication, or live service APIs. The booking database and policy text are mock data.

## Project files

- `graph_proposed.py` — six-module LangGraph workflow.
- `graph_baseline.py` — single-graph comparison baseline.
- `state.py` — shared graph state.
- `llm.py` — rule-based, Ollama, Gemini, and optional OpenAI adapters.
- `tools/backend.py` — mock bookings and policy passages.
- `server.py` and `web/` — local browser demo.
- `evaluate.py` and `tests/test_conversations.py` — evaluation runner and sample cases.
