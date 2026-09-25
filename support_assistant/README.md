# Module 3 — Zepto Support Assistant

A small GenAI/RAG-style support service using Zepto's supplied policy corpus.

## Graded baseline

The graded path is **fully offline mock mode**.

`MOCK_LLM` is unset by default, which is treated as mock mode. You can also explicitly use:

```powershell
$env:MOCK_LLM="1"
```

No LLM API key or paid service is required for the graded baseline.

Embeddings are generated locally using:

```text
sentence-transformers
all-MiniLM-L6-v2
```

and stored in ChromaDB.

## Folder structure

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── data/
│   └── chroma_db/              # generated
├── ingestion.py
├── prompts.py
├── graph.py
├── main.py
├── run_examples.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Setup

Open PowerShell in this folder:

```powershell
pip install -r requirements.txt
```

## Build the vector store

Run:

```powershell
python ingestion.py
```

This loads all 8 exact supplied policy documents, uses one chunk per document, creates local `all-MiniLM-L6-v2` embeddings, and stores them in the ChromaDB collection `zepto_policy`.

## Test the LangGraph flow

Run:

```powershell
python run_examples.py
```

The first example contains a policy keyword and routes through:

```text
START
  ↓
classify_intent
  ↓
retrieve_and_answer
  ↓
END
```

The second example is unrelated to Zepto policy and routes through:

```text
START
  ↓
classify_intent
  ↓
direct_answer
  ↓
END
```

### Example JSON responses

Policy-style query:

```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume.",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

General query:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

The exact retrieved text may vary only in the snippet returned by the embedding search.

## FastAPI

Run:

```powershell
uvicorn main:app --reload
```

Then send:

```text
POST /ask
```

with:

```json
{
  "query": "How long does Zepto delivery take?"
}
```

or:

```json
{
  "query": "Who is the president of India?"
}
```

The response schema is:

```json
{
  "answer": "string",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

## Structured prompt

`prompts.py` contains the required role-context-task-format-length skeleton, an explicit negative constraint, and a few-shot example.

The mock baseline does not call an LLM. The prompt exists for the optional real-LLM extension.

## Architecture

```text
                 ┌──────────────────┐
                 │  8 policy docs   │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │    ingestion.py  │
                 │ chunk + embed    │
                 └────────┬─────────┘
                          ↓
                 ┌──────────────────┐
                 │     ChromaDB     │
                 │   zepto_policy   │
                 └────────┬─────────┘
                          ↑
                          │ top-3 retrieval
                          │
User → FastAPI → LangGraph classify_intent
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
 retrieve_and_answer      direct_answer
          │                   │
          ↓                   ↓
  top retrieved chunk     fixed mock answer
          │                   │
          └─────────┬─────────┘
                    ↓
             Pydantic response
```

### Pipeline stages

**1. Ingestion**

`ingestion.py` reads the eight files in `docs/`, uses one chunk per document, and assigns IDs such as `doc_01`.

**2. Embedding**

`sentence-transformers` with `all-MiniLM-L6-v2` converts each chunk to a local embedding. ChromaDB stores the embeddings and document metadata.

**3. Retrieval**

The `retrieve_and_answer` LangGraph node embeds the query and asks ChromaDB for the top 3 chunks using cosine similarity.

**4. Generation**

In the required mock mode, the retrieved answer is generated deterministically from the top chunk:

```text
Based on the retrieved context: <top chunk snippet>
```

For a general question, `direct_answer` returns the fixed policy-only response.

`MOCK_LLM` affects the generation/classification branch. Retrieval itself always runs for policy questions because embeddings and ChromaDB are local and do not require an LLM API.

## Docker

Build:

```powershell
docker build -t zepto-support-assistant .
```

Run:

```powershell
docker run -p 7860:7860 zepto-support-assistant
```

Then call:

```text
POST http://localhost:7860/ask
```

The container uses `MOCK_LLM=1`, so no LLM API key is required.

## Optional real LLM

The assignment treats the real-LLM path as optional and ungraded. The structured prompt in `prompts.py` is ready to be used as the prompt for a real provider.

Do not commit API keys to GitHub.
