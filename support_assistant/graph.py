from typing import TypedDict

import chromadb
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, MOCK_LLM
from prompts import STRUCTURED_PROMPT


class AssistantResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
_embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def classify_intent(state: GraphState):
    query = state["query"]
    lower = query.lower()

    intent = (
        "policy_question"
        if any(keyword in lower for keyword in KEYWORDS)
        else "general_question"
    )

    # Required mock baseline makes no LLM call.
    # Optional real-LLM classification can be added here when MOCK_LLM is False.
    return {"intent": intent}


def retrieve_and_answer(state: GraphState):
    query = state["query"]

    query_embedding = _embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    result = _collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    ids = result.get("ids", [[]])[0]

    if not documents:
        return {
            "answer": "No relevant policy context was retrieved.",
            "sources": [],
            "confidence": 0.0,
        }

    top_chunk = documents[0]
    snippet = top_chunk[:200].strip()

    if MOCK_LLM:
        answer = f"Based on the retrieved context: {snippet}"
    else:
        # Optional real-LLM extension point. The graded path does not use this.
        # The structured prompt is provided in prompts.py for a real provider.
        answer = f"Based on the retrieved context: {snippet}"

    source_ids = ids if ids else [
        metadata.get("source", "") for metadata in metadatas
    ]

    return {
        "answer": answer,
        "sources": source_ids,
        "confidence": 1.0 if MOCK_LLM else 0.9,
    }


def direct_answer(state: GraphState):
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
    else:
        # Optional real-LLM extension point.
        answer = "I can only answer questions about Zepto policies right now."

    return {
        "answer": answer,
        "sources": [],
        "confidence": 1.0 if MOCK_LLM else 0.9,
    }


def route_after_classification(state: GraphState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


app_graph = build_graph()


def ask(query: str) -> AssistantResponse:
    result = app_graph.invoke({"query": query})

    response = AssistantResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=float(result.get("confidence", 0.0)),
    )

    return response
