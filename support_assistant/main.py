from fastapi import FastAPI
from pydantic import BaseModel, Field

from graph import AssistantResponse, ask

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0.0",
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running.",
        "mock_llm": True,
        "endpoint": "POST /ask",
    }


@app.post("/ask", response_model=AssistantResponse)
def ask_endpoint(request: AskRequest):
    return ask(request.query)
