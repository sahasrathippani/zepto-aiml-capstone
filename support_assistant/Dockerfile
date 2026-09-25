FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The embedding model is downloaded by sentence-transformers on first startup.
# This Dockerfile is locally buildable/runnable; no LLM API key is required.
ENV MOCK_LLM=1

RUN python ingestion.py

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
