FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/support_assistant/data
RUN mkdir -p /app/support_assistant/docs

# Copy the app files
COPY support_assistant /app/support_assistant/

# We need to run ingestion inside the container if it doesn't carry over the data folder properly,
# but assuming the docs are copied, let's run ingestion at build time to populate the persistent store
ENV PYTHONPATH=/app
RUN python -m support_assistant.ingestion

EXPOSE 8000

ENV MOCK_LLM=1

# Run FastAPI via Uvicorn
CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
