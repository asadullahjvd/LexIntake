FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ingest the category corpus into ChromaDB at build time so the container
# starts with data already loaded. Re-run manually if you add categories.
RUN python -m app.rag.ingest

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
