FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the HuggingFace embedding model into the image
# So container starts instantly without downloading on first run
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('Embedding model cached.')"

COPY . .

RUN mkdir -p data vector_store

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
