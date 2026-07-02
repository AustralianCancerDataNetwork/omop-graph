FROM python:3.12-slim
RUN pip install --no-cache-dir ".[postgres,emb,pgvector,faiss-cpu]"
WORKDIR /workspace
