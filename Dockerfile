FROM python:3.10-slim

RUN useradd -m -u 1000 appuser

RUN apt-get update && apt-get install -y \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1 — PyTorch CPU (must be first, special index URL)
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Step 2 — PyTorch Geometric
RUN pip install --no-cache-dir \
    torch-geometric==2.7.0 \
    torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.11.0+cpu.html

# Step 3 — All other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4 — Copy app files (NO large CSV files)
COPY app.py .
COPY config.json .
COPY df_sample.csv .
COPY gat_model.pth .
COPY graph_data.pkl .

# Step 5 — Streamlit config for HF Spaces (port 7860)
RUN mkdir -p /app/.streamlit && echo '\
[server]\n\
port = 7860\n\
address = "0.0.0.0"\n\
headless = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /app/.streamlit/config.toml

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]