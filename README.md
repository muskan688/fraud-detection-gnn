# 🔍 Real-Time Fraud Detection System
### Graph Attention Network + LLM Explainability

[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face_Spaces-yellow)](https://huggingface.co/spaces/muskan688/fraud_detection)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-ff4b4b)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Deployed-2496ED)](https://docker.com)

> An end-to-end fraud detection pipeline trained on the IEEE-CIS dataset (590K transactions), combining Graph Neural Networks for pattern detection with LLM-powered explainability for human-readable fraud alerts.

---

## 🚀 Live Demo

**Try it here:** [https://huggingface.co/spaces/muskan688/fraud_detection](https://huggingface.co/spaces/muskan688/fraud_detection)

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Dataset | IEEE-CIS Fraud Detection (590K rows, 220 features) |
| Graph Size | 50K nodes, 397K edges |
| Model | Graph Attention Network (GAT) |
| Parameters | 3,078 |
| **AUROC** | **0.74** |
| Explainability | Llama 3 via Groq API |
| Deployment | Docker + Hugging Face Spaces |

---

## 🏗️ Architecture

```
Raw Transactions (IEEE-CIS)
        ↓
Data Cleaning & Feature Engineering
        ↓
Graph Construction (PyTorch Geometric)
   [Nodes = Transactions | Edges = Shared cards/emails]
        ↓
Graph Attention Network (GAT)
   [Learns fraud patterns from transaction relationships]
        ↓
Fraud Probability Score
        ↓
LLM Explainability (Llama 3 via Groq)
   [Generates human-readable fraud alerts]
        ↓
Streamlit Dashboard (Deployed via Docker)
```

---

## 🛠️ Tech Stack

**Machine Learning**
- PyTorch + PyTorch Geometric — GAT model training
- Scikit-learn — preprocessing, evaluation metrics
- NetworkX — graph analysis

**LLM & Explainability**
- Groq API — ultra-fast inference
- Llama 3.1 8B — natural language fraud explanations

**Web App & Deployment**
- Streamlit — interactive dashboard
- Docker — containerization
- Hugging Face Spaces — cloud deployment

---

## 📁 Project Structure

```
fraud-detection-gnn/
├── app.py                    # Streamlit dashboard
├── Dockerfile                # Container configuration
├── requirements.txt          # Python dependencies
├── gat_model.pth             # Trained GAT model weights
├── df_sample.csv             # Sample transactions for demo
├── 01_data_exploration.ipynb # EDA notebook
└── .gitignore
```

---

## 🗓️ Development Journey (6 Weeks)

| Week | Task | Status |
|------|------|--------|
| 1 | Data loading & cleaning (IEEE-CIS, 590K rows) | ✅ |
| 2 | Graph construction (50K nodes, 397K edges) | ✅ |
| 3 | GAT model training (AUROC 0.74) | ✅ |
| 4 | LLM explainability (Llama 3 via Groq) | ✅ |
| 5 | Streamlit dashboard | ✅ |
| 6 | Docker + Hugging Face Spaces deployment | ✅ |

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/muskan688/fraud-detection-gnn.git
cd fraud-detection-gnn

# Create conda environment
conda create -n fraud_gnn python=3.10
conda activate fraud_gnn

# Install dependencies
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.7.0
pip install -r requirements.txt

# Add your Groq API key to config.json
# {"groq_api_key": "your_key_here", "model": "llama-3.1-8b-instant"}

# Run the app
streamlit run app.py
```

---

## 🔐 Security Note

API keys are managed via environment variables and Hugging Face Secrets — never hardcoded in the codebase.

---

## 👩‍💻 Author

**Muskan** — Data Science Student  
📎 [GitHub](https://github.com/muskan688) | 🤗 [Hugging Face](https://huggingface.co/muskan688)