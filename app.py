
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from groq import Groq
import pickle
import time
import json
import os

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# ── Load config ───────────────────────────────────────────
PATH = os.path.dirname(os.path.abspath(__file__)) + "/"

@st.cache_resource
def load_everything():
    # Load data
    df = pd.read_csv(PATH + "df_sample.csv")
    
    # Load graph
    with open(PATH + "graph_data.pkl", "rb") as f:
        graph_data = pickle.load(f)
    
    # Load config
    config_path = PATH + "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
        "model": "llama-3.1-8b-instant",
        "week4_status": "complete"
    }
    
    return df, graph_data, config

# ── GAT Model ─────────────────────────────────────────────
class FraudGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(FraudGAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1, dropout=0.3)
        self.dropout = torch.nn.Dropout(0.3)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

@st.cache_resource
def load_model(_graph_data):
    model = FraudGAT(input_dim=_graph_data.num_features,
                     hidden_dim=32, output_dim=2)
    try:
        model.load_state_dict(torch.load(PATH + "gat_model.pth",
                              map_location="cpu"))
    except:
        pass
    model.eval()
    return model

# ── LLM Explainer ─────────────────────────────────────────
def explain_transaction(row, prob, client):
    prompt = f"""You are a fraud analyst. Explain in exactly 3 bullet points why this transaction is suspicious.
Transaction Amount: ${row.get("TransactionAmt", 0):.2f}
Email Domain: {row.get("P_emaildomain", "unknown")}
Addresses linked to card (C1): {row.get("C1", 0)}
Cards linked to email (C2): {row.get("C2", 0)}
Fraud Probability: {prob:.1%}
Be specific and professional. 3 bullet points only."""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250
    )
    return response.choices[0].message.content

# ── Main App ──────────────────────────────────────────────
def main():
    st.title("🔍 Real-Time Fraud Detection System")
    st.markdown("*Powered by Graph Attention Network + LLM Explainability*")
    st.divider()

    # Load everything
    with st.spinner("Loading model and data..."):
        df, graph_data, config = load_everything()
        model = load_model(graph_data)
        groq_api_key = os.environ.get("GROQ_API_KEY") or config.get("groq_api_key")
        groq_client = Groq(api_key=groq_api_key)

    # ── Sidebar ───────────────────────────────────────────
    st.sidebar.title("⚙️ Controls")
    threshold = st.sidebar.slider("Fraud Alert Threshold", 0.1, 0.9, 0.5, 0.05)
    n_transactions = st.sidebar.slider("Transactions to scan", 5, 50, 10)
    run_scan = st.sidebar.button("🚀 Start Scanning", type="primary")

    # ── Metrics Row ───────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    total_box    = col1.metric("Total Scanned", "0")
    fraud_box    = col2.metric("Fraud Alerts", "0")
    legit_box    = col3.metric("Legitimate", "0")
    rate_box     = col4.metric("Fraud Rate", "0%")

    st.divider()

    # ── Transaction Feed ──────────────────────────────────
    st.subheader("📡 Live Transaction Feed")
    feed_placeholder = st.empty()

    # ── Alerts Panel ──────────────────────────────────────
    st.subheader("🚨 Fraud Alerts")
    alert_placeholder = st.empty()

    if run_scan:
        # Sample transactions
        sample = df.sample(n=n_transactions, random_state=42).reset_index(drop=True)
        
        transactions_log = []
        fraud_alerts = []
        total = 0
        frauds = 0

        for idx, row in sample.iterrows():
            total += 1
            
            # Get fraud probability from model
            with torch.no_grad():
                out = model(graph_data.x, graph_data.edge_index)
                probs = torch.softmax(out, dim=1)
                node_idx = min(idx, graph_data.num_nodes - 1)
                fraud_prob = probs[node_idx][1].item()

            is_fraud = fraud_prob >= threshold
            if is_fraud:
                frauds += 1

            # Add to log
            transactions_log.append({
                "ID": row["TransactionID"],
                "Amount": f"${row['TransactionAmt']:.2f}",
                "Email": row.get("P_emaildomain", "unknown"),
                "Fraud Prob": f"{fraud_prob:.1%}",
                "Status": "🚨 FRAUD" if is_fraud else "✅ Legit"
            })

            # Update metrics
            col1.metric("Total Scanned", total)
            col2.metric("Fraud Alerts", frauds)
            col3.metric("Legitimate", total - frauds)
            col4.metric("Fraud Rate", f"{frauds/total:.1%}")

            # Update feed
            feed_df = pd.DataFrame(transactions_log)
            feed_placeholder.dataframe(feed_df, use_container_width=True)

            # Generate explanation for fraud
            if is_fraud:
                with st.spinner(f"🤖 Generating explanation for Transaction {row['TransactionID']}..."):
                    explanation = explain_transaction(row, fraud_prob, groq_client)
                
                fraud_alerts.append({
                    "id": row["TransactionID"],
                    "amount": row["TransactionAmt"],
                    "prob": fraud_prob,
                    "explanation": explanation
                })

                # Show alerts
                alert_html = ""
                for alert in fraud_alerts[-3:]:
                    explanation_clean = alert["explanation"].replace("`", "").replace("<", "").replace(">", "")
                    alert_html += f"""
<div style="background:#ff4b4b22;border-left:4px solid #ff4b4b;padding:12px;border-radius:8px;margin-bottom:10px;">
<b>🚨 Transaction {alert["id"]}</b> — Amount: ${alert["amount"]:.2f} — Risk: {alert["prob"]:.1%}<br><br>
{explanation_clean}
</div>"""
                alert_placeholder.markdown(alert_html, unsafe_allow_html=True)

            time.sleep(0.3)

        st.success(f"✅ Scan complete! Found {frauds} fraud alerts out of {total} transactions.")

if __name__ == "__main__":
    main()
