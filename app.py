# app_streamlit.py
import streamlit as st
import pandas as pd
import torch
import plotly.express as px
from gpu_plagiarism import (
    extract_text_from_uploaded,
    check_documents,
    benchmark,
    set_use_gpu,
)
# ========================== PAGE CONFIG ==========================
st.set_page_config(
    page_title="GPU-Accelerated Plagiarism Detection System",
    page_icon="🧠",
    layout="wide",
)
# ========================== HEADER ==========================
st.markdown(
    """
    <h1 style='text-align: center; color: #00BFFF;'>
        🧠 GPU-Accelerated Plagiarism Detection System
    </h1>
    <p style='text-align: center; color: gray; font-size: 16px;'>
        Compare multiple documents (.txt or .pdf) using TF-IDF, SBERT, or Hybrid similarity models powered by GPU acceleration.
    </p>
    """,
    unsafe_allow_html=True,
)
# ========================== SIDEBAR SETTINGS ==========================
st.sidebar.header("⚙️ Settings")
method = st.sidebar.selectbox(
    "Select Detection Method",
    ("TF-IDF (Surface-level)", "SBERT (Semantic)", "Hybrid (TF-IDF + SBERT)"),
)
use_gpu_checkbox = st.sidebar.checkbox("Use GPU if available", value=True)
set_use_gpu(use_gpu_checkbox)
alpha = 0.5
if method.startswith("Hybrid"):
    alpha = st.sidebar.slider(
        "Hybrid Weight (SBERT influence)", 0.0, 1.0, 0.5, 0.1
    )
run_bench = st.sidebar.checkbox("Show CPU vs GPU Runtime", value=True)
st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ Environment Info")
st.sidebar.write(f"**PyTorch Version:** {torch.__version__}")
st.sidebar.write(f"**CUDA Available:** {torch.cuda.is_available()}")
if torch.cuda.is_available():
    st.sidebar.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
# ========================== FILE UPLOAD ==========================
st.markdown("### 📂 Upload Documents (.txt or .pdf)")
uploaded_files = st.file_uploader(
    "Upload 2 or more documents for plagiarism checking:",
    type=["txt", "pdf"],
    accept_multiple_files=True,
)
# ========================== MAIN LOGIC ==========================
if uploaded_files:
    texts, names = extract_text_from_uploaded(uploaded_files)
    if len(texts) < 2:
        st.warning("⚠️ Please upload at least two documents.")
    else:
        st.success(f"✅ {len(texts)} documents uploaded successfully!")

        method_key = (
            "tfidf"
            if "TF-IDF" in method
            else "sbert"
            if "SBERT" in method
            else "hybrid"
        )
        with st.spinner("🔍 Computing document similarity..."):
            pairs, sim_matrix, doc_names = check_documents(
                texts, doc_names=names, method=method_key, alpha=alpha
            )
        df_pairs = pd.DataFrame(pairs)
        # ========================== RESULTS TABLE ==========================
        st.markdown("### 📊 Similarity Results")
        st.dataframe(
            df_pairs.style.background_gradient(cmap="Greens").format(
                {"similarity": "{:.2f}"}
            ),
            use_container_width=True,
        )
        # ========================== BAR CHART VISUALIZATION ==========================
        st.markdown("### 📈Overview")
        top_n = min(10, len(df_pairs))
        top_df = df_pairs.head(top_n).sort_values("similarity", ascending=True)
        fig = px.bar(
            top_df,
            x="similarity",
            y=top_df.apply(lambda r: f"{r['doc1']} ↔ {r['doc2']}", axis=1),
            orientation="h",
            text="similarity",
            color="similarity",
            color_continuous_scale="blues",
            labels={"similarity": "Similarity (%)", "y": "Document Pair"},
            title="Document Similarity Analysis",
            )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(
            showlegend=False,
            xaxis_title="Similarity (%)",
            yaxis_title=None,
            height=500,
            margin=dict(l=120, r=60, t=80, b=40),
            title_font=dict(size=22, color="#00BFFF"),
        )
        st.plotly_chart(fig, use_container_width=True)
        # ========================== PERFORMANCE RUNTIME ==========================
        if run_bench:
            with st.spinner("⚙️ Measuring CPU vs GPU runtime..."):
                cpu_time, gpu_time = benchmark(
                    texts, run_gpu=True, sbert=(method_key != "tfidf")
                )
            st.markdown("### ⏱️ Performance Summary")
            st.write(f"**CPU Runtime:** {cpu_time:.4f} seconds")

            if gpu_time > 0:
                st.write(f"**GPU Runtime:** {gpu_time:.4f} seconds")
                if gpu_time < cpu_time:
                    speedup = cpu_time / gpu_time
                    st.success(f"🚀 GPU accelerated ~{speedup:.1f}× faster than CPU.")
            else:
                st.warning("⚠️ GPU not detected or CUDA not configured.")
