GPU-Accelerated Plagiarism Checker (Streamlit UI)
================================================

What this project contains:
- app_streamlit.py : Streamlit frontend + glue code
- gpu_plagiarism.py : Core TF-IDF and cosine similarity logic using PyTorch (uses GPU if available)
- requirements.txt : Python dependencies
- sample_docs/ : sample text files for testing

How to run (Windows / Linux / macOS):
1. Extract this folder.
2. Create a virtual environment (recommended):
   python -m venv venv
   # Windows (cmd):
   venv\Scripts\activate.bat
   # Linux / macOS:
   source venv/bin/activate
3. Install dependencies:
   pip install -r requirements.txt
   # If you want GPU support, install a CUDA-enabled PyTorch wheel from https://pytorch.org/get-started/locally
4. Run Streamlit app:
   streamlit run app_streamlit.py
5. Open the local URL Streamlit prints in your browser (usually http://localhost:8501)

Notes:
- Upload plain .txt files. The app computes pairwise TF-IDF cosine similarity and shows a similarity table and heatmap.
- Use the benchmark button to compare CPU vs GPU time (if CUDA is available).
# GPU_based-plagiarism--Detection-and-monitoring
