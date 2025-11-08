# gpu_plagiarism.py
import re
import time
from collections import Counter
from typing import List, Optional, Tuple, Sequence, Any
import numpy as np
import torch
# Optional imports (lazy) - sentence_transformers and fitz may not be installed initially
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except Exception:
    SentenceTransformer = None
    _HAS_SBERT = False
try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    fitz = None
    _HAS_FITZ = False
# ----------------------------
# Configuration / Globals
# ----------------------------
USE_GPU = True  # global flag; set via set_use_gpu()
DEFAULT_SBERT_MODEL = "all-MiniLM-L6-v2"
MAX_VOCAB = 20000  # for TF-IDF
# ----------------------------
# Utility / config functions
# ----------------------------
def set_use_gpu(flag: bool):
    """Enable or disable GPU usage globally for TF-IDF path. SentenceTransformer will use device param separately."""
    global USE_GPU
    USE_GPU = bool(flag)
def device_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"
# ----------------------------
# File/Text helpers
# ----------------------------
def extract_text_from_uploaded(uploaded_files: Sequence[Any]) -> Tuple[List[str], List[str]]:
    texts = []
    names = []
    for f in uploaded_files:
        # f must support .name and .read()
        name = getattr(f, "name", None) or "file"
        names.append(name)
        raw = f.read()
        # ensure bytes
        if isinstance(raw, str):
            # already a decoded string
            texts.append(raw)
            continue
        if name.lower().endswith(".pdf") and _HAS_FITZ:
            try:
                pdf = fitz.open(stream=raw, filetype="pdf")
                pages_text = []
                for page in pdf:
                    pages_text.append(page.get_text())
                pdf.close()
                texts.append("\n".join(pages_text))
            except Exception:
                # fallback to plain decode
                try:
                    texts.append(raw.decode("utf-8", errors="ignore"))
                except Exception:
                    texts.append(raw.decode("latin-1", errors="ignore"))
        else:
            # plain text file
            try:
                texts.append(raw.decode("utf-8"))
            except Exception:
                texts.append(raw.decode("latin-1", errors="ignore"))
    return texts, names
# ----------------------------
# Tokenization & TF-IDF (PyTorch)
# ----------------------------
def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1]
    return tokens
def build_vocab(docs_tokens: List[List[str]], max_vocab: int = MAX_VOCAB) -> dict:
    counter = Counter()
    for toks in docs_tokens:
        counter.update(toks)
    most = counter.most_common(max_vocab)
    vocab = {w: i for i, (w, _) in enumerate(most)}
    return vocab
def docs_to_tf_matrix(docs_tokens: List[List[str]], vocab: dict) -> torch.Tensor:
    D = len(docs_tokens)
    V = len(vocab)
    tf = torch.zeros((D, V), dtype=torch.float32)
    for i, toks in enumerate(docs_tokens):
        ctr = Counter(toks)
        for w, c in ctr.items():
            idx = vocab.get(w)
            if idx is not None:
                tf[i, idx] = float(c)
    return tf
def compute_tfidf_from_texts(docs: List[str], use_gpu: Optional[bool] = None) -> Tuple[np.ndarray, List[str]]:
    if use_gpu is None:
        use_gpu = USE_GPU
    docs_tokens = [simple_tokenize(d) for d in docs]
    vocab = build_vocab(docs_tokens, max_vocab=MAX_VOCAB)
    if len(vocab) == 0:
        D = len(docs)
        return np.zeros((D, D), dtype=float), [f"Doc{i+1}" for i in range(D)]
    tf = docs_to_tf_matrix(docs_tokens, vocab)  # CPU tensor
    if use_gpu and torch.cuda.is_available():
        tf = tf.cuda()
    # IDF weighting
    D = tf.size(0)
    df = torch.sum((tf > 0).float(), dim=0)
    idf = torch.log((1.0 + float(D)) / (1.0 + df)) + 1.0
    tfidf = tf * idf.unsqueeze(0)
    # normalize rows
    norms = torch.norm(tfidf, dim=1, keepdim=True).clamp(min=1e-9)
    tfidf = tfidf / norms
    sim = torch.matmul(tfidf, tfidf.t()).clamp(min=0.0, max=1.0)
    sim_cpu = sim.cpu().numpy()
    names = [f"Doc{i+1}" for i in range(len(docs))]
    return sim_cpu, names
# ----------------------------
# SBERT (Sentence-BERT) embedding similarity
# ----------------------------
def compute_sbert_similarity(docs: List[str],
                             model_name: str = DEFAULT_SBERT_MODEL,
                             device: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
    if not _HAS_SBERT:
        raise ImportError("sentence-transformers is not installed. Run: pip install sentence-transformers")
    if device is None:
        device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
    # load model (it will use device param)
    model = SentenceTransformer(model_name, device=device)
    # encode (returns torch tensor if convert_to_tensor True)
    embeddings = model.encode(docs, convert_to_tensor=True, device=device)  # tensor on device
    # normalize embeddings
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True).clamp(min=1e-9)
    # cosine similarity matrix
    sim = torch.mm(embeddings, embeddings.t()).clamp(min=0.0, max=1.0)
    sim_cpu = sim.cpu().numpy()
    names = [f"Doc{i+1}" for i in range(len(docs))]
    return sim_cpu, names
# ----------------------------
# Hybrid similarity (weighted average)
# ----------------------------
def compute_hybrid_similarity(docs: List[str],
                              alpha: float = 0.5,
                              sbert_model: str = DEFAULT_SBERT_MODEL,
                              use_gpu_sbert: Optional[bool] = None,
                              use_gpu_tfidf: Optional[bool] = None) -> Tuple[np.ndarray, List[str]]:
    if use_gpu_sbert is None:
        use_gpu_sbert = USE_GPU
    if use_gpu_tfidf is None:
        use_gpu_tfidf = USE_GPU

    tfidf_sim, _ = compute_tfidf_from_texts(docs, use_gpu=use_gpu_tfidf)
    try:
        sbert_sim, _ = compute_sbert_similarity(docs, model_name=sbert_model,
                                                device=("cuda" if (use_gpu_sbert and torch.cuda.is_available()) else "cpu"))
    except ImportError:
        # if SBERT unavailable, fallback to TF-IDF only
        sbert_sim = tfidf_sim.copy()
    # Ensure numeric and same shape
    tfidf_sim = np.array(tfidf_sim, dtype=float)
    sbert_sim = np.array(sbert_sim, dtype=float)
    # normalize both to [0,1] (they already are, but numeric stability)
    tfidf_sim = np.clip(tfidf_sim, 0.0, 1.0)
    sbert_sim = np.clip(sbert_sim, 0.0, 1.0)
    final = alpha * sbert_sim + (1.0 - alpha) * tfidf_sim
    names = [f"Doc{i+1}" for i in range(len(docs))]
    return final, names
# ----------------------------
# Utilities for producing pairs list like before
# ----------------------------
def similarity_matrix_to_pairs(sim_matrix: np.ndarray, doc_names: Optional[List[str]] = None):
    D = sim_matrix.shape[0]
    if doc_names is None:
        doc_names = [f"Doc{i+1}" for i in range(D)]
    pairs = []
    for i in range(D):
        for j in range(i + 1, D):
            score = float(sim_matrix[i, j])
            percent = round(score * 100.0, 2)
            pairs.append({'doc1': doc_names[i], 'doc2': doc_names[j], 'similarity': percent})
    pairs = sorted(pairs, key=lambda x: x['similarity'], reverse=True)
    return pairs
# ----------------------------
# Benchmarking utilities
# ----------------------------
def benchmark(docs: List[str], run_gpu: bool = True, sbert: bool = False) -> Tuple[float, float]:
    # CPU run
    set_use_gpu(False)
    t0 = time.time()
    if sbert:
        try:
            _ = compute_sbert_similarity(docs, device="cpu")
        except ImportError:
            # SBERT not installed -> fallback to TF-IDF
            _ = compute_tfidf_from_texts(docs, use_gpu=False)
    else:
        _ = compute_tfidf_from_texts(docs, use_gpu=False)
    cpu_time = time.time() - t0
    gpu_time = -1.0
    if run_gpu and torch.cuda.is_available():
        set_use_gpu(True)
        if sbert:
            try:
                # warmup
                _ = compute_sbert_similarity(docs, device="cuda")
                t1 = time.time()
                _ = compute_sbert_similarity(docs, device="cuda")
                gpu_time = time.time() - t1
            except ImportError:
                # SBERT not installed -> skip gpu sbert
                gpu_time = -1.0
        else:
            # warmup
            _ = compute_tfidf_from_texts(docs, use_gpu=True)
            t1 = time.time()
            _ = compute_tfidf_from_texts(docs, use_gpu=True)
            gpu_time = time.time() - t1
        set_use_gpu(False)
    return cpu_time, gpu_time
# ----------------------------
# Backwards-compatible wrapper functions (for your app)
# ----------------------------
def check_documents(docs: List[str], doc_names: Optional[List[str]] = None,
                    method: str = "tfidf", alpha: float = 0.5, sbert_model: str = DEFAULT_SBERT_MODEL):
    if method == "sbert":
        sim, names = compute_sbert_similarity(docs, model_name=sbert_model,
                                              device=("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"))
    elif method == "hybrid":
        sim, names = compute_hybrid_similarity(docs, alpha=alpha, sbert_model=sbert_model)
    else:
        sim, names = compute_tfidf_from_texts(docs, use_gpu=USE_GPU)
    if doc_names is not None:
        names = doc_names
    pairs = similarity_matrix_to_pairs(sim, doc_names=names)
    return pairs, sim, names
