"""
PlantDoc Streamlit web application.

End-to-end user flow:
  1. User uploads a leaf photo.
  2. CNN predicts disease + confidence.
  3. Grad-CAM shows which image regions drove the prediction.
  4. RAG pipeline fetches treatment advice grounded in our disease KB.
  5. User can ask follow-up questions in a chat interface.

Production-grade features:
  - Rate limiting (see src/utils.RateLimiter)
  - Structured logging to logs/plantdoc.log
  - Error handling so the app never shows a stack trace to users

Run locally with:
    streamlit run app.py
"""

import os
from pathlib import Path
import uuid

import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image

from src.model import load_checkpoint
from src.predict import predict_topk, predict_with_gradcam
from src.rag import PlantDocRAG, advice_for_predicted_disease
from src.utils import setup_logger, RateLimiter, safe_call


load_dotenv()

st.set_page_config(
    page_title="PlantPulse - AI Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/plantdoc_resnet50.pt")
CHROMA_DIR = os.getenv("CHROMA_DIR", "models/chroma_db")
NUM_CLASSES = 38
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = setup_logger("plantdoc.app")

# Rate limiter: each session gets its own client_id (see st.session_state below).
# 20 predictions or RAG calls per minute per session.
rate_limiter = RateLimiter(max_calls=20, window_seconds=60)


# ---- Cached resource loaders -------------------------------------------------
@st.cache_resource
def load_model_cached():
    """Load the trained CNN once per Streamlit process."""
    logger.info(f"Loading model from {MODEL_PATH} onto {DEVICE}")
    if not Path(MODEL_PATH).exists():
        return None, None
    model, class_names = load_checkpoint(MODEL_PATH, num_classes=NUM_CLASSES, device=DEVICE)
    return model, class_names


@st.cache_resource
def load_rag_cached():
    """Load the RAG pipeline once per Streamlit process."""
    logger.info(f"Loading RAG from {CHROMA_DIR}")
    if not Path(CHROMA_DIR).exists():
        return None
    return PlantDocRAG(persist_directory=CHROMA_DIR)


# ---- Safe wrappers around model + RAG calls ----------------------------------
@safe_call(logger, fallback=lambda e: {"error": str(e)})
def run_prediction(model, image, class_names):
    """Predict disease + Grad-CAM overlay. Returns dict (or {'error': ...})."""
    top1, prob, overlay = predict_with_gradcam(model, image, class_names, device=DEVICE)
    topk = predict_topk(model, image, class_names, k=3, device=DEVICE)
    return {"top1": top1, "prob": prob, "overlay": overlay, "topk": topk}


@safe_call(logger, fallback=lambda e: {"error": str(e)})
def run_rag_advice(rag, predicted_class):
    return advice_for_predicted_disease(rag, predicted_class)


@safe_call(logger, fallback=lambda e: {"error": str(e)})
def run_rag_question(rag, question, class_filter=None):
    return rag.answer(question, class_filter=class_filter)


if "client_id" not in st.session_state:
    st.session_state.client_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


st.title("🌿 PlantPulse")
st.caption("AI-powered plant disease detection and treatment advisor")

with st.sidebar:
    st.header("About")
    st.markdown(
        "Upload a photo of a plant leaf and PlantDoc will:\n"
        "1. Identify the disease using a fine-tuned ResNet-50\n"
        "2. Show where the model looked (Grad-CAM)\n"
        "3. Retrieve treatment advice via a RAG pipeline\n"
        "4. Answer follow-up questions"
    )
    st.divider()
    st.caption(f"Running on: `{DEVICE}`")
    if st.button("🔄 Reset session"):
        for k in ["chat_history", "last_prediction"]:
            st.session_state[k] = [] if k == "chat_history" else None
        st.rerun()

model, class_names = load_model_cached()
rag = load_rag_cached()

# Graceful degradation if model/RAG aren't available
if model is None:
    st.error(
        f"Model checkpoint not found at `{MODEL_PATH}`. "
        f"Train the model first via `notebooks/02_training.ipynb`."
    )
    st.stop()

if rag is None:
    st.warning(
        f"RAG vector store not found at `{CHROMA_DIR}`. "
        f"Build it first with: `python -m src.build_knowledge_base`. "
        f"Predictions will still work, but treatment advice won't be available."
    )

# Image upload UI
uploaded_file = st.file_uploader(
    "Upload a leaf photo",
    type=["jpg", "jpeg", "png"],
    help="For best results, photograph a single leaf with clear lighting against a plain background.",
)

if uploaded_file is not None:
    # Rate-limit predictions
    allowed, reset_in = rate_limiter.check(st.session_state.client_id)
    if not allowed:
        st.error(f"Rate limit reached. Try again in {reset_in} seconds.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        logger.exception(f"Failed to decode uploaded image: {e}")
        st.error("Could not read that image. Please try a JPG or PNG file.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your image")
        st.image(image, use_container_width=True)

    with st.spinner("Analyzing leaf..."):
        result = run_prediction(model, image, class_names)

    if "error" in result:
        st.error(f"Prediction failed: {result['error']}")
        st.stop()

    st.session_state.last_prediction = result["top1"]

    with col2:
        st.subheader("Grad-CAM (where the model looked)")
        st.image(result["overlay"], use_container_width=True)

    # Pretty-print the class name
    pretty_name = result["top1"].replace("___", " · ").replace("_", " ")
    st.success(f"**Prediction: {pretty_name}** (confidence: {result['prob']*100:.1f}%)")

    with st.expander("Top 3 predictions"):
        for cls, p in result["topk"]:
            st.write(f"- {cls.replace('___', ' · ').replace('_', ' ')}: {p*100:.1f}%")

    logger.info(
        f"Prediction: client={st.session_state.client_id[:8]} "
        f"class={result['top1']} conf={result['prob']:.3f}"
    )

    # Fetch treatment advice via RAG
    if rag is not None:
        st.divider()
        st.subheader("💊 Treatment Advice")
        with st.spinner("Retrieving treatment information..."):
            advice = run_rag_advice(rag, result["top1"])

        if "error" in advice:
            st.error(f"RAG query failed: {advice['error']}")
        else:
            st.markdown(advice["answer"])
            with st.expander("Sources used"):
                for src in advice["sources"]:
                    st.json(src)


# Follow-up chat interface (only shown after a prediction has been made)
if st.session_state.last_prediction and rag is not None:
    st.divider()
    st.subheader("💬 Ask a follow-up")

    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)

    user_q = st.chat_input("Ask about your plant's diagnosis or treatment...")
    if user_q:
        allowed, reset_in = rate_limiter.check(st.session_state.client_id)
        if not allowed:
            st.error(f"Rate limit reached. Try again in {reset_in} seconds.")
        else:
            st.session_state.chat_history.append(("user", user_q))
            with st.chat_message("user"):
                st.markdown(user_q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = run_rag_question(
                        rag, user_q, class_filter=st.session_state.last_prediction
                    )
                if "error" in answer:
                    st.error(f"Error: {answer['error']}")
                else:
                    st.markdown(answer["answer"])
                    st.session_state.chat_history.append(("assistant", answer["answer"]))
                    logger.info(
                        f"RAG query: client={st.session_state.client_id[:8]} "
                        f"q={user_q[:60]!r}"
                    )
