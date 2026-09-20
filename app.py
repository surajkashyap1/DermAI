"""DermAI — Streamlit app: upload, inference, Grad-CAM explanation, and chat.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from dermai import __version__
from dermai.classifier import CLASSES, SkinLesionClassifier
from dermai.config import groq_enabled
from dermai.rag import DermatologyRAG

st.set_page_config(page_title="DermAI", page_icon=":microscope:", layout="wide")


# --- Cached heavy objects --------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_classifier() -> SkinLesionClassifier:
    return SkinLesionClassifier()


@st.cache_resource(show_spinner="Building the dermatology knowledge index...")
def get_rag() -> DermatologyRAG:
    rag = DermatologyRAG()
    rag._ensure_store()  # warm the FAISS index
    return rag


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[tuple[str, str]]
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None


# --- Header ----------------------------------------------------------------
st.title(":microscope: DermAI")
st.caption(
    "Dermatology chatbot with HAM10000 skin-cancer classification, Grad-CAM "
    "explainability, and LangChain + FAISS retrieval-augmented Q&A."
)

with st.sidebar:
    st.header("About")
    st.write(
        "DermAI classifies a skin-lesion image into one of seven HAM10000 "
        "classes, explains the prediction with Grad-CAM, and answers "
        "dermatology questions grounded in curated sources."
    )
    st.write(f"**LLM backend:** {'Groq (live)' if groq_enabled() else 'Offline extractive fallback'}")
    st.write(f"**Version:** {__version__}")
    st.divider()
    st.subheader("Lesion classes")
    for info in CLASSES:
        st.markdown(f"- **{info.name}** — {info.malignancy}")
    st.divider()
    st.info(
        "Informational only. DermAI does not provide a medical diagnosis. "
        "Consult a licensed dermatologist for any medical decision.",
        icon=":material/warning:",
    )


tab_classify, tab_chat = st.tabs([":camera: Image classification", ":speech_balloon: Chat"])


# --- Tab 1: image classification + Grad-CAM --------------------------------
with tab_classify:
    st.subheader("Upload a skin-lesion image")
    classifier = get_classifier()

    if not classifier.is_available:
        st.warning(
            "Trained model weights were not found at `models/best_model.h5`. "
            "Add the file or run `python -m dermai.classifier.train` to enable "
            "classification. Chat still works."
        )

    uploaded = st.file_uploader(
        "PNG, JPG, or JPEG", type=["png", "jpg", "jpeg"], key="image_uploader"
    )

    if uploaded is not None and classifier.is_available:
        image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
        with st.spinner("Running inference and Grad-CAM..."):
            prediction = classifier.predict(image, with_gradcam=True)
        st.session_state.last_prediction = prediction

        col_img, col_cam = st.columns(2)
        with col_img:
            st.markdown("**Uploaded image**")
            st.image(image, use_container_width=True)
        with col_cam:
            st.markdown("**Grad-CAM explanation**")
            if prediction.overlay is not None:
                st.image(prediction.overlay, use_container_width=True)
            else:
                st.caption("Grad-CAM overlay unavailable for this image.")

        info = prediction.class_info
        st.success(f"**Diagnosis:** {info.name}  ({info.malignancy})")
        st.metric("Confidence", f"{prediction.confidence * 100:.2f}%")
        st.write(info.description)

        st.markdown("**Class probabilities**")
        st.bar_chart(
            {c.name: prediction.probabilities[c.index] for c in CLASSES}
        )
        st.caption(
            "Disclaimer: this is an automated pattern classification, not a "
            "diagnosis. Please consult a professional dermatologist."
        )


# --- Tab 2: grounded chat --------------------------------------------------
with tab_chat:
    st.subheader("Ask about dermatology")
    rag = get_rag()

    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(bot_msg)

    question = st.chat_input("Ask about skin conditions, lesions, or your result...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        # Fold the latest image result into the question as extra context.
        prediction = st.session_state.last_prediction
        augmented = question
        if prediction is not None:
            augmented = (
                f"{question}\n\n(Context: the user's uploaded image was classified "
                f"as '{prediction.class_info.name}' with "
                f"{prediction.confidence * 100:.1f}% confidence.)"
            )

        with st.chat_message("assistant"):
            with st.spinner("Retrieving grounded answer..."):
                result = rag.ask(augmented, chat_history=st.session_state.chat_history)
            st.write(result.answer)
            if result.sources:
                with st.expander("Sources"):
                    for src in result.sources:
                        page = f" p.{src['page']}" if src.get("page") else ""
                        st.markdown(f"**{src['source']}{page}** — {src['snippet']}...")

        st.session_state.chat_history.append((question, result.answer))
