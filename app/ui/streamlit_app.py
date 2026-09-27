"""Streamlit front end. Talks to the API over HTTP so the UI stays a thin
client and the service can be demoed, load-tested or swapped independently."""
from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Multimodal RAG", page_icon="📄", layout="wide")
st.title("Multimodal RAG for Technical Documents")
st.caption("Grounded answers over diagram-heavy papers, with page-level citations.")

with st.sidebar:
    st.subheader("Index")
    try:
        health = requests.get(f"{API}/healthz", timeout=5).json()
        if health.get("status") == "ok":
            st.metric("Text chunks", health.get("text_chunks", 0))
            st.metric("Figures", health.get("figures", 0))
            st.caption(
                f"top_k={health.get('top_k')} · chunk={health.get('chunk_size')} · "
                f"hybrid={health.get('hybrid')} · {health.get('generator')}"
            )
        else:
            st.warning(health.get("detail", "Index unavailable."))
    except Exception:
        st.error(f"API unreachable at {API}")

    k = st.slider("Text passages (k)", 1, 12, 5)
    k_fig = st.slider("Figures", 0, 5, 2)
    if st.button("Rebuild index"):
        requests.post(f"{API}/ingest", timeout=10)
        st.info("Ingest started; watch the counts above.")

question = st.text_input(
    "Question",
    placeholder="How does ByteTrack use low-confidence detections?",
)

if question:
    with st.spinner("Retrieving and answering..."):
        try:
            resp = requests.post(
                f"{API}/query",
                json={"question": question, "k": k, "k_figures": k_fig},
                timeout=120,
            )
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.stop()

    if resp.status_code != 200:
        st.error(resp.json().get("detail", resp.text))
        st.stop()

    data = resp.json()
    st.markdown("### Answer")
    st.write(data["answer"])

    if not data["grounded_shape"]:
        st.warning("This answer cites nothing. Treat it as unverified.")

    if data["citations"]:
        st.markdown("**Sources:** " + " · ".join(data["citations"]))
    st.caption(f"{data['latency_ms']} ms")

    figures = data.get("figures", [])
    if figures:
        st.markdown("### Figures used")
        for fig, col in zip(figures, st.columns(min(len(figures), 3))):
            with col:
                st.image(
                    f"{API}/figures/{fig['figure_id']}",
                    caption=f"[{fig['block']}] {fig['source']} p.{fig['page']}",
                    use_container_width=True,
                )

    with st.expander("Retrieved context"):
        for ctx in data["contexts"]:
            st.markdown(
                f"**[{ctx['block']}]** `{ctx['kind']}` · {ctx['citation']} · "
                f"score {ctx['score']}"
            )
            st.text(ctx["excerpt"])
            st.divider()
