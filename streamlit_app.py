"""HAIKU ZERO desk sheet — the basic front end that ships with this repository.

No model code lives here. Every poem and every number comes from
`backend.server.generate_poems`, the same function the HTTP API calls, so the desk sheet
and `/api/generate` can never disagree.

    python -m streamlit run streamlit_app.py      (or: start_desk.bat)
"""
from __future__ import annotations

import streamlit as st

from backend import server

st.set_page_config(page_title="HAIKU ZERO — desk sheet", page_icon="🌸", layout="centered")

models = server.registry_payload()
if not models:
    st.error("No `checkpoints*/best.pt` found. Train one with `train.bat` first.")
    st.stop()

ids = [m["id"] for m in models]
labels = {m["id"]: m["label"] for m in models}

with st.sidebar:
    st.header("Desk")
    model_id = st.selectbox("Model", ids, format_func=lambda i: labels[i])
    strict = st.checkbox("Strict 5-7-5", value=True,
                         help="The constrained decoder masks every token that would break the form")
    best_of = st.slider("Best of", 1, 16, 4, help="candidates drawn per poem, best one kept")
    count = st.slider("Poems", 1, 6, 1)
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.05)
    sampling_seed = st.number_input("Sampling seed (0 = random)", min_value=0, value=0, step=1)
    st.caption("Seed word, model and knobs all come from the same payload the API accepts.")

st.title("HAIKU ZERO")
st.caption("A 15.8M-parameter model writes 5-7-5 on this machine. Give it a word.")

seed_word = st.text_input("Seed word", value="harbor", max_chars=40)

if st.button("Write", type="primary"):
    with st.spinner("Writing…"):
        try:
            out = server.generate_poems({
                "model": model_id,
                "prompt": seed_word,
                "count": count,
                "strict": strict,
                "best_of": best_of,
                "temperature": temperature,
                "seed": int(sampling_seed) or None,
            })
        except Exception as exc:                       # a broken run must not blank the page
            st.error(f"{type(exc).__name__}: {exc}")
            st.stop()

    val_loss = out["val_loss"]
    val_text = f"{val_loss:.3f}" if isinstance(val_loss, (int, float)) else str(val_loss)
    st.caption(f"{out['model_label']} · epoch {out['epoch']} · val {val_text} · "
               f"{out['seconds']}s for {out['best_of']} candidates per poem")

    if not out["poems"]:
        st.warning("Nothing survived the rules. More candidates, or turn strict off.")
    for poem in out["poems"]:
        st.markdown(f"**{' · '.join(str(p) for p in poem['pattern'])}** — "
                    f"{'exact 5-7-5' if poem['exact'] else str(poem['distance']) + ' syllables off'}")
        for line in poem["lines"]:
            st.markdown(f"### {line}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("logprob", f"{poem['logprob']:.3f}")
        c2.metric("repeats", f"{poem['repeated_rate']:.2f}")
        c3.metric("unattested", f"{poem['unattested_rate']:.2f}")
        c4.metric("seed used", "yes" if poem["seed_hit"] else "no")
        st.markdown("---")

st.caption("Every number above is measured after the poem is written — syllable pattern, "
           "distance from 5-7-5, mean log-likelihood, repeat rate and how much of the "
           "vocabulary is attested in the corpus.")
