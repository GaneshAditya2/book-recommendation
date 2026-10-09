import os
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Book Recommendation System", page_icon="📚", layout="wide")

ARTIFACT_FILE = "recommender_artifacts.pkl"

ABOUT = (
    "Enter a reader's User ID and the app suggests books they have not read yet. "
    "It learns from real reader ratings in the Book-Crossing dataset, using collaborative "
    "filtering and deep learning to match each reader with books that similar readers enjoyed."
)
STEPS = ["Enter a User ID", "The model scores every unread book", "Get the top picks, ranked"]
METRIC_INFO = [
    ("Precision@10", "Of the 10 books shown, how many the reader liked."),
    ("Recall@10", "Of all the books the reader liked, how many we found."),
    ("HitRate@10", "Share of readers who got at least one good pick."),
]

CSS = """<style>
.block-container{max-width:1120px;padding-top:2.5rem}
.hero-title{font-size:2.9rem;font-weight:800;line-height:1.1;margin:0 0 .8rem}
.hero-text{font-size:1.1rem;line-height:1.6;color:#9AA5B8;margin:0 0 1.4rem}
.step{display:flex;gap:14px;align-items:center;margin:0 0 .8rem;font-size:1.05rem}
.step span{width:32px;height:32px;border-radius:50%;background:#F5B83D;color:#1A1405;font-weight:700;display:flex;align-items:center;justify-content:center;flex:none}
.metric-card{background:#161B26;border:1px solid #262E3F;border-radius:12px;padding:20px}
.metric-name{color:#F5B83D;font-weight:600;font-size:.9rem}
.metric-val{font-size:2rem;font-weight:700;margin:.4rem 0}
.metric-desc{color:#9AA5B8;font-size:.9rem;line-height:1.4}
div[data-testid="stForm"]{background:#161B26;border:1px solid #262E3F;border-radius:12px;padding:1.4rem}
div[data-testid="stFormSubmitButton"] button{width:100%;height:3rem;background:#F5B83D;border:0}
div[data-testid="stFormSubmitButton"] button p{color:#1A1405;font-weight:700}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    with open(ARTIFACT_FILE, "rb") as f:
        return pickle.load(f)


if not os.path.exists(ARTIFACT_FILE):
    st.error(f"'{ARTIFACT_FILE}' not found. Run the notebook up to the last cell and "
             "keep that file in the same folder as app.py.")
    st.stop()

art = load_artifacts()
books = art["books"]
MODEL = art["default_model"]  # best personalised model chosen in the notebook
IS_RATING_MODEL = MODEL in set(art["rating_models"])


def to_table(items, score_label):
    rows = []
    for rank, (isbn, score) in enumerate(items, start=1):
        b = books.get(isbn, {})
        rows.append({
            "Rank": rank,
            "Title": b.get("title", isbn),
            "Author": b.get("author", ""),
            "Year": b.get("year"),
            score_label: score,
        })
    df = pd.DataFrame(rows)
    df["Year"] = df["Year"].astype("Int64")
    return df


# ------------------------------------------------------------------ hero: about + form
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="hero-title">Book Recommendation System</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-text">{ABOUT}</div>', unsafe_allow_html=True)
    st.markdown(
        "".join(f'<div class="step"><span>{i}</span>{t}</div>' for i, t in enumerate(STEPS, 1)),
        unsafe_allow_html=True,
    )

with right:
    with st.form("user_form"):
        user_text = st.text_input("User ID", placeholder="e.g. 11676")
        top_n = st.selectbox("Books to show", [5, 10, 15, 20, 30], index=1)
        submitted = st.form_submit_button("Get Recommendations", type="primary")
    st.caption("Try these User IDs: " + ", ".join(str(u) for u in art["sample_users"]))

# ------------------------------------------------------------------ results
if submitted:
    if not user_text.strip().isdigit():
        st.error("Please enter a numeric User ID.")
    else:
        uid = int(user_text.strip())
        recs = art["recs"][MODEL].get(uid)

        if recs is None:
            st.warning(
                f"User ID {uid} was not found, so we can't personalise yet. "
                "Here are the most popular books instead."
            )
            st.dataframe(to_table(art["popular"][:top_n], "Popularity score"),
                         hide_index=True)
        else:
            st.subheader(f"Top picks for User {uid}")
            label = "Predicted rating" if IS_RATING_MODEL else "Match score"
            st.dataframe(to_table(recs[:top_n], label), hide_index=True)

            liked = art["history"].get(uid, [])
            if liked:
                with st.expander("Books this user rated highest"):
                    st.dataframe(
                        pd.DataFrame({
                            "Title": [books.get(i, {}).get("title", i) for i, _ in liked],
                            "Author": [books.get(i, {}).get("author", "") for i, _ in liked],
                            "Their rating": [r for _, r in liked],
                        }),
                        hide_index=True,
                    )

# ------------------------------------------------------------------ how we measure quality
st.subheader("How we measure quality")
st.caption(f"Each metric is checked on ratings the model never saw during training.  Model in use: {MODEL}")

row = next((m for m in art["metrics"] if m["Model"] == MODEL), {})


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "-"


cols = st.columns(3)
for col, (name, desc) in zip(cols, METRIC_INFO):
    col.markdown(
        f'<div class="metric-card"><div class="metric-name">{name}</div>'
        f'<div class="metric-val">{fmt(row.get(name))}</div>'
        f'<div class="metric-desc">{desc}</div></div>',
        unsafe_allow_html=True,
    )
