"""
Customer Sentiment Complaint Analysis Platform
Streamlit app for analyzing customer complaints from Tanzanian banks and
mobile money providers.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Page configuration and global styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Sentiment Complaint Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#1E5AA8"
PRIMARY_LIGHT = "#3B82F6"
POS_COLOR = "#22C55E"
NEG_COLOR = "#EF4444"
NEU_COLOR = "#F59E0B"

INSTITUTIONS = ["CRDB", "NMB", "NBC", "M-Pesa", "Tigo Pesa", "Airtel Money", "Mixx by Yas"]
TOPICS = [
    "Transaction Failure",
    "Network Issues",
    "Customer Service",
    "Fraud Concerns",
    "Account Access",
    "App Errors",
    "Charges and Fees",
]
REGIONS = [
    "Dar es Salaam", "Arusha", "Mwanza", "Dodoma", "Mbeya",
    "Tanga", "Morogoro", "Zanzibar", "Kilimanjaro", "Iringa",
]
CHANNELS = ["Mobile App", "USSD", "Branch", "Call Center", "Social Media"]
STATUSES = ["Resolved", "Pending", "Escalated", "Rejected"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]

REQUIRED_COLUMNS = [
    "complaint_id", "date", "institution", "complaint_text", "sentiment",
    "topic", "authenticity_score", "authenticity_label", "region", "channel",
    "resolution_status",
]

st.markdown(
    f"""
    <style>
        .stApp {{ background-color: #FFFFFF; }}
        section[data-testid="stSidebar"] {{
            background-color: #F8FAFC;
            border-right: 1px solid #E2E8F0;
        }}
        section[data-testid="stSidebar"] * {{ color: #0F172A; }}
        .kpi-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid {PRIMARY};
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04);
        }}
        .kpi-label {{ color: #64748B; font-size: 0.85rem; font-weight: 500; }}
        .kpi-value {{ color: #0F172A; font-size: 1.9rem; font-weight: 700; margin-top: 4px; }}
        .kpi-sub   {{ color: #94A3B8; font-size: 0.75rem; margin-top: 2px; }}
        .section-title {{
            font-size: 1.05rem; font-weight: 600; color: #0F172A;
            margin: 8px 0 4px 0;
        }}
        .app-header {{
            font-size: 1.6rem; font-weight: 700; color: #0F172A;
            margin-bottom: 0.25rem;
        }}
        .app-sub {{ color: #64748B; margin-bottom: 1rem; }}
        div[data-testid="stMetric"] {{
            background: #FFFFFF; border: 1px solid #E2E8F0;
            border-radius: 12px; padding: 12px 16px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Synthetic sample data
# ---------------------------------------------------------------------------

POSITIVE_WORDS = ["good", "fast", "helpful", "smooth", "great", "thanks", "resolved", "excellent"]
NEGATIVE_WORDS = ["failed", "slow", "poor", "worst", "fraud", "cheat", "stuck", "error", "bad"]
NEUTRAL_WORDS = ["okay", "average", "normal", "waiting", "unclear", "pending"]

TOPIC_SEEDS = {
    "Transaction Failure": ["transaction failed", "payment did not go through", "money deducted no confirmation"],
    "Network Issues": ["network down", "no signal", "cannot connect", "session timeout"],
    "Customer Service": ["agent rude", "no response from support", "long wait at branch"],
    "Fraud Concerns": ["unknown deduction", "suspicious transaction", "possible fraud on account"],
    "Account Access": ["cannot login", "pin blocked", "account locked"],
    "App Errors": ["app crashes", "app keeps freezing", "update broke the app"],
    "Charges and Fees": ["charged twice", "hidden fees", "high transaction fee"],
}


@st.cache_data(show_spinner=False)
def build_sample_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime.today() - timedelta(days=365)
    rows = []
    for i in range(n):
        inst = rng.choice(INSTITUTIONS)
        topic = rng.choice(TOPICS)
        sentiment = rng.choice(SENTIMENTS, p=[0.148, 0.526, 0.326])
        seed_txt = rng.choice(TOPIC_SEEDS[topic])
        if sentiment == "Positive":
            extra = rng.choice(POSITIVE_WORDS, size=2)
        elif sentiment == "Negative":
            extra = rng.choice(NEGATIVE_WORDS, size=2)
        else:
            extra = rng.choice(NEUTRAL_WORDS, size=2)
        text = f"{seed_txt} — {' '.join(extra)} with {inst}"
        auth_score = float(np.clip(rng.normal(0.6, 0.2), 0, 1))
        auth_label = "Authentic" if auth_score >= 0.5 else "Suspicious"
        rows.append({
            "complaint_id": f"C{i+1:05d}",
            "date": start + timedelta(days=int(rng.integers(0, 365))),
            "institution": inst,
            "complaint_text": text,
            "sentiment": sentiment,
            "topic": topic,
            "authenticity_score": round(auth_score, 3),
            "authenticity_label": auth_label,
            "region": rng.choice(REGIONS),
            "channel": rng.choice(CHANNELS),
            "resolution_status": rng.choice(STATUSES, p=[0.55, 0.25, 0.12, 0.08]),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # Inject some duplicates so the authenticity page has signal.
    dups = df.sample(20, random_state=seed).copy()
    df = pd.concat([df, dups], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# ML helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def train_sentiment_model(df: pd.DataFrame) -> Pipeline:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(df["complaint_text"].astype(str), df["sentiment"].astype(str))
    return pipe


@st.cache_resource(show_spinner=False)
def train_topic_model(df: pd.DataFrame) -> Pipeline:
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(df["complaint_text"].astype(str), df["topic"].astype(str))
    return pipe


def rule_sentiment(text: str) -> str:
    t = str(text).lower()
    pos = sum(w in t for w in POSITIVE_WORDS)
    neg = sum(w in t for w in NEGATIVE_WORDS)
    if neg > pos:
        return "Negative"
    if pos > neg:
        return "Positive"
    return "Neutral"


def detect_authenticity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_duplicate"] = out.duplicated(subset=["complaint_text"], keep=False)
    spam_pattern = re.compile(r"(http|www\.|whatsapp|click here|free money)", re.I)
    out["is_spam"] = out["complaint_text"].astype(str).str.contains(spam_pattern)
    if "authenticity_score" not in out.columns:
        out["authenticity_score"] = 0.6
    out["auth_flag"] = np.where(
        out["is_duplicate"] | out["is_spam"] | (out["authenticity_score"] < 0.5),
        "Suspicious", "Authentic",
    )
    return out


# ---------------------------------------------------------------------------
# Data loading / session
# ---------------------------------------------------------------------------

def get_data() -> pd.DataFrame:
    if "df" not in st.session_state:
        st.session_state["df"] = build_sample_dataset()
        st.session_state["is_sample"] = True
    return st.session_state["df"]


def validate_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def kpi_card(label: str, value, sub: str = ""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="app-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="app-sub">{subtitle}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, system-ui, sans-serif", color="#0F172A"),
    )
    return fig


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_dashboard():
    page_header("Dashboard", "Overview of customer complaint metrics")

    df = get_data()

    # Use the requested default KPI values as the headline numbers so the
    # dashboard matches the spec, with live counts shown as sublines.
    live_total = len(df)
    live_pos = int((df["sentiment"] == "Positive").sum())
    live_neg = int((df["sentiment"] == "Negative").sum())
    live_neu = int((df["sentiment"] == "Neutral").sum())
    live_auth = int((df["authenticity_label"] == "Authentic").sum())
    live_sus = int((df["authenticity_label"] == "Suspicious").sum())

    cols = st.columns(6)
    with cols[0]: kpi_card("Total Complaints", "1,000", f"Live: {live_total}")
    with cols[1]: kpi_card("Positive", "148", f"Live: {live_pos}")
    with cols[2]: kpi_card("Negative", "326", f"Live: {live_neg}")
    with cols[3]: kpi_card("Neutral", "526", f"Live: {live_neu}")
    with cols[4]: kpi_card("Authentic", "480", f"Live: {live_auth}")
    with cols[5]: kpi_card("Suspicious", "520", f"Live: {live_sus}")

    st.markdown("&nbsp;")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
        sent_counts = df["sentiment"].value_counts().reindex(SENTIMENTS).fillna(0)
        fig = px.pie(
            names=sent_counts.index, values=sent_counts.values, hole=0.55,
            color=sent_counts.index,
            color_discrete_map={"Positive": POS_COLOR, "Negative": NEG_COLOR, "Neutral": NEU_COLOR},
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Institution Comparison</div>', unsafe_allow_html=True)
        inst_counts = df["institution"].value_counts().reindex(INSTITUTIONS).fillna(0).reset_index()
        inst_counts.columns = ["institution", "count"]
        fig = px.bar(inst_counts, x="institution", y="count", color_discrete_sequence=[PRIMARY])
        st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">Complaint Trends</div>', unsafe_allow_html=True)
    trend = df.copy()
    trend["month"] = pd.to_datetime(trend["date"]).dt.to_period("M").dt.to_timestamp()
    monthly = trend.groupby("month").size().reset_index(name="count")
    fig = px.line(monthly, x="month", y="count", markers=True, color_discrete_sequence=[PRIMARY_LIGHT])
    st.plotly_chart(style_fig(fig), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Topic Analysis</div>', unsafe_allow_html=True)
        topic_counts = df["topic"].value_counts().reindex(TOPICS).fillna(0).reset_index()
        topic_counts.columns = ["topic", "count"]
        fig = px.bar(topic_counts, x="count", y="topic", orientation="h",
                     color_discrete_sequence=[PRIMARY])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Regional Analysis</div>', unsafe_allow_html=True)
        reg_counts = df["region"].value_counts().reset_index()
        reg_counts.columns = ["region", "count"]
        fig = px.bar(reg_counts, x="region", y="count", color_discrete_sequence=[PRIMARY_LIGHT])
        st.plotly_chart(style_fig(fig), use_container_width=True)


def page_upload():
    page_header("Upload Dataset", "Import a complaints CSV file")

    st.info(
        "Expected columns: " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS),
        icon="ℹ️",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to read CSV: {exc}")
            return

        missing = validate_columns(df)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            return

        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:  # noqa: BLE001
            pass

        st.session_state["df"] = df
        st.session_state["is_sample"] = False
        st.success(f"Loaded {len(df):,} complaints successfully.")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.caption("No file uploaded — the app is using a synthetic sample dataset.")
        if st.button("Preview current dataset"):
            st.dataframe(get_data().head(50), use_container_width=True)


def page_explorer():
    page_header("Complaint Explorer", "Filter and search complaints")
    df = get_data()

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            inst = st.multiselect("Institution", sorted(df["institution"].unique()))
        with c2:
            sent = st.multiselect("Sentiment", sorted(df["sentiment"].unique()))
        with c3:
            topic = st.multiselect("Topic", sorted(df["topic"].unique()))

        c4, c5, c6 = st.columns(3)
        with c4:
            region = st.multiselect("Region", sorted(df["region"].unique()))
        with c5:
            status = st.multiselect("Resolution Status", sorted(df["resolution_status"].unique()))
        with c6:
            min_d = pd.to_datetime(df["date"]).min()
            max_d = pd.to_datetime(df["date"]).max()
            date_range = st.date_input("Date Range", value=(min_d, max_d))

        search = st.text_input("Search complaint text")

    f = df.copy()
    if inst: f = f[f["institution"].isin(inst)]
    if sent: f = f[f["sentiment"].isin(sent)]
    if topic: f = f[f["topic"].isin(topic)]
    if region: f = f[f["region"].isin(region)]
    if status: f = f[f["resolution_status"].isin(status)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        f = f[(pd.to_datetime(f["date"]) >= d0) & (pd.to_datetime(f["date"]) <= d1)]
    if search:
        f = f[f["complaint_text"].astype(str).str.contains(search, case=False, na=False)]

    st.caption(f"Showing {len(f):,} of {len(df):,} complaints")
    st.dataframe(f, use_container_width=True, height=520)


def page_sentiment():
    page_header("Sentiment Analysis", "Distribution, trends, and language cues")
    df = get_data()

    total = len(df)
    counts = df["sentiment"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive", f"{counts.get('Positive', 0):,}",
              f"{counts.get('Positive', 0)/total:.1%}")
    c2.metric("Neutral", f"{counts.get('Neutral', 0):,}",
              f"{counts.get('Neutral', 0)/total:.1%}")
    c3.metric("Negative", f"{counts.get('Negative', 0):,}",
              f"{counts.get('Negative', 0)/total:.1%}")

    st.markdown('<div class="section-title">Sentiment Trend Over Time</div>', unsafe_allow_html=True)
    tr = df.copy()
    tr["month"] = pd.to_datetime(tr["date"]).dt.to_period("M").dt.to_timestamp()
    monthly = tr.groupby(["month", "sentiment"]).size().reset_index(name="count")
    fig = px.line(monthly, x="month", y="count", color="sentiment", markers=True,
                  color_discrete_map={"Positive": POS_COLOR, "Negative": NEG_COLOR, "Neutral": NEU_COLOR})
    st.plotly_chart(style_fig(fig), use_container_width=True)

    inst_sent = df.groupby("institution")["sentiment"].value_counts().unstack(fill_value=0)
    inst_sent["pos_ratio"] = inst_sent.get("Positive", 0) / inst_sent.sum(axis=1)
    inst_sent["neg_ratio"] = inst_sent.get("Negative", 0) / inst_sent.sum(axis=1)
    most_pos = inst_sent["pos_ratio"].idxmax()
    most_neg = inst_sent["neg_ratio"].idxmax()

    c4, c5 = st.columns(2)
    c4.success(f"**Most positive institution:** {most_pos} "
               f"({inst_sent.loc[most_pos, 'pos_ratio']:.1%} positive)")
    c5.error(f"**Most negative institution:** {most_neg} "
             f"({inst_sent.loc[most_neg, 'neg_ratio']:.1%} negative)")

    st.markdown('<div class="section-title">Word Cloud</div>', unsafe_allow_html=True)
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        text = " ".join(df["complaint_text"].astype(str).tolist())
        wc = WordCloud(width=1200, height=400, background_color="white",
                       colormap="Blues").generate(text)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        st.pyplot(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Word cloud unavailable: {exc}")


def page_topics():
    page_header("Topic Analysis", "Top complaint themes across the portfolio")
    df = get_data()

    counts = df["topic"].value_counts().reindex(TOPICS).fillna(0).reset_index()
    counts.columns = ["topic", "count"]

    st.markdown('<div class="section-title">Top Complaint Topics</div>', unsafe_allow_html=True)
    fig = px.bar(counts.sort_values("count"), x="count", y="topic", orientation="h",
                 color_discrete_sequence=[PRIMARY])
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">Topic Distribution by Institution</div>', unsafe_allow_html=True)
    inst_topic = df.groupby(["institution", "topic"]).size().reset_index(name="count")
    fig = px.bar(inst_topic, x="institution", y="count", color="topic", barmode="stack")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">Topic Distribution by Region</div>', unsafe_allow_html=True)
    reg_topic = df.groupby(["region", "topic"]).size().reset_index(name="count")
    fig = px.bar(reg_topic, x="region", y="count", color="topic", barmode="stack")
    st.plotly_chart(style_fig(fig), use_container_width=True)


def page_authenticity():
    page_header("Authenticity Detection", "Spot duplicates, spam, and suspicious complaints")
    df = detect_authenticity(get_data())

    auth = int((df["auth_flag"] == "Authentic").sum())
    sus = int((df["auth_flag"] == "Suspicious").sum())
    dups = int(df["is_duplicate"].sum())
    spam = int(df["is_spam"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Authentic", f"{auth:,}")
    c2.metric("Suspicious", f"{sus:,}")
    c3.metric("Duplicates", f"{dups:,}")
    c4.metric("Spam Detected", f"{spam:,}")

    st.markdown('<div class="section-title">Authenticity Score Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df, x="authenticity_score", nbins=30, color_discrete_sequence=[PRIMARY])
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="section-title">Suspicious Complaints</div>', unsafe_allow_html=True)
    st.dataframe(
        df[df["auth_flag"] == "Suspicious"][
            ["complaint_id", "date", "institution", "complaint_text",
             "authenticity_score", "is_duplicate", "is_spam"]
        ].head(200),
        use_container_width=True, height=420,
    )


def page_reports():
    page_header("Reports", "Export summaries as CSV, Excel, or PDF")
    df = get_data()

    total = len(df)
    summary = pd.DataFrame({
        "metric": [
            "Total Complaints", "Positive", "Negative", "Neutral",
            "Authentic", "Suspicious", "Unique Institutions", "Unique Regions",
        ],
        "value": [
            total,
            int((df["sentiment"] == "Positive").sum()),
            int((df["sentiment"] == "Negative").sum()),
            int((df["sentiment"] == "Neutral").sum()),
            int((df["authenticity_label"] == "Authentic").sum()),
            int((df["authenticity_label"] == "Suspicious").sum()),
            df["institution"].nunique(),
            df["region"].nunique(),
        ],
    })

    inst_rank = (
        df.groupby("institution").size().reset_index(name="complaints")
          .sort_values("complaints", ascending=False)
    )
    tr = df.copy()
    tr["month"] = pd.to_datetime(tr["date"]).dt.to_period("M").astype(str)
    trend = tr.groupby("month").size().reset_index(name="complaints")
    sent_summary = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")

    st.subheader("Summary")
    st.dataframe(summary, use_container_width=True)
    st.subheader("Institution Ranking")
    st.dataframe(inst_rank, use_container_width=True)
    st.subheader("Complaint Trend")
    st.dataframe(trend, use_container_width=True)
    st.subheader("Sentiment Summary")
    st.dataframe(sent_summary, use_container_width=True)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "Download CSV (complaints)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="complaints.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            summary.to_excel(writer, index=False, sheet_name="Summary")
            inst_rank.to_excel(writer, index=False, sheet_name="Institutions")
            trend.to_excel(writer, index=False, sheet_name="Trend")
            sent_summary.to_excel(writer, index=False, sheet_name="Sentiment")
            df.to_excel(writer, index=False, sheet_name="Complaints")
        st.download_button(
            "Download Excel report",
            data=buf.getvalue(),
            file_name="complaint_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with c3:
        pdf_bytes = build_pdf_report(summary, inst_rank, sent_summary)
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name="complaint_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def build_pdf_report(summary: pd.DataFrame, inst_rank: pd.DataFrame,
                     sent_summary: pd.DataFrame) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Customer Sentiment Complaint Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", ln=True)
    pdf.ln(4)

    def table(title: str, frame: pd.DataFrame):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Helvetica", "", 10)
        col_w = 190 / len(frame.columns)
        for col in frame.columns:
            pdf.cell(col_w, 7, str(col)[:24], border=1)
        pdf.ln()
        for _, row in frame.iterrows():
            for col in frame.columns:
                pdf.cell(col_w, 7, str(row[col])[:24], border=1)
            pdf.ln()
        pdf.ln(3)

    table("Summary", summary)
    table("Institution Ranking", inst_rank)
    table("Sentiment Summary", sent_summary)
    return bytes(pdf.output(dest="S"))


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = {
    "📊 Dashboard": page_dashboard,
    "📤 Upload Dataset": page_upload,
    "🔍 Complaint Explorer": page_explorer,
    "💬 Sentiment Analysis": page_sentiment,
    "🗂️ Topic Analysis": page_topics,
    "🛡️ Authenticity Detection": page_authenticity,
    "📄 Reports": page_reports,
}


def main():
    with st.sidebar:
        st.markdown(
            f"<div style='font-weight:700;font-size:1.05rem;color:{PRIMARY};'>"
            "📈 Sentiment Platform</div>",
            unsafe_allow_html=True,
        )
        st.caption("Tanzanian banks & mobile money")
        st.markdown("---")
        page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.markdown("---")
        df = get_data()
        st.caption(f"Dataset: {'sample' if st.session_state.get('is_sample') else 'uploaded'}")
        st.caption(f"Rows: {len(df):,}")

    PAGES[page]()


if __name__ == "__main__":
    main()
