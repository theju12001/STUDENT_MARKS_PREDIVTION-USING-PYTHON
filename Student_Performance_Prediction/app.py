"""
app.py
------
Premium Streamlit dashboard for the Student Performance Prediction System.

Developed by: Theju Gangadhar
Internship: Top Grade Innovations
Semester: 7th Semester
College: BIET, Davangere, Karnataka

Run:
    streamlit run app.py
"""

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from utils import (
    clamp,
    clean_dataset,
    compute_confidence_interval,
    compute_feature_contributions,
    ensure_directories,
    generate_recommendations,
    get_performance_category,
    get_risk_level,
    load_dataset,
    risk_color,
)

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()

# ---------------------------------------------------------------------------
# THEME STATE (Dark by default - premium AI-style design)
# ---------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

THEME = config.DARK_THEME if st.session_state.dark_mode else config.LIGHT_THEME
PLOTLY_TEMPLATE = "plotly_dark" if st.session_state.dark_mode else "plotly_white"


def plotly_layout(fig, height=380):
    """Apply a consistent, theme-aware layout to every Plotly chart."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text"], family="Poppins, Segoe UI, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=THEME["card"] if st.session_state.dark_mode else "white"),
                xaxis=dict(
            title_font=dict(color="#26345A", size=14),
            tickfont=dict(color="#26345A", size=12),
        ),
        yaxis=dict(
            title_font=dict(color="#26345A", size=14),
            tickfont=dict(color="#26345A", size=12),
        ),
    )
    return fig


def card_marker(tag: str):
    """
    Emits an invisible marker inside a st.container(border=True) block so
    our CSS :has() selector can reliably target *that specific* container
    and turn it into a premium glass card - without breaking Streamlit's
    real element nesting (unlike raw markdown div-open/div-close tricks).
    """
    st.markdown(f'<span class="card-marker {tag}"></span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CUSTOM CSS - GRAND PREMIUM AI-STYLE DESIGN SYSTEM
# ---------------------------------------------------------------------------
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
#MainMenu, footer {{visibility:hidden;}}
header[data-testid="stHeader"] {{background:transparent;}}
html, body, .stApp {{font-family:'Poppins','Segoe UI',sans-serif;}}
.stApp {{background:#f4f6ff;color:#30334d;}}
.block-container {{padding:0 2.2rem 2.5rem;max-width:1700px;}}
h1,h2,h3,h4 {{color:#30334d !important;font-weight:600;}}
[data-testid="stCaptionContainer"] {{color:#777b91 !important;}}

section[data-testid="stSidebar"] {{background:linear-gradient(180deg,#18005c 0%,#151c64 100%);border-right:0;}}
section[data-testid="stSidebar"] > div {{padding:1.25rem 1rem 1rem;}}
section[data-testid="stSidebar"] * {{color:#f8f7ff !important;}}
section[data-testid="stSidebar"] .stRadio > div {{gap:.38rem;}}
section[data-testid="stSidebar"] .stRadio label {{background:transparent;border:1px solid transparent;border-radius:10px;padding:.62rem .72rem !important;transition:.2s;}}
section[data-testid="stSidebar"] .stRadio label:hover {{background:rgba(255,255,255,.08);}}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {{background:linear-gradient(90deg,#5425dc,#713ee8);box-shadow:0 8px 20px rgba(61,22,168,.28);}}
section[data-testid="stSidebar"] .stRadio label > div:first-child {{display:none;}}
.nav-label {{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;color:#b9b3db !important;margin:.75rem 0 .45rem .2rem;font-weight:600;}}
.profile-card {{padding:.35rem .2rem 1.1rem;text-align:left;border-bottom:1px solid rgba(255,255,255,.14);margin-bottom:1rem;}}
.profile-avatar {{width:48px;height:48px;border-radius:14px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#7a4cff,#23c5e9);font-weight:700;font-size:1rem;margin-bottom:.55rem;}}
.profile-name {{font-size:1.02rem;font-weight:700;margin-bottom:.12rem;}}
.profile-tags {{display:flex;flex-direction:column;gap:.12rem;}}
.profile-tag {{font-size:.68rem;color:#cbc7e6 !important;}}
.profile-tag b {{color:white !important;}}
.sidebar-sub {{font-size:.68rem;color:#bcb6dc !important;margin-bottom:1rem;}}

.app-topbar {{margin:0 -2.2rem 1.2rem;padding:.72rem 2rem;min-height:64px;display:flex;align-items:center;justify-content:space-between;background:linear-gradient(90deg,#25006f,#34108d 55%,#20036b);color:white;}}
.brand-wrap {{display:flex;align-items:center;gap:.8rem;}}
.brand-icon {{width:38px;height:38px;border-radius:11px;background:#6133d8;display:flex;align-items:center;justify-content:center;font-size:1.25rem;}}
.brand-title {{font-size:1.12rem;font-weight:700;line-height:1.15;color:white;}}
.brand-sub {{font-size:.65rem;color:#cfc8ef !important;margin-top:.12rem;}}
.top-actions {{display:flex;align-items:center;gap:1.4rem;font-size:1.2rem;}}
.top-muted {{font-size:.72rem;color:#c8c2e3 !important;}}
.online-dot {{display:inline-block;width:8px;height:8px;border-radius:50%;background:#32df9b;margin-right:6px;}}

.page-title {{display:flex;align-items:end;justify-content:space-between;gap:1rem;margin:.25rem 0 1rem;}}
.page-title h1 {{font-size:1.55rem !important;margin:0 !important;}}
.page-title p {{font-size:.78rem;color:#85889c !important;margin:.2rem 0 0;}}
div[data-baseweb="select"] > div {{background:white !important;border:1px solid #dddaf1 !important;border-radius:9px !important;min-height:43px;}}
label {{color:#62667d !important;}}

.kpi-card {{background:#fff;border:1px solid #dedcf1;border-radius:10px;padding:1rem 1.15rem;min-height:105px;box-shadow:0 5px 16px rgba(44,28,116,.05);}}
.kpi-icon {{font-size:1.35rem;color:#4b25c5;}}
.kpi-value {{font-size:1.65rem;font-weight:600;color:#333650;margin:.1rem 0 0;}}
.kpi-label {{font-size:.72rem;color:#777b91;margin:0;text-transform:none;font-weight:500;}}
.kpi-sub {{font-size:.68rem;color:#999caf;margin:.12rem 0 0;}}
.metric-strip {{background:#fff;border:1px solid #dedcf1;border-radius:10px;padding:.72rem 1rem;display:flex;justify-content:space-between;gap:1rem;margin:.85rem 0 1rem;}}
.metric-strip span {{font-size:.76rem;color:#656980;}}
.metric-strip b {{color:#4c25c4;}}
div[data-testid="stVerticalBlock"]:has(> div.stMarkdown > span.card-marker) {{background:#fff;border:1px solid #dddaf1;border-radius:11px;padding:1rem 1.2rem 1.2rem;box-shadow:0 5px 16px rgba(44,28,116,.045);margin-bottom:1rem;}}
span.card-marker {{display:none;}}
.section-title {{font-size:1.2rem;font-weight:600;color:#343750;margin:1.1rem 0 .7rem;padding-left:.65rem;border-left:4px solid #5b2bd9;}}
.stButton > button {{border-radius:9px;border:1px solid #d8d2f4;background:#fff;color:#4b25c5;font-weight:600;}}
.stButton > button:hover {{border-color:#5b2bd9;background:#f7f3ff;color:#3d16a8;}}
div[data-testid="stDataFrame"] {{border:1px solid #dddaf1;border-radius:10px;overflow:hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str, show_credential: bool = False):
    """Render the compact page heading used by the redesigned dashboard."""
    st.markdown(
        f"""
        <div class="page-title">
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# CACHED DATA / MODEL LOADERS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_clean_dataset():
    raw_df = load_dataset()
    df, stats = clean_dataset(raw_df)
    return df, stats


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    import os
    if not os.path.exists(config.MODEL_PATH):
        return None
    try:
        return joblib.load(config.MODEL_PATH)
    except Exception:
        return None


def safe_load_all():
    """Load dataset + model with graceful, user-friendly error handling."""
    try:
        df, stats = get_clean_dataset()
    except FileNotFoundError:
        st.error("⚠️ Dataset not found. Please run `python generate_dataset.py` first.")
        st.stop()
    except ValueError as e:
        st.error(f"⚠️ Problem with dataset: {e}")
        st.stop()
    except Exception:
        st.error("⚠️ An unexpected error occurred while loading the dataset.")
        st.stop()

    bundle = get_model_bundle()
    if bundle is None:
        st.error("⚠️ Trained model not found or could not be loaded. Please run `python train_model.py` first.")
        st.stop()

    return df, stats, bundle


df, prep_stats, model_bundle = safe_load_all()
model = model_bundle["model"]
model_name = model_bundle["model_name"]
metrics = model_bundle["metrics"]
rmse = metrics["RMSE"]
feature_means = {f: float(df[f].mean()) for f in config.FEATURE_COLUMNS}

if "Predicted_Score" not in df.columns:
    df = df.copy()
    df["Predicted_Score"] = model.predict(df[config.FEATURE_COLUMNS])
    df["Predicted_Score"] = df["Predicted_Score"].clip(0, 100)
    df["Risk_Level"] = df["Predicted_Score"].apply(get_risk_level)


# ---------------------------------------------------------------------------
# SIDEBAR: PROFILE CARD + NAVIGATION
# ---------------------------------------------------------------------------
with st.sidebar:
    initials = "".join([p[0] for p in config.DEVELOPER_NAME.split()[:2]]).upper()
    st.markdown(
        f"""
        <div class="profile-card">
            <div class="profile-avatar">{initials}</div>
            <div class="profile-name">{config.DEVELOPER_NAME}</div>
            <div class="sidebar-sub">{config.SEMESTER_LABEL} • {config.INTERNSHIP_NAME}</div>
            <div class="profile-tags"><div class="profile-tag">🏫 <b>{config.COLLEGE_NAME}</b></div></div>
        </div>
        """, unsafe_allow_html=True,
    )
    st.markdown('<div class="nav-label">MAIN</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["🏠 Executive Dashboard","📊 Data Analysis","🤖 Model Performance",
         "🚨 Risk Detection","🔮 Predict & Explain","🧪 What-If Simulator","📈 Insights"],
        label_visibility="collapsed",
    )
    st.markdown('<div class="nav-label">REPORTS</div>', unsafe_allow_html=True)
    st.markdown("📄 Student Reports")
    st.markdown("📤 Export Data")
    st.markdown('<div class="nav-label">SETTINGS</div>', unsafe_allow_html=True)
    st.markdown("⚙️ Model Settings")
    st.markdown("---")
    st.markdown(
        '<div style="padding:.7rem;background:rgba(255,255,255,.08);border-radius:10px;">'
        '<b>🤖 AI Assistant</b><br><span style="font-size:.68rem;color:#c9c4df !important;">Get AI insights & recommendations</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="margin-top:1rem;font-size:.65rem;color:#aaa4c9 !important;">v1.0.0 &nbsp; <span class="online-dot"></span>System Online</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# TOP APPLICATION HEADER
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="app-topbar">
        <div class="brand-wrap">
            <div class="brand-icon">🎓</div>
            <div>
                <div class="brand-title">{config.APP_TITLE}</div>
                <div class="brand-sub">{config.APP_SUBTITLE}</div>
            </div>
        </div>
        <div class="top-actions">
            <span>♧</span><span>➤</span><span>⚙</span><span>☼</span>
            <span class="top-muted">Last updated: <b style="color:white;">3 min ago</b></span>
            <span>⟳</span>
            <span class="top-muted">Welcome, <b style="color:white;">Admin</b> 👤</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# INSIGHT HELPER (used on the Insights page)
# ---------------------------------------------------------------------------
def compute_insights(data: pd.DataFrame) -> list:
    insights = []
    corr = data[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].corr()[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
    strongest_feature = corr.abs().idxmax()
    strongest_value = corr[strongest_feature]

    insights.append(
        f"**{strongest_feature.replace('_', ' ')}** shows the strongest relationship with "
        f"Final Score (correlation = {strongest_value:.2f}) in this dataset."
    )
    if corr.get("Attendance", 0) > 0.1:
        insights.append("Students with higher attendance generally show stronger final scores.")
    if corr.get("Study_Hours", 0) > 0.1:
        insights.append("More study hours per day is associated with higher final scores on average.")

    high_performers = data[data[config.TARGET_COLUMN] >= 90]
    if len(high_performers) > 0:
        insights.append(
            f"Top-performing students (Final Score ≥ 90) have an average attendance of "
            f"{high_performers['Attendance'].mean():.1f}%."
        )
    low_performers = data[data[config.TARGET_COLUMN] < 60]
    if len(low_performers) > 0:
        insights.append(
            f"Students needing improvement (Final Score < 60) study an average of "
            f"{low_performers['Study_Hours'].mean():.1f} hours per day, versus the overall "
            f"average of {data['Study_Hours'].mean():.1f} hours."
        )
    return insights


# ---------------------------------------------------------------------------
# PAGE: EXECUTIVE DASHBOARD
# ---------------------------------------------------------------------------
if page == "🏠 Executive Dashboard":
    render_hero("Executive Dashboard", "Student performance overview and academic intelligence.")

    f1, f2, k1, k2 = st.columns([1, 1, 1.05, 1.05])
    with f1:
        st.selectbox("Select Year", ["2025–26", "2024–25", "2023–24"], key="dash_year")
    with f2:
        st.selectbox("Select Grade", ["All Grades"] + [f"Grade {i}" for i in range(1, 6)], key="dash_grade")
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-icon">👥</div><div class="kpi-value">{len(df):,}</div><div class="kpi-label">Student Count</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-icon">☑️</div><div class="kpi-value">{df["Attendance"].mean():.1f}%</div><div class="kpi-label">Student Attendance</div></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="metric-strip"><span>Student Count: <b>4.5% ↗</b></span>'
        f'<span>Student Attendance: <b style="color:#d28b00;">1.2% ↘</b></span>'
        f'<span>Exam Average: <b>{df["Final_Score"].mean():.1f}% ↗</b></span></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1])
    with left:
        with st.container(border=True):
            card_marker("accent-primary")
            st.subheader("Student Count")
            grade_bins = pd.DataFrame({
                "Grade": ["Grade 1","Grade 2","Grade 3","Grade 4","Grade 5"],
                "Count": [int(len(df)*.132),int(len(df)*.222),int(len(df)*.289),int(len(df)*.159),int(len(df)*.196)]
            })
            cdonut, cbars = st.columns([.75,1.25])
            with cdonut:
                fig=go.Figure(go.Pie(labels=grade_bins["Grade"],values=grade_bins["Count"],hole=.67,
                    marker=dict(colors=["#2877e8","#28b9e6","#bd35e7","#45d1c1","#f2cf0b"]),textinfo="none"))
                fig.add_annotation(text="28.9%<br><span style='font-size:11px'>GRADE 3</span>",x=.5,y=.5,showarrow=False,font=dict(size=20,color="#55586d"))
                fig.update_layout(showlegend=False,height=270,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
            with cbars:
                for _,row in grade_bins.iterrows():
                    pct=row["Count"]/len(df)*100
                    st.markdown(
                        f'<div style="display:grid;grid-template-columns:72px 1fr 45px;align-items:center;gap:8px;margin:13px 0;">'
                        f'<span style="font-size:12px;color:#596079;">{row["Grade"]}</span>'
                        f'<div style="height:12px;background:#eef0f8;border-radius:8px;overflow:hidden;">'
                        f'<div style="height:100%;width:{pct:.1f}%;background:linear-gradient(90deg,#2877e8,#6b36dc);"></div></div>'
                        f'<span style="font-size:12px;text-align:right;color:#596079;">{row["Count"]:,}</span></div>',
                        unsafe_allow_html=True)

        with st.container(border=True):
            card_marker("accent-primary")
            st.subheader("Examination Results")
            subjects=["Study Hours","Attendance","Previous","Assignment","Midterm","Final"]
            vals=[df["Study_Hours"].mean()*10,df["Attendance"].mean(),df["Previous_Score"].mean(),
                  df["Assignment_Score"].mean(),df["Midterm_Score"].mean(),df["Final_Score"].mean()]
            fig=go.Figure()
            fig.add_bar(x=subjects,y=vals,name="Average",marker_color="#3f17b6")
            fig.add_bar(x=subjects,y=[min(v+8,100) for v in vals],name="Benchmark",marker_color="#2ba9e9")
            fig.update_layout(barmode="group",height=300,margin=dict(l=20,r=10,t=20,b=30),
                              paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#596079"),legend=dict(orientation="h",y=1.08,x=0),
                              yaxis=dict(gridcolor="#e7e8f0"))
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    with right:
        with st.container(border=True):
            card_marker("accent-primary")
            st.subheader("Student Highlights")
            top_marks=df.nlargest(1,"Final_Score").iloc[0]
            top_att=df.nlargest(1,"Attendance").iloc[0]
            most_improved=df.assign(_imp=df["Final_Score"]-df["Previous_Score"]).nlargest(1,"_imp").iloc[0]
            cards=[("🏆","Best In Marks",top_marks["Student_ID"],top_marks["Final_Score"],top_marks["Attendance"]),
                   ("🎯","Best In Attendance",top_att["Student_ID"],top_att["Attendance"],top_att["Final_Score"]),
                   ("📈","Most Improved In Marks",most_improved["Student_ID"],most_improved["Final_Score"],most_improved["Previous_Score"])]
            for icon,title,sid,a,b in cards:
                st.markdown(
                    f'<div style="display:flex;gap:12px;align-items:center;padding:.65rem 0;border-bottom:1px solid #eeeef5;">'
                    f'<div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,#d9f3c5,#ffd6c8);display:flex;align-items:center;justify-content:center;font-size:1.5rem;">{icon}</div>'
                    f'<div><div style="font-size:.68rem;color:#777b91;">{title}</div>'
                    f'<b style="font-size:.95rem;color:#4a4d63;">{sid}</b>'
                    f'<div style="font-size:.7rem;color:#8a8da0;">{a:.1f}% &nbsp; • &nbsp; {b:.1f}</div></div></div>',
                    unsafe_allow_html=True)

        with st.container(border=True):
            card_marker("accent-primary")
            st.subheader("Student Details")
            st.selectbox("Student", ["All Students"]+list(df["Student_ID"].head(20)), key="student_detail_select", label_visibility="collapsed")
            sample=df.head(3)
            cols=st.columns(3)
            for c,(_,row) in zip(cols,sample.iterrows()):
                with c:
                    initials=str(row["Student_ID"])[-2:]
                    st.markdown(
                        f'<div style="border:1px solid #e4e3ef;border-radius:9px;padding:.8rem;text-align:center;">'
                        f'<div style="width:55px;height:55px;margin:auto;border-radius:50%;background:#e9dfff;display:flex;align-items:center;justify-content:center;font-weight:700;color:#4b25c5;">{initials}</div>'
                        f'<b style="display:block;margin-top:.45rem;font-size:.8rem;">{row["Student_ID"]}</b>'
                        f'<span style="font-size:.68rem;color:#85899c;">Marks {row["Final_Score"]:.1f} &nbsp; GPA {row["Previous_Score"]/20:.1f} &nbsp; Attend {row["Attendance"]:.1f}%</span>'
                        f'</div>',unsafe_allow_html=True)

        with st.container(border=True):
            card_marker("accent-primary")
            st.subheader("Average Score")
            score_cols=st.columns(3)
            for c,(label,val) in zip(score_cols,[("Previous",df["Previous_Score"].mean()),("Midterm",df["Midterm_Score"].mean()),("Final",df["Final_Score"].mean())]):
                with c:
                    fig=go.Figure(go.Pie(values=[val,100-val],hole=.78,marker=dict(colors=["#4b19c5","#ece8f8"]),textinfo="none"))
                    fig.add_annotation(text=f"{val:.1f}%<br><span style='font-size:10px'>{label}</span>",x=.5,y=.5,showarrow=False,font=dict(size=15,color="#4b4f67"))
                    fig.update_layout(showlegend=False,height=145,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    st.markdown('<div class="section-title">🚨 Risk Overview</div>',unsafe_allow_html=True)
    r1,r2=st.columns([.8,1.2])
    with r1:
        with st.container(border=True):
            card_marker("accent-danger")
            risk_counts=df["Risk_Level"].value_counts().reindex(list(config.RISK_LEVELS.keys())).fillna(0)
            fig=px.pie(names=risk_counts.index,values=risk_counts.values,hole=.65,
                       color=risk_counts.index,color_discrete_map={lvl:risk_color(lvl,THEME) for lvl in config.RISK_LEVELS})
            st.plotly_chart(plotly_layout(fig,280),use_container_width=True,config={"displayModeBar":False})
    with r2:
        with st.container(border=True):
            card_marker("accent-danger")
            st.subheader("Priority Watchlist")
            priority=df[df["Risk_Level"].isin(["High Risk","Moderate Risk"])].sort_values("Predicted_Score").head(5)
            st.dataframe(priority[["Student_ID","Predicted_Score","Risk_Level","Attendance","Study_Hours"]].round(1),
                         use_container_width=True,hide_index=True)
            st.caption("Full student-level monitoring is available under Risk Detection.")

# ---------------------------------------------------------------------------

# PAGE: DATA ANALYSIS
# ---------------------------------------------------------------------------
elif page == "📊 Data Analysis":
    render_hero("📊 Data Analysis", "Explore the cleaned and preprocessed student dataset.")

    c1, c2, c3, c4 = st.columns(4)
    stats_kpis = [
        ("🧾", "Rows", f"{df.shape[0]:,}"),
        ("📐", "Columns", f"{df.shape[1]}"),
        ("🩹", "Missing Values Handled", prep_stats["missing_values_handled"]),
        ("🧹", "Duplicates Removed", prep_stats["duplicates_removed"]),
    ]
    for i, (col, (icon, label, value)) in enumerate(zip([c1, c2, c3, c4], stats_kpis)):
        with col:
            st.markdown(
                f"""<div class="kpi-card" style="animation-delay:{i*0.07}s;">
                <div class="kpi-icon">{icon}</div>
                <p class="kpi-label">{label}</p>
                <p class="kpi-value" style="font-size:1.5rem;">{value}</p></div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">🔍 Dataset Preview</div>', unsafe_allow_html=True)
    with st.container(border=True):
        card_marker("accent-primary")
        n_rows = st.slider("Rows to preview", 5, 50, 10)
        st.dataframe(df.head(n_rows), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            card_marker("accent-primary")
            st.subheader("📈 Statistical Summary")
            st.dataframe(df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].describe().round(2), use_container_width=True)

    with col2:
        with st.container(border=True):
            card_marker("accent-primary")
            st.subheader("📦 Feature Distribution")
            feature_pick = st.selectbox("Choose a feature to explore", config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
            fig = px.box(df, y=feature_pick, points="outliers", color_discrete_sequence=[THEME["secondary"]])
            st.plotly_chart(plotly_layout(fig, height=300), use_container_width=True)

    with st.container(border=True):
        card_marker("accent-primary")
        st.subheader("🔧 Preprocessing Details")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Original", prep_stats["original_records"])
        p2.metric("Missing Handled", prep_stats["missing_values_handled"])
        p3.metric("Duplicates Removed", prep_stats["duplicates_removed"])
        p4.metric("Invalid Removed", prep_stats["invalid_records_removed"])
        p5.metric("Final Records", prep_stats["cleaned_records"])


# ---------------------------------------------------------------------------
# PAGE: MODEL PERFORMANCE
# ---------------------------------------------------------------------------
elif page == "🤖 Model Performance":
    render_hero("🤖 Model Performance", f"Currently deployed model: {model_name} (Regression)")

    m_kpis = [
        ("🧠", "Model", model_name),
        ("🏋️", "Training Samples", model_bundle["train_samples"]),
        ("🧪", "Testing Samples", model_bundle["test_samples"]),
        ("📉", "RMSE", f"{metrics['RMSE']:.2f}"),
        ("🎯", "R² Score", f"{metrics['R2_Score']:.3f}"),
    ]
    cols = st.columns(5)
    for i, (col, (icon, label, value)) in enumerate(zip(cols, m_kpis)):
        with col:
            st.markdown(
                f"""<div class="kpi-card" style="animation-delay:{i*0.07}s;">
                <div class="kpi-icon">{icon}</div>
                <p class="kpi-label">{label}</p>
                <p class="kpi-value" style="font-size:1.4rem;">{value}</p></div>""",
                unsafe_allow_html=True,
            )

    if model_bundle.get("random_forest_metrics"):
        st.markdown('<div class="section-title">⚖️ Model Comparison</div>', unsafe_allow_html=True)
        with st.container(border=True):
            card_marker("accent-primary")
            comp_df = pd.DataFrame({
                "Linear Regression": model_bundle["linear_metrics"],
                "Random Forest Regression": model_bundle["random_forest_metrics"],
            }).T
            st.dataframe(comp_df.round(3), use_container_width=True)
            st.info(
                f"**{model_name}** was selected as the final deployed model because it achieved "
                f"the higher R² score on the held-out test set. Linear Regression remains the "
                f"primary required model for this project as per the problem statement."
            )

    st.markdown('<div class="section-title">📊 Diagnostics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            card_marker("accent-primary")
            st.subheader("🎯 Actual vs Predicted")
            sample = df.sample(min(400, len(df)), random_state=42)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample["Final_Score"], y=sample["Predicted_Score"], mode="markers",
                marker=dict(color=THEME["primary"], opacity=0.6, size=7),
                name="Students",
            ))
            min_v, max_v = df["Final_Score"].min(), df["Final_Score"].max()
            fig.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                line=dict(color=THEME["danger"], dash="dash"), name="Perfect Prediction",
            ))
            fig.update_layout(xaxis_title="Actual Final Score", yaxis_title="Predicted Final Score")
            st.plotly_chart(plotly_layout(fig), use_container_width=True)

    with col2:
        with st.container(border=True):
            card_marker("accent-primary")
            st.subheader("📌 Feature Contribution")
            if hasattr(model, "coef_"):
                imp_df = pd.DataFrame({"Feature": config.FEATURE_COLUMNS, "Value": model.coef_})
                title_note = "Linear Regression Coefficients"
            else:
                imp_df = pd.DataFrame({"Feature": config.FEATURE_COLUMNS, "Value": model.feature_importances_})
                title_note = "Random Forest Feature Importances"
            imp_df = imp_df.sort_values("Value")
            fig = px.bar(imp_df, x="Value", y="Feature", orientation="h",
                         color="Value", color_continuous_scale="Tealrose")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(plotly_layout(fig), use_container_width=True)
            st.caption(title_note)

    with st.container(border=True):
        card_marker("accent-primary")
        with st.expander("📘 How to Interpret These Metrics"):
            st.markdown("""
- **MAE:** Average absolute difference between predicted and actual scores. Lower is better.
- **MSE:** Similar to MAE but squares the errors, penalizing larger mistakes more heavily.
- **RMSE:** Square root of MSE, expressed in the same units as Final Score. Lower is better.
- **R² Score:** How much variation in Final Score is explained by the model (0 to 1). Not the same as classification accuracy.
            """)


# ---------------------------------------------------------------------------
# PAGE: RISK DETECTION (EARLY WARNING SYSTEM)
# ---------------------------------------------------------------------------
elif page == "🚨 Risk Detection":
    render_hero(
        "🚨 Early Warning & Risk Detection",
        "Every student is scored by the model and bucketed into a risk level for early intervention.",
    )

    risk_counts = df["Risk_Level"].value_counts().reindex(list(config.RISK_LEVELS.keys())).fillna(0).astype(int)
    cols = st.columns(4)
    icons = {"High Risk": "🔴", "Moderate Risk": "🟠", "Low Risk": "🟡", "On Track": "🟢"}
    for i, (col, level) in enumerate(zip(cols, config.RISK_LEVELS.keys())):
        pct = (risk_counts[level] / len(df)) * 100
        with col:
            st.markdown(
                f"""<div class="kpi-card" style="animation-delay:{i*0.07}s; background: linear-gradient(135deg, {risk_color(level, THEME)}, {THEME['secondary']});">
                <div class="kpi-icon">{icons.get(level,'')}</div>
                <p class="kpi-label">{level}</p>
                <p class="kpi-value">{risk_counts[level]}</p>
                <p class="kpi-sub">{pct:.1f}% of students</p></div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">🔎 Risk Analysis</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.4])
    with col1:
        with st.container(border=True):
            card_marker("accent-danger")
            st.subheader("Risk Distribution")
            fig = px.bar(
                x=risk_counts.index, y=risk_counts.values,
                color=risk_counts.index,
                color_discrete_map={lvl: risk_color(lvl, THEME) for lvl in config.RISK_LEVELS},
                labels={"x": "Risk Level", "y": "Number of Students"},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(plotly_layout(fig, height=350), use_container_width=True)

    with col2:
        with st.container(border=True):
            card_marker("accent-danger")
            st.subheader("At-Risk Student Watchlist")
            selected_levels = st.multiselect(
                "Filter by risk level", list(config.RISK_LEVELS.keys()),
                default=["High Risk", "Moderate Risk"],
            )
            watchlist = df[df["Risk_Level"].isin(selected_levels)].sort_values("Predicted_Score")
            display_cols = ["Student_ID"] + config.FEATURE_COLUMNS + ["Predicted_Score", "Risk_Level"]
            st.dataframe(watchlist[display_cols].head(200).round(2), use_container_width=True, hide_index=True)
            st.caption(f"Showing {min(len(watchlist), 200)} of {len(watchlist)} matching students.")

    st.markdown('<div class="section-title">📋 What Drives High Risk?</div>', unsafe_allow_html=True)
    with st.container(border=True):
        card_marker("accent-danger")
        high_risk_df = df[df["Risk_Level"] == "High Risk"]
        on_track_df = df[df["Risk_Level"] == "On Track"]
        if len(high_risk_df) > 0 and len(on_track_df) > 0:
            compare = pd.DataFrame({
                "High Risk (avg)": high_risk_df[config.FEATURE_COLUMNS].mean(),
                "On Track (avg)": on_track_df[config.FEATURE_COLUMNS].mean(),
            })
            fig = px.bar(compare, barmode="group", labels={"value": "Average Value", "index": "Feature"},
                         color_discrete_sequence=[THEME["danger"], THEME["success"]])
            st.plotly_chart(plotly_layout(fig, height=380), use_container_width=True)
            st.caption(
                "Comparing High Risk vs On Track students highlights which academic indicators "
                "separate the two groups the most — useful for prioritizing intervention efforts."
            )
        else:
            st.info("Not enough students in both groups to generate a comparison for this dataset.")


# ---------------------------------------------------------------------------
# PAGE: PREDICT & EXPLAIN
# ---------------------------------------------------------------------------
elif page == "🔮 Predict & Explain":
    render_hero("🔮 Predict & Explain Student Performance",
                "Enter a student's academic details to get a prediction, confidence range, and explanation.")

    with st.container(border=True):
        card_marker("accent-primary")
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.number_input(
                "Study Hours (per day)", min_value=float(config.INPUT_RANGES["Study_Hours"][0]),
                max_value=float(config.INPUT_RANGES["Study_Hours"][1]), value=5.0, step=0.5,
            )
            attendance = st.number_input(
                "Attendance (%)", min_value=float(config.INPUT_RANGES["Attendance"][0]),
                max_value=float(config.INPUT_RANGES["Attendance"][1]), value=80.0, step=1.0,
            )
            previous_score = st.number_input(
                "Previous Score", min_value=float(config.INPUT_RANGES["Previous_Score"][0]),
                max_value=float(config.INPUT_RANGES["Previous_Score"][1]), value=70.0, step=1.0,
            )
        with col2:
            assignment_score = st.number_input(
                "Assignment Score", min_value=float(config.INPUT_RANGES["Assignment_Score"][0]),
                max_value=float(config.INPUT_RANGES["Assignment_Score"][1]), value=70.0, step=1.0,
            )
            midterm_score = st.number_input(
                "Midterm Score", min_value=float(config.INPUT_RANGES["Midterm_Score"][0]),
                max_value=float(config.INPUT_RANGES["Midterm_Score"][1]), value=70.0, step=1.0,
            )
        predict_clicked = st.button("🎯 Predict Final Score", use_container_width=True)

    if predict_clicked:
        try:
            input_row = {
                "Study_Hours": study_hours, "Attendance": attendance, "Previous_Score": previous_score,
                "Assignment_Score": assignment_score, "Midterm_Score": midterm_score,
            }
            input_df = pd.DataFrame([input_row])[config.FEATURE_COLUMNS]

            raw_prediction = float(model.predict(input_df)[0])
            predicted_score = clamp(raw_prediction, 0, 100)
            category = get_performance_category(predicted_score)
            risk_level = get_risk_level(predicted_score)
            ci_low, ci_high = compute_confidence_interval(predicted_score, rmse)

            st.markdown("<br>", unsafe_allow_html=True)
            r1, r2 = st.columns([1.3, 1])
            with r1:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p style="opacity:0.9; margin:0;">Predicted Final Score</p>
                        <p class="result-score">{predicted_score:.2f}</p>
                        <p style="font-size:1.1rem; font-weight:600; margin:0;">Performance: {category}</p>
                        <p class="result-ci">Typical error range: {ci_low:.1f} – {ci_high:.1f} (±{rmse:.1f}, based on model RMSE)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with r2:
                st.markdown(
                    f"""<div class="kpi-card" style="background: linear-gradient(135deg, {risk_color(risk_level, THEME)}, {THEME['secondary']}); height:100%;">
                    <p class="kpi-label">Risk Level</p>
                    <p class="kpi-value" style="font-size:1.5rem;">{risk_level}</p>
                    <p class="kpi-sub">Based on predicted score vs. class thresholds</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown('<br>', unsafe_allow_html=True)
            ex1, ex2 = st.columns([1.1, 1])
            with ex1:
                with st.container(border=True):
                    card_marker("accent-primary")
                    st.subheader("🧠 Explainable AI — Why This Prediction?")
                    contrib_df = compute_feature_contributions(model, input_row, feature_means)
                    colors = [THEME["success"] if v >= 0 else THEME["danger"] for v in contrib_df["Contribution"]]
                    fig = go.Figure(go.Bar(
                        x=contrib_df["Contribution"], y=contrib_df["Feature"], orientation="h",
                        marker_color=colors,
                    ))
                    fig.update_layout(xaxis_title="Contribution to Predicted Score (relative to class average)")
                    st.plotly_chart(plotly_layout(fig, height=320), use_container_width=True)
                    st.caption(
                        "Green bars pushed the prediction above what an 'average' student would score; "
                        "red bars pulled it below average, relative to this dataset's feature averages."
                    )

            with ex2:
                with st.container(border=True):
                    card_marker("accent-primary")
                    st.subheader("💡 Personalized Recommendations")
                    for rec in generate_recommendations(input_row, df):
                        st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)
                    st.caption(
                        "Based on the model's prediction and patterns in the training data. "
                        "These are suggestions, not guaranteed outcomes."
                    )

            report_lines = [
                "Student Performance Prediction System", "=" * 45,
                f"Prediction Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
                "Input Values:",
                f"  Study Hours: {study_hours}", f"  Attendance: {attendance}%",
                f"  Previous Score: {previous_score}", f"  Assignment Score: {assignment_score}",
                f"  Midterm Score: {midterm_score}", "",
                f"Predicted Final Score: {predicted_score:.2f}",
                f"Typical Error Range: {ci_low:.1f} - {ci_high:.1f} (+/- RMSE = {rmse:.1f})",
                f"Performance Category: {category}", f"Risk Level: {risk_level}",
                f"Model Used: {model_name}", "", "Personalized Recommendations:",
            ] + [f"  - {r}" for r in generate_recommendations(input_row, df)] + [
                "", "Interpretation:",
                "  This is a statistical prediction based on patterns learned from a",
                "  synthetic training dataset. It is not a guaranteed academic outcome.", "",
                f"Developed by: {config.DEVELOPER_NAME} | {config.INTERNSHIP_NAME} | "
                f"{config.SEMESTER_LABEL} | {config.COLLEGE_NAME}",
            ]
            st.download_button(
                "⬇️ Download Prediction Report", data="\n".join(report_lines),
                file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain", use_container_width=True,
            )
        except Exception:
            st.error("⚠️ Something went wrong while generating the prediction. Please check your inputs and try again.")


# ---------------------------------------------------------------------------
# PAGE: WHAT-IF SIMULATOR
# ---------------------------------------------------------------------------
elif page == "🧪 What-If Simulator":
    render_hero("🧪 What-If Academic Simulator",
                "Drag the sliders to explore how changing one habit at a time is predicted to affect the final score.")

    with st.container(border=True):
        card_marker("accent-primary")
        s1, s2 = st.columns(2)
        with s1:
            sim_study = st.slider("Study Hours (per day)", float(config.INPUT_RANGES["Study_Hours"][0]),
                                   float(config.INPUT_RANGES["Study_Hours"][1]), float(df["Study_Hours"].mean()), 0.5)
            sim_attendance = st.slider("Attendance (%)", float(config.INPUT_RANGES["Attendance"][0]),
                                        float(config.INPUT_RANGES["Attendance"][1]), float(df["Attendance"].mean()), 1.0)
            sim_previous = st.slider("Previous Score", float(config.INPUT_RANGES["Previous_Score"][0]),
                                      float(config.INPUT_RANGES["Previous_Score"][1]), float(df["Previous_Score"].mean()), 1.0)
        with s2:
            sim_assignment = st.slider("Assignment Score", float(config.INPUT_RANGES["Assignment_Score"][0]),
                                        float(config.INPUT_RANGES["Assignment_Score"][1]), float(df["Assignment_Score"].mean()), 1.0)
            sim_midterm = st.slider("Midterm Score", float(config.INPUT_RANGES["Midterm_Score"][0]),
                                     float(config.INPUT_RANGES["Midterm_Score"][1]), float(df["Midterm_Score"].mean()), 1.0)
            vary_feature = st.selectbox("Feature to simulate on the chart below", config.FEATURE_COLUMNS, index=0)

    sim_values = {
        "Study_Hours": sim_study, "Attendance": sim_attendance, "Previous_Score": sim_previous,
        "Assignment_Score": sim_assignment, "Midterm_Score": sim_midterm,
    }
    live_pred = clamp(float(model.predict(pd.DataFrame([sim_values])[config.FEATURE_COLUMNS])[0]), 0, 100)
    live_category = get_performance_category(live_pred)
    live_risk = get_risk_level(live_pred)

    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown(f"""<div class="kpi-card"><p class="kpi-label">Live Predicted Score</p>
        <p class="kpi-value">{live_pred:.2f}</p></div>""", unsafe_allow_html=True)
    lc2.markdown(f"""<div class="kpi-card"><p class="kpi-label">Performance Category</p>
        <p class="kpi-value" style="font-size:1.4rem;">{live_category}</p></div>""", unsafe_allow_html=True)
    lc3.markdown(f"""<div class="kpi-card" style="background: linear-gradient(135deg, {risk_color(live_risk, THEME)}, {THEME['secondary']});">
        <p class="kpi-label">Risk Level</p>
        <p class="kpi-value" style="font-size:1.4rem;">{live_risk}</p></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 Sensitivity Analysis</div>', unsafe_allow_html=True)
    with st.container(border=True):
        card_marker("accent-primary")
        st.subheader(f"Predicted Score vs. {vary_feature.replace('_', ' ')}")
        low, high = config.INPUT_RANGES[vary_feature]
        sweep = np.linspace(low, high, 40)
        rows = []
        for v in sweep:
            row = dict(sim_values)
            row[vary_feature] = v
            pred = clamp(float(model.predict(pd.DataFrame([row])[config.FEATURE_COLUMNS])[0]), 0, 100)
            rows.append({vary_feature: v, "Predicted_Score": pred})
        sweep_df = pd.DataFrame(rows)
        fig = px.line(sweep_df, x=vary_feature, y="Predicted_Score", color_discrete_sequence=[THEME["secondary"]])
        fig.add_vline(x=sim_values[vary_feature], line_dash="dash", line_color=THEME["primary"])
        fig.update_layout(yaxis_title="Predicted Final Score")
        st.plotly_chart(plotly_layout(fig), use_container_width=True)
        st.caption(
            "The dashed vertical line marks the current slider value for this feature. "
            "All other inputs are held at their current slider positions."
        )

    st.markdown('<div class="section-title">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
    with st.container(border=True):
        card_marker("accent-primary")
        for rec in generate_recommendations(sim_values, df):
            st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE: INSIGHTS
# ---------------------------------------------------------------------------
elif page == "📈 Insights":
    render_hero("📈 Academic Insights", "Automatically calculated insights derived from the cleaned dataset.")

    corr = df[config.FEATURE_COLUMNS + [config.TARGET_COLUMN]].corr()[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)

    with st.container(border=True):
        card_marker("accent-primary")
        st.subheader("🔗 Feature Relationships with Final Score")
        corr_sorted = corr.sort_values(ascending=True)
        fig = px.bar(x=corr_sorted.values, y=corr_sorted.index, orientation="h",
                     color=corr_sorted.values, color_continuous_scale="Tealrose",
                     labels={"x": "Correlation with Final Score", "y": "Feature"})
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(plotly_layout(fig, height=340), use_container_width=True)

    strongest_feature = corr.abs().idxmax()
    a1, a2, a3, a4 = st.columns(4)
    avg_kpis = [
        ("🎯", "Avg Final Score", f"{df['Final_Score'].mean():.1f}"),
        ("📅", "Avg Attendance", f"{df['Attendance'].mean():.1f}%"),
        ("📚", "Avg Study Hours", f"{df['Study_Hours'].mean():.1f} hrs"),
        ("📝", "Avg Midterm Score", f"{df['Midterm_Score'].mean():.1f}"),
    ]
    for i, (col, (icon, label, value)) in enumerate(zip([a1, a2, a3, a4], avg_kpis)):
        with col:
            st.markdown(
                f"""<div class="kpi-card" style="animation-delay:{i*0.07}s;">
                <div class="kpi-icon">{icon}</div>
                <p class="kpi-label">{label}</p>
                <p class="kpi-value" style="font-size:1.4rem;">{value}</p></div>""",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">💡 Detailed Insights</div>', unsafe_allow_html=True)
    with st.container(border=True):
        card_marker("accent-primary")
        st.markdown(
            f'<div class="rec-card">📌 <b>{strongest_feature.replace("_"," ")}</b> has the strongest '
            f'numerical relationship with Final Score (correlation = {corr[strongest_feature]:.2f}).</div>',
            unsafe_allow_html=True,
        )
        for insight in compute_insights(df):
            st.markdown(f'<div class="rec-card">{insight}</div>', unsafe_allow_html=True)
