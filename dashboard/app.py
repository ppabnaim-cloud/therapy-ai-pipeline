"""
Ward 72 & Discharge Lounge — Clinical Analytics Dashboard
Author : Deputy Hospital Director – AI & Data Science Unit
Stack  : Streamlit · Pandas · NumPy · Matplotlib · Seaborn · Plotly · scikit-learn
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score, roc_curve
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import re

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ward 72 & Discharge Lounge Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F4E79 0%, #375623 100%);
        padding: 1.2rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.85; }

    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #1F4E79;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .kpi-card.green { border-left-color: #375623; }
    .kpi-card.orange { border-left-color: #C55A11; }
    .kpi-card.red { border-left-color: #C00000; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1F4E79; }
    .kpi-label { font-size: 0.78rem; color: #666; margin-top: 0.2rem; }
    .kpi-delta { font-size: 0.82rem; margin-top: 0.3rem; }

    .section-header {
        background: #1F4E79;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        margin: 1rem 0 0.8rem;
    }
    .section-header.green { background: #375623; }

    .insight-box {
        background: #EBF3FB;
        border-left: 4px solid #1F4E79;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        font-size: 0.87rem;
        margin-top: 0.5rem;
    }
    .insight-box.green { background: #EDF7ED; border-left-color: #375623; }
    .insight-box.warning { background: #FFF4E5; border-left-color: #C55A11; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR  (mirrors the Excel dummy exactly)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_ward72_data():
    np.random.seed(42)
    n = 12
    names = [
        "Ahmad Faris bin Abdullah","Nurul Ain binti Mohd Yusof","Rajesh Kumar a/l Subramaniam",
        "Lim Siew Ling","Siti Norzahara binti Kamaruddin","Tan Wei Ming",
        "Mohd Hafiz bin Zainudin","Priya a/p Krishnan","Zulaikha binti Osman",
        "Lee Chong Keat","Faridah binti Hamid","Shankar a/l Raman"
    ]
    primary_dx = [
        "Dengue fever with warning signs","Unstable angina","Cellulitis of lower limb",
        "Acute pyelonephritis","Community-acquired pneumonia","Acute gastroenteritis with dehydration",
        "Hyperosmolar hyperglycaemic state","Deep vein thrombosis",
        "Urinary tract infection with systemic features","Exacerbation of COPD",
        "Acute cholecystitis","Anaemia requiring transfusion"
    ]
    primary_codes = [
        "A97.1","I20.0","L03.1","N10","J18.9","A09",
        "E11.01","I82.4","N10","J44.1","K81.0","D64.9"
    ]
    secondary_dx = [
        "Type 2 diabetes mellitus","Essential hypertension","Bronchial asthma",
        "Chronic kidney disease stage 3","Hyperlipidaemia","Ischaemic heart disease",
        "Nil","Nil","Obesity","Hypothyroidism","Gout","Chronic liver disease"
    ]
    secondary_codes = [
        "E11.9","I10","J45.9","N18.3","E78.5","I25.1",
        "","","E66.9","E03.9","M10.9","K74.6"
    ]
    teams  = ["General Medicine","Cardiology","Infectious Disease","Respiratory","Nephrology","General Surgery"]
    genders = ["Male","Female"]
    dispositions = ["Discharged Home","Step-down to Discharge Lounge","Transfer to Ward"]

    triage_times    = pd.date_range("2025-02-24 07:00", periods=n, freq="45min") + pd.to_timedelta(np.random.randint(0,30,n), unit='m')
    refer_delta     = pd.to_timedelta(np.random.randint(15,90,n),  unit='m')
    encounter_delta = pd.to_timedelta(np.random.randint(15,120,n), unit='m')
    decision_delta  = pd.to_timedelta(np.random.randint(15,90,n),  unit='m')
    push_delta      = pd.to_timedelta(np.random.randint(15,180,n), unit='m')

    refer_times    = triage_times    + refer_delta
    encounter_times= refer_times     + encounter_delta
    decision_times = encounter_times + decision_delta
    push_times     = decision_times  + push_delta

    bed_wait   = (push_times - decision_times).total_seconds() / 60
    total_wait = (push_times - triage_times).total_seconds()   / 60
    los        = np.random.uniform(6, 72, n).round(1)

    df = pd.DataFrame({
        "No"                          : range(1, n+1),
        "Transfer_From_ED"            : ["Yes"] * n,
        "Patient_Name"                : names,
        "Gender"                      : np.random.choice(genders, n),
        "Age"                         : np.random.randint(18, 81, n),
        "Primary_Team"                : np.random.choice(teams, n),
        "Primary_ICD11_Text"          : primary_dx,
        "Primary_ICD11_Code"          : primary_codes,
        "Secondary_ICD11_Text"        : secondary_dx,
        "Secondary_ICD11_Code"        : secondary_codes,
        "Triage_DateTime"             : triage_times,
        "Referral_DateTime"           : refer_times,
        "First_Encounter_DateTime"    : encounter_times,
        "Decision_Admit_DateTime"     : decision_times,
        "Push_to_Ward72_DateTime"     : push_times,
        "Bed_Waiting_Time_min"        : bed_wait.round(1),
        "Total_ED_Waiting_Time_min"   : total_wait.round(1),
        "Discharged_Within_72hrs"     : ["Yes" if x <= 72 else "No" for x in los],
        "Actual_LOS_hrs"              : los,
        "Disposition"                 : np.random.choice(dispositions, n),
        "Readmission_72hrs"           : np.random.choice(["No","No","No","Yes"], n),
        "Adverse_Event"               : np.random.choice(["No","No","No","No","Yes"], n),
    })
    return df


@st.cache_data
def generate_discharge_lounge_data():
    np.random.seed(99)
    n = 8
    names = [
        "Wong Mei Lin","Abdul Razak bin Ismail","Noor Azura binti Saad",
        "Chong Boon Hoe","Kavitha a/p Pillai","Mohd Ridhwan bin Hamzah",
        "Yeoh Soo Yin","Amirul Syafiq bin Johari"
    ]
    primary_dx = [
        "Dengue fever with warning signs","Cellulitis resolving",
        "Community-acquired pneumonia resolving","Post-procedural observation",
        "Acute gastroenteritis resolved","Urinary tract infection treated",
        "Exacerbation of COPD improving","Anaemia post-transfusion"
    ]
    primary_codes = ["A97.1","L03.1","J18.9","Z48.8","A09","N10","J44.1","D64.9"]
    secondary_dx  = [
        "Type 2 diabetes mellitus","Essential hypertension","Nil",
        "Chronic kidney disease stage 3","Nil","Hyperlipidaemia",
        "Bronchial asthma","Ischaemic heart disease"
    ]
    secondary_codes = ["E11.9","I10","","N18.3","","E78.5","J45.9","I25.1"]
    wards   = ["Ward 72","Ward 5","Ward 6","Ward 7","Ward 8"]
    genders = ["Male","Female"]

    decision_times = pd.date_range("2025-02-24 07:00", periods=n, freq="30min") + pd.to_timedelta(np.random.randint(0,20,n), unit='m')
    push_delta     = pd.to_timedelta(np.random.randint(10,90,n),  unit='m')
    pickup_delta   = pd.to_timedelta(np.random.randint(30,360,n), unit='m')

    push_times   = decision_times + push_delta
    pickup_times = push_times     + pickup_delta

    dwell_min    = (pickup_times - push_times).total_seconds()    / 60
    total_dw_min = (pickup_times - decision_times).total_seconds()/ 60
    family_resp  = dwell_min.copy()
    same_day     = [
        "Yes" if p.date() == d.date() else "No"
        for p, d in zip(pickup_times, decision_times)
    ]

    df = pd.DataFrame({
        "No"                           : range(1, n+1),
        "Transfer_From_Ward"           : ["Yes"] * n,
        "Source_Ward"                  : np.random.choice(wards, n),
        "Patient_Name"                 : names,
        "Gender"                       : np.random.choice(genders, n),
        "Age"                          : np.random.randint(18, 81, n),
        "Primary_ICD11_Text"           : primary_dx,
        "Primary_ICD11_Code"           : primary_codes,
        "Secondary_ICD11_Text"         : secondary_dx,
        "Secondary_ICD11_Code"         : secondary_codes,
        "Discharge_Decision_DateTime"  : decision_times,
        "Push_to_Lounge_DateTime"      : push_times,
        "Family_Pickup_DateTime"       : pickup_times,
        "Dwell_Time_min"               : dwell_min.round(1),
        "Total_Discharge_Wait_min"     : total_dw_min.round(1),
        "Same_Day_Discharge"           : same_day,
        "Family_Response_Time_min"     : family_resp.round(1),
        "Bed_No"                       : range(1, n+1),
        "Satisfaction_Score"           : np.random.randint(3, 6, n),
        "Adverse_Event_Lounge"         : np.random.choice(["No","No","No","No","Yes"], n),
        "Unplanned_Return_to_Ward"     : np.random.choice(["No","No","No","Yes"], n),
        "Readmission_72hrs"            : np.random.choice(["No","No","No","Yes"], n),
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (Google Sheets CSV or fallback to synthetic)
# ─────────────────────────────────────────────────────────────────────────────
SHEET_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ07ibaphXVdG8KvCFdxRhszx802STcdX2dJnqgPeX0RCdcOKCtQuKikOmc9NxyMA/pub?output=csv"

@st.cache_data(ttl=300)
def load_google_sheet(url):
    try:
        df = pd.read_csv(url)
        return df, True
    except Exception:
        return None, False

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label, value, delta=None, colour="blue"):
    cls = {"blue":"", "green":"green", "orange":"orange", "red":"red"}.get(colour,"")
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card {cls}">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def section(title, green=False):
    cls = "green" if green else ""
    st.markdown(f'<div class="section-header {cls}">{title}</div>', unsafe_allow_html=True)

def insight(text, variant="blue"):
    cls = {"blue":"", "green":"green", "orange":"warning"}.get(variant,"")
    st.markdown(f'<div class="insight-box {cls}">💡 {text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Caduceus.svg/100px-Caduceus.svg.png", width=50)
    st.markdown("### ⚙️ Dashboard Settings")

    data_source = st.radio("Data Source", ["Google Sheets (Live)", "Built-in Dummy Data"], index=1)

    st.markdown("---")
    st.markdown("### 📂 Upload CSV")
    uploaded_w72 = st.file_uploader("Ward 72 CSV", type=["csv"])
    uploaded_dl  = st.file_uploader("Discharge Lounge CSV", type=["csv"])

    st.markdown("---")
    page = st.radio("📊 Navigate", [
        "🏠 Overview & KPIs",
        "🛏️ Ward 72 – Analytics",
        "🚪 Discharge Lounge – Analytics",
        "🤖 Machine Learning",
        "🔬 ICD-11 NLP Analysis",
        "📋 Raw Data"
    ])

    st.markdown("---")
    st.caption("Ward 72 & Discharge Lounge\nClinical Analytics Platform\nv1.0 | AI & Data Science Unit")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_w72:
    df_w72 = pd.read_csv(uploaded_w72)
    df_dl  = pd.read_csv(uploaded_dl) if uploaded_dl else generate_discharge_lounge_data()
elif data_source == "Google Sheets (Live)":
    raw, ok = load_google_sheet(SHEET_CSV)
    if ok and raw is not None:
        df_w72 = raw
        df_dl  = generate_discharge_lounge_data()
        st.sidebar.success("✅ Live data loaded")
    else:
        df_w72 = generate_ward72_data()
        df_dl  = generate_discharge_lounge_data()
        st.sidebar.warning("⚠️ Could not reach Google Sheets. Using built-in dummy data.")
else:
    df_w72 = generate_ward72_data()
    df_dl  = generate_discharge_lounge_data()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏥 Ward 72 & Discharge Lounge — Clinical Analytics Dashboard</h1>
    <p>AI & Data Science Unit · Deputy Hospital Director Office · Real-time Performance Monitoring</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & KPIs
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview & KPIs":

    section("🛏️ Ward 72 — Key Performance Indicators")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Total Patients", len(df_w72))
    with c2: kpi_card("Avg Bed Wait (min)",
                       f"{df_w72['Bed_Waiting_Time_min'].mean():.0f}" if 'Bed_Waiting_Time_min' in df_w72 else "N/A",
                       "Target: < 60 min", "green")
    with c3: kpi_card("Avg ED Wait (min)",
                       f"{df_w72['Total_ED_Waiting_Time_min'].mean():.0f}" if 'Total_ED_Waiting_Time_min' in df_w72 else "N/A",
                       "Target: < 240 min", "orange")
    with c4:
        if 'Discharged_Within_72hrs' in df_w72:
            pct = (df_w72['Discharged_Within_72hrs'] == 'Yes').mean() * 100
            col = "green" if pct >= 90 else "red"
            kpi_card("Discharged ≤72hrs", f"{pct:.0f}%", "Target: ≥ 90%", col)
    with c5:
        if 'Actual_LOS_hrs' in df_w72:
            kpi_card("Avg LOS (hrs)", f"{df_w72['Actual_LOS_hrs'].mean():.1f}", "Target: < 48 hrs")
    with c6:
        if 'Readmission_72hrs' in df_w72:
            pct_r = (df_w72['Readmission_72hrs'] == 'Yes').mean() * 100
            col = "green" if pct_r < 5 else "red"
            kpi_card("Readmission Rate", f"{pct_r:.0f}%", "Target: < 5%", col)

    st.markdown("<br>", unsafe_allow_html=True)
    section("🚪 Discharge Lounge — Key Performance Indicators", green=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Total Patients", len(df_dl), colour="green")
    with c2:
        if 'Dwell_Time_min' in df_dl:
            kpi_card("Avg Dwell Time (min)", f"{df_dl['Dwell_Time_min'].mean():.0f}", "Target: < 120 min", "green")
    with c3:
        if 'Total_Discharge_Wait_min' in df_dl:
            kpi_card("Avg Total Discharge Wait", f"{df_dl['Total_Discharge_Wait_min'].mean():.0f} min", "Target: < 180 min", "green")
    with c4:
        if 'Same_Day_Discharge' in df_dl:
            pct_s = (df_dl['Same_Day_Discharge'] == 'Yes').mean() * 100
            col = "green" if pct_s >= 95 else "orange"
            kpi_card("Same-Day Discharge", f"{pct_s:.0f}%", "Target: ≥ 95%", col)
    with c5:
        if 'Satisfaction_Score' in df_dl:
            kpi_card("Avg Satisfaction", f"{df_dl['Satisfaction_Score'].mean():.1f}/5", "Target: ≥ 4.0", "green")
    with c6:
        if 'Unplanned_Return_to_Ward' in df_dl:
            pct_u = (df_dl['Unplanned_Return_to_Ward'] == 'Yes').mean() * 100
            col = "green" if pct_u < 3 else "red"
            kpi_card("Unplanned Return", f"{pct_u:.0f}%", "Target: < 3%", col)

    st.markdown("<br>", unsafe_allow_html=True)
    section("📊 Combined Capacity Overview")

    col1, col2 = st.columns(2)

    with col1:
        # Capacity gauge – Ward 72
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=len(df_w72),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Ward 72 Current Census", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 12]},
                'bar': {'color': "#1F4E79"},
                'steps': [
                    {'range': [0, 6],  'color': "#D6E4F0"},
                    {'range': [6, 10], 'color': "#9DC3E6"},
                    {'range': [10, 12],'color': "#2E75B6"},
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 12}
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=len(df_dl),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Discharge Lounge Current Census", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 8]},
                'bar': {'color': "#375623"},
                'steps': [
                    {'range': [0, 4], 'color': "#E2EFDA"},
                    {'range': [4, 7], 'color': "#A9D18E"},
                    {'range': [7, 8], 'color': "#548235"},
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 8}
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — WARD 72 ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛏️ Ward 72 – Analytics":

    section("🛏️ Ward 72 — Operational Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Patient Distribution by Primary Team")
        if 'Primary_Team' in df_w72:
            team_counts = df_w72['Primary_Team'].value_counts().reset_index()
            team_counts.columns = ['Team', 'Count']
            fig = px.bar(team_counts, x='Count', y='Team', orientation='h',
                         color='Count', color_continuous_scale='Blues',
                         text='Count')
            fig.update_layout(height=300, margin=dict(t=10, b=10),
                               coloraxis_showscale=False, yaxis_title="")
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Gender Distribution")
        if 'Gender' in df_w72:
            g_counts = df_w72['Gender'].value_counts()
            fig = px.pie(values=g_counts.values, names=g_counts.index,
                         color_discrete_sequence=["#1F4E79","#9DC3E6"],
                         hole=0.4)
            fig.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Age Distribution — Ward 72")
        if 'Age' in df_w72:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.histplot(df_w72['Age'], bins=8, kde=True, color='#2E75B6', ax=ax)
            ax.axvline(df_w72['Age'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df_w72['Age'].mean():.1f} yrs")
            ax.set_xlabel("Age (years)"); ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.markdown("##### Bed Waiting Time Distribution (min)")
        if 'Bed_Waiting_Time_min' in df_w72:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.boxplot(x=df_w72['Bed_Waiting_Time_min'], color='#9DC3E6', ax=ax)
            ax.axvline(60, color='red', linestyle='--', label='Target: 60 min')
            ax.set_xlabel("Bed Waiting Time (min)")
            ax.legend(fontsize=8)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    section("⏱️ Time-to-Event Analysis")
    if all(c in df_w72 for c in ['Bed_Waiting_Time_min', 'Total_ED_Waiting_Time_min']):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Waiting Time Breakdown by Patient")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Bed Wait (min)', x=df_w72.index+1,
                y=df_w72['Bed_Waiting_Time_min'], marker_color='#1F4E79'))
            fig.add_trace(go.Bar(
                name='Total ED Wait (min)', x=df_w72.index+1,
                y=df_w72['Total_ED_Waiting_Time_min'], marker_color='#9DC3E6'))
            fig.add_hline(y=60,  line_dash="dot", line_color="red",
                          annotation_text="Bed Wait Target")
            fig.add_hline(y=240, line_dash="dot", line_color="orange",
                          annotation_text="ED Wait Target")
            fig.update_layout(barmode='group', height=320,
                               xaxis_title="Patient No.", yaxis_title="Minutes",
                               margin=dict(t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("##### LOS Distribution vs 72-hour Threshold")
            if 'Actual_LOS_hrs' in df_w72:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ['#C00000' if v > 72 else '#1F4E79' for v in df_w72['Actual_LOS_hrs']]
                ax.barh(range(len(df_w72)), df_w72['Actual_LOS_hrs'], color=colors)
                ax.axvline(72, color='red', linestyle='--', linewidth=2, label='72-hr limit')
                ax.set_xlabel("LOS (hours)")
                ax.set_ylabel("Patient No.")
                ax.legend(fontsize=8)
                sns.despine()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    section("📦 Disposition & Outcome Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        if 'Disposition' in df_w72:
            d_counts = df_w72['Disposition'].value_counts()
            fig = px.pie(values=d_counts.values, names=d_counts.index,
                         title="Disposition at Discharge",
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'Readmission_72hrs' in df_w72:
            r_counts = df_w72['Readmission_72hrs'].value_counts()
            fig = px.pie(values=r_counts.values, names=r_counts.index,
                         title="Readmission Within 72hrs",
                         color_discrete_map={"Yes":"#C00000","No":"#375623"})
            fig.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if 'Adverse_Event' in df_w72:
            ae_counts = df_w72['Adverse_Event'].value_counts()
            fig = px.pie(values=ae_counts.values, names=ae_counts.index,
                         title="Adverse Events in Ward",
                         color_discrete_map={"Yes":"#C55A11","No":"#1F4E79"})
            fig.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    if 'Bed_Waiting_Time_min' in df_w72 and 'Total_ED_Waiting_Time_min' in df_w72:
        bwt_mean = df_w72['Bed_Waiting_Time_min'].mean()
        edwt_mean = df_w72['Total_ED_Waiting_Time_min'].mean()
        insight(f"Average bed waiting time is {bwt_mean:.0f} min. "
                f"{'Within target (< 60 min).' if bwt_mean < 60 else 'Exceeds target — review admission workflow.'}")
        insight(f"Average total ED waiting time is {edwt_mean:.0f} min. "
                f"{'Within target (< 240 min).' if edwt_mean < 240 else 'Exceeds target — consider fast-track pathway review.'}",
                variant="orange")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DISCHARGE LOUNGE ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚪 Discharge Lounge – Analytics":

    section("🚪 Discharge Lounge — Operational Analytics", green=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Dwell Time per Patient (min)")
        if 'Dwell_Time_min' in df_dl:
            colors_dwell = ['#C00000' if v > 120 else '#375623' for v in df_dl['Dwell_Time_min']]
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bars = ax.bar(range(1, len(df_dl)+1), df_dl['Dwell_Time_min'], color=colors_dwell)
            ax.axhline(120, color='red', linestyle='--', linewidth=1.5, label='Target: 120 min')
            ax.set_xlabel("Patient No."); ax.set_ylabel("Dwell Time (min)")
            ax.legend(fontsize=8)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        st.markdown("##### Total Discharge Waiting Time (min)")
        if 'Total_Discharge_Wait_min' in df_dl:
            fig = px.bar(x=list(range(1, len(df_dl)+1)),
                         y=df_dl['Total_Discharge_Wait_min'],
                         labels={'x':'Patient No.','y':'Total Wait (min)'},
                         color=df_dl['Total_Discharge_Wait_min'],
                         color_continuous_scale='Greens')
            fig.add_hline(y=180, line_dash="dot", line_color="red",
                          annotation_text="Target: 180 min")
            fig.update_layout(height=300, margin=dict(t=10, b=10),
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if 'Same_Day_Discharge' in df_dl:
            sd_counts = df_dl['Same_Day_Discharge'].value_counts()
            fig = px.pie(values=sd_counts.values, names=sd_counts.index,
                         title="Same-Day Discharge",
                         color_discrete_map={"Yes":"#375623","No":"#C00000"})
            fig.update_layout(height=260, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'Unplanned_Return_to_Ward' in df_dl:
            ur_counts = df_dl['Unplanned_Return_to_Ward'].value_counts()
            fig = px.pie(values=ur_counts.values, names=ur_counts.index,
                         title="Unplanned Return to Ward",
                         color_discrete_map={"Yes":"#C55A11","No":"#375623"})
            fig.update_layout(height=260, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if 'Satisfaction_Score' in df_dl:
            fig, ax = plt.subplots(figsize=(4, 3))
            score_counts = df_dl['Satisfaction_Score'].value_counts().sort_index()
            ax.bar(score_counts.index, score_counts.values, color='#548235')
            ax.axvline(4.0, color='red', linestyle='--', linewidth=1.5, label='Target: 4.0')
            ax.set_xlabel("Score (1–5)"); ax.set_ylabel("Count")
            ax.set_title("Patient Satisfaction Distribution")
            ax.legend(fontsize=8)
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    section("🏥 Source Ward Analysis", green=True)
    if 'Source_Ward' in df_dl:
        col1, col2 = st.columns(2)
        with col1:
            ward_counts = df_dl['Source_Ward'].value_counts().reset_index()
            ward_counts.columns = ['Ward', 'Count']
            fig = px.bar(ward_counts, x='Ward', y='Count',
                         color='Count', color_continuous_scale='Greens', text='Count')
            fig.update_layout(height=280, margin=dict(t=10, b=10),
                               coloraxis_showscale=False, xaxis_title="Source Ward")
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'Dwell_Time_min' in df_dl:
                dwell_ward = df_dl.groupby('Source_Ward')['Dwell_Time_min'].mean().reset_index()
                fig = px.bar(dwell_ward, x='Source_Ward', y='Dwell_Time_min',
                             title="Avg Dwell Time by Source Ward",
                             color='Dwell_Time_min', color_continuous_scale='RdYlGn_r',
                             text=dwell_ward['Dwell_Time_min'].round(0))
                fig.add_hline(y=120, line_dash="dot", line_color="red")
                fig.update_layout(height=280, margin=dict(t=30, b=10),
                                   coloraxis_showscale=False)
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

    if 'Family_Response_Time_min' in df_dl:
        insight(f"Average family response time: {df_dl['Family_Response_Time_min'].mean():.0f} min. "
                "Consider proactive family notification at the time of morning round discharge decision to reduce lounge dwell time.",
                variant="green")

    if 'Same_Day_Discharge' in df_dl:
        pct_sd = (df_dl['Same_Day_Discharge'] == 'Yes').mean() * 100
        if pct_sd < 95:
            insight(f"Same-day discharge rate is {pct_sd:.0f}% — below 95% target. "
                    "Review late family pickup patterns and consider structured family notification protocol.",
                    variant="orange")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Machine Learning":

    section("🤖 Machine Learning — Predictive Models")

    tab1, tab2, tab3 = st.tabs([
        "📈 LOS Regression (Ward 72)",
        "🎯 Discharge Classifier (Lounge)",
        "📦 Patient Clustering"
    ])

    # ── TAB 1: LOS Regression ──────────────────────────────────────────────
    with tab1:
        st.markdown("##### Predict Actual Length of Stay (hrs) — Random Forest Regressor")
        insight("Features: Age, Bed Waiting Time, Total ED Waiting Time. "
                "Target: Actual LOS (hrs). Trained on Ward 72 data.")

        if all(c in df_w72 for c in ['Age','Bed_Waiting_Time_min','Total_ED_Waiting_Time_min','Actual_LOS_hrs']):
            X = df_w72[['Age','Bed_Waiting_Time_min','Total_ED_Waiting_Time_min']].dropna()
            y = df_w72['Actual_LOS_hrs'].loc[X.index]

            if len(X) > 5:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

                mae  = mean_absolute_error(y_te, y_pred)
                rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                r2   = r2_score(y_te, y_pred)

                c1, c2, c3 = st.columns(3)
                with c1: kpi_card("MAE (hrs)", f"{mae:.2f}")
                with c2: kpi_card("RMSE (hrs)", f"{rmse:.2f}")
                with c3: kpi_card("R² Score", f"{r2:.2f}")

                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.scatter(y_te, y_pred, color='#1F4E79', alpha=0.7, s=80)
                    lims = [min(y_te.min(), y_pred.min())-2, max(y_te.max(), y_pred.max())+2]
                    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
                    ax.set_xlabel("Actual LOS (hrs)")
                    ax.set_ylabel("Predicted LOS (hrs)")
                    ax.set_title("Actual vs Predicted LOS")
                    ax.legend(fontsize=8)
                    sns.despine()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
                    fig, ax = plt.subplots(figsize=(5, 4))
                    importances.plot(kind='barh', color='#2E75B6', ax=ax)
                    ax.set_title("Feature Importance")
                    ax.set_xlabel("Importance Score")
                    sns.despine()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                st.markdown("##### 🔮 Interactive LOS Predictor")
                col1, col2, col3 = st.columns(3)
                with col1: age_in  = st.slider("Patient Age", 18, 90, 50)
                with col2: bwt_in  = st.slider("Bed Waiting Time (min)", 15, 300, 60)
                with col3: edwt_in = st.slider("Total ED Waiting Time (min)", 30, 600, 180)

                pred_los = model.predict([[age_in, bwt_in, edwt_in]])[0]
                flag = "✅ Expected ≤72hrs" if pred_los <= 72 else "⚠️ May exceed 72hrs"
                st.success(f"Predicted LOS: **{pred_los:.1f} hrs** — {flag}")
            else:
                st.warning("Insufficient data for train/test split. Need at least 6 rows.")
        else:
            st.warning("Required columns not found in Ward 72 dataset.")

    # ── TAB 2: Discharge Classifier ────────────────────────────────────────
    with tab2:
        st.markdown("##### Predict Same-Day Discharge — Logistic Regression")
        insight("Features: Age, Dwell Time, Total Discharge Wait, Family Response Time. "
                "Target: Same-Day Discharge (Yes/No).", variant="green")

        if all(c in df_dl for c in ['Age','Dwell_Time_min','Total_Discharge_Wait_min','Family_Response_Time_min','Same_Day_Discharge']):
            X = df_dl[['Age','Dwell_Time_min','Total_Discharge_Wait_min','Family_Response_Time_min']].dropna()
            y_raw = df_dl['Same_Day_Discharge'].loc[X.index]
            le = LabelEncoder()
            y  = le.fit_transform(y_raw)

            if len(X) >= 4 and len(np.unique(y)) > 1:
                scaler = StandardScaler()
                X_sc   = scaler.fit_transform(X)
                model  = LogisticRegression(max_iter=500, random_state=42)
                model.fit(X_sc, y)
                y_pred     = model.predict(X_sc)
                y_prob     = model.predict_proba(X_sc)[:, 1]

                st.text(classification_report(y, y_pred, target_names=le.classes_))

                col1, col2 = st.columns(2)
                with col1:
                    cm = confusion_matrix(y, y_pred)
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
                    ax.set_title("Confusion Matrix")
                    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    coef = pd.Series(model.coef_[0], index=X.columns).sort_values()
                    fig, ax = plt.subplots(figsize=(4, 3.5))
                    colors = ['#C00000' if c < 0 else '#375623' for c in coef]
                    coef.plot(kind='barh', color=colors, ax=ax)
                    ax.set_title("Logistic Regression Coefficients")
                    ax.set_xlabel("Coefficient Value")
                    sns.despine()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            else:
                st.info("Dataset is small. Model trained on all available data as demonstration.")
        else:
            st.warning("Required columns not found in Discharge Lounge dataset.")

    # ── TAB 3: Clustering ──────────────────────────────────────────────────
    with tab3:
        st.markdown("##### Patient Clustering — K-Means (Ward 72)")
        insight("Cluster patients by Age, Bed Waiting Time, and LOS to identify admission pattern groups.")

        if all(c in df_w72 for c in ['Age','Bed_Waiting_Time_min','Actual_LOS_hrs']):
            X_cl = df_w72[['Age','Bed_Waiting_Time_min','Actual_LOS_hrs']].dropna()
            scaler_cl = StandardScaler()
            X_sc_cl   = scaler_cl.fit_transform(X_cl)

            n_clusters = st.slider("Number of Clusters", 2, 4, 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_sc_cl)
            X_cl = X_cl.copy()
            X_cl['Cluster'] = labels.astype(str)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(X_cl, x='Age', y='Actual_LOS_hrs',
                                 color='Cluster', size='Bed_Waiting_Time_min',
                                 color_discrete_sequence=px.colors.qualitative.Bold,
                                 title="Clusters: Age vs LOS",
                                 labels={'Actual_LOS_hrs':'LOS (hrs)'})
                fig.update_layout(height=320, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                cluster_summary = X_cl.groupby('Cluster').agg(
                    Avg_Age=('Age','mean'),
                    Avg_BedWait=('Bed_Waiting_Time_min','mean'),
                    Avg_LOS=('Actual_LOS_hrs','mean'),
                    Count=('Age','count')
                ).round(1)
                st.dataframe(cluster_summary, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ICD-11 NLP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 ICD-11 NLP Analysis":

    section("🔬 ICD-11 NLP Text Analysis — Diagnosis Frequency & TF-IDF")
    insight("In production, this module is replaced by a fine-tuned LLM or scikit-learn pipeline "
            "to automatically map free-text diagnoses to ICD-11 codes. "
            "This demonstration uses TF-IDF vectorisation.")

    tab1, tab2 = st.tabs(["Ward 72 Diagnoses", "Discharge Lounge Diagnoses"])

    def nlp_tab(df, text_col, code_col, colour):
        if text_col not in df.columns:
            st.warning("Diagnosis column not found."); return

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Top Primary Diagnoses")
            dx_counts = df[text_col].value_counts().head(10).reset_index()
            dx_counts.columns = ['Diagnosis','Count']
            fig = px.bar(dx_counts, x='Count', y='Diagnosis', orientation='h',
                         color='Count',
                         color_continuous_scale='Blues' if colour=='blue' else 'Greens',
                         text='Count')
            fig.update_layout(height=350, margin=dict(t=10, b=10),
                               coloraxis_showscale=False, yaxis_title="")
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if code_col in df.columns:
                st.markdown("##### ICD-11 Code Distribution")
                code_counts = df[code_col][df[code_col] != ''].value_counts().reset_index()
                code_counts.columns = ['Code','Count']
                fig = px.treemap(code_counts, path=['Code'], values='Count',
                                 color='Count',
                                 color_continuous_scale='Blues' if colour=='blue' else 'Greens')
                fig.update_layout(height=350, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

        # TF-IDF
        st.markdown("##### TF-IDF Term Importance in Diagnoses")
        corpus = df[text_col].dropna().tolist()
        if len(corpus) >= 2:
            tfidf = TfidfVectorizer(stop_words='english', max_features=20)
            tfidf.fit(corpus)
            tfidf_scores = dict(zip(tfidf.get_feature_names_out(),
                                    tfidf.transform(corpus).toarray().mean(axis=0)))
            tfidf_df = pd.DataFrame(list(tfidf_scores.items()),
                                    columns=['Term','TF-IDF Score']).sort_values('TF-IDF Score', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 3.5))
            bar_color = '#2E75B6' if colour == 'blue' else '#548235'
            ax.barh(tfidf_df['Term'][:15], tfidf_df['TF-IDF Score'][:15], color=bar_color)
            ax.set_xlabel("Mean TF-IDF Score")
            ax.set_title("Top Clinical Terms by TF-IDF Weight")
            ax.invert_yaxis()
            sns.despine()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("##### ICD-11 Auto-Mapping Simulation")
            user_text = st.text_input("Enter free-text diagnosis:",
                                       value="fever with joint pain and low platelet",
                                       key=f"nlp_{text_col}")
            if user_text:
                # Simple keyword matching simulation
                mappings = {
                    "dengue":("Dengue fever","A97.1"),
                    "angina":("Unstable angina","I20.0"),
                    "cellulitis":("Cellulitis","L03.1"),
                    "pneumonia":("Community-acquired pneumonia","J18.9"),
                    "pyelonephritis":("Acute pyelonephritis","N10"),
                    "gastroenteritis":("Acute gastroenteritis","A09"),
                    "copd":("Exacerbation of COPD","J44.1"),
                    "dvt":("Deep vein thrombosis","I82.4"),
                    "anaemia":("Anaemia","D64.9"),
                    "cholecystitis":("Acute cholecystitis","K81.0"),
                    "platelet":("Dengue fever with warning signs","A97.1"),
                    "fever":("Dengue fever","A97.1"),
                }
                text_lower = user_text.lower()
                matched = [(k, v) for k, v in mappings.items() if k in text_lower]
                if matched:
                    kw, (dx_name, dx_code) = matched[0]
                    st.success(f"🏷️ Suggested ICD-11: **{dx_name}** → `{dx_code}` (keyword: '{kw}')")
                    st.caption("In production: replace with fine-tuned BERT/BioBERT or WHO ICD-11 API call.")
                else:
                    st.info("No direct keyword match found. In production, the LLM/NLP model would perform semantic matching against the full ICD-11 ontology.")

    with tab1:
        nlp_tab(df_w72, 'Primary_ICD11_Text', 'Primary_ICD11_Code', 'blue')
    with tab2:
        nlp_tab(df_dl, 'Primary_ICD11_Text', 'Primary_ICD11_Code', 'green')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Raw Data":

    section("📋 Raw Data — Ward 72")
    display_cols_w72 = [c for c in df_w72.columns if c not in ['Patient_Name']]
    st.dataframe(df_w72[display_cols_w72], use_container_width=True, height=350)

    c1, c2, c3 = st.columns(3)
    with c1:
        csv_w72 = df_w72.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Ward 72 CSV", csv_w72,
                           "ward72_data.csv", "text/csv")
    with c2:
        buf = io.StringIO()
        df_w72.describe().to_csv(buf)
        st.download_button("⬇️ Download Ward 72 Summary Stats", buf.getvalue().encode(),
                           "ward72_summary.csv", "text/csv")

    section("📋 Raw Data — Discharge Lounge", green=True)
    display_cols_dl = [c for c in df_dl.columns if c not in ['Patient_Name']]
    st.dataframe(df_dl[display_cols_dl], use_container_width=True, height=280)

    c1, c2 = st.columns(2)
    with c1:
        csv_dl = df_dl.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Discharge Lounge CSV", csv_dl,
                           "discharge_lounge_data.csv", "text/csv")

    section("📊 Descriptive Statistics")
    tab1, tab2 = st.tabs(["Ward 72", "Discharge Lounge"])
    with tab1:
        st.dataframe(df_w72.describe().round(2), use_container_width=True)
    with tab2:
        st.dataframe(df_dl.describe().round(2), use_container_width=True)
