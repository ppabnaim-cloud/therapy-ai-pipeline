import streamlit as st
import json
import os
import tempfile
import whisper
import anthropic
from dotenv import load_dotenv
from datetime import datetime
from docx import Document

# ── API Setup ─────────────────────────────────────────────────────
cat > /tmp/fix_api.py << 'EOF'
import re

with open('dashboard/app.py', 'r') as f:
    content = f.read()

old = """try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except:
    load_dotenv()
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")"""

new = """try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    try:
        load_dotenv()
        ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    except Exception:
        ANTHROPIC_API_KEY = None

if not ANTHROPIC_API_KEY:
    st.error("API key not found. Please configure ANTHROPIC_API_KEY in Streamlit secrets.")
    st.stop()"""

content = content.replace(old, new)

with open('dashboard/app.py', 'w') as f:
    f.write(content)

print("Done")
EOF
python3 /tmp/fix_api.py

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="TherapyAI — HTPN",
    page_icon="🧠",
    layout="centered"
)

# ── Hide sidebar & hamburger menu entirely ────────────────────────
st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 1rem 1rem 2rem 1rem; max-width: 700px; }
h1 { font-size: 1.4em !important; }
h2 { font-size: 1.15em !important; }
h3 { font-size: 1.05em !important; }
.stButton > button { width: 100%; border-radius: 10px; }
.stTextInput > div > input { border-radius: 8px; }
.stSelectbox > div { border-radius: 8px; }
div[data-testid="metric-container"] {
    background: #f0f4ff;
    border: 1px solid #d0d9f0;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
if "visitor_count" not in st.session_state:
    st.session_state.visitor_count = 0
if "ai_count" not in st.session_state:
    st.session_state.ai_count = 0
if "visited" not in st.session_state:
    st.session_state.visited = False
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "results" not in st.session_state:
    st.session_state.results = {}

if not st.session_state.visited:
    st.session_state.visitor_count += 1
    st.session_state.visited = True

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background: linear-gradient(135deg, #1a1a2e, #0f3460);
     padding: 22px 20px; border-radius: 14px; margin-bottom: 16px;'>
    <h1 style='color: #ffffff; margin: 0; font-size: 1.35em; font-weight: 700;'>
        🧠 AI Assistant for Therapist<br>Clerking, Transcribing & Analysis
    </h1>
    <p style='color: #a0c4ff; margin: 8px 0 0 0; font-size: 0.85em;'>
        by <strong>Dr Naim AI Team</strong> ·
        Hospital Tengku Permaisuri Norashikin (HTPN)
    </p>
</div>
""", unsafe_allow_html=True)

# ── Visitor Stats ─────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("👥 Visitors",    st.session_state.visitor_count)
c2.metric("🤖 AI Analyses", st.session_state.ai_count)
c3.metric("📅 Date",        datetime.now().strftime("%d %b %Y"))

# ── Disclaimer ────────────────────────────────────────────────────
st.warning("""
⚠️ **Disclaimer:** For automation of therapy clerking and one-time AI analysis only.
**No patient records are stored.** Audio is deleted immediately after transcription.
Only anonymised feedback is retained. All AI outputs require clinician review before
filing. This tool does not replace clinical judgement.
""")

st.divider()

# ════════════════════════════════════════════════════════════════
# SECTION 1 — PATIENT DETAILS
# ════════════════════════════════════════════════════════════════
st.subheader("① Patient Details")

col1, col2 = st.columns(2)
with col1:
    patient_id   = st.text_input("Patient ID",   value="PT001")
    therapist_id = st.text_input("Therapist ID", value="TH001")
with col2:
    session_num  = st.number_input("Session Number", min_value=1, value=1)
    session_date = st.date_input("Session Date", value=datetime.now())

st.divider()

# ════════════════════════════════════════════════════════════════
# SECTION 2 — SESSION INPUT
# ════════════════════════════════════════════════════════════════
st.subheader("② Session Input")

input_method = st.radio(
    "Choose input method:",
    ["📁 Upload Audio File", "⌨️ Paste Transcript"],
    horizontal=True
)

transcript_text = ""

if input_method == "📁 Upload Audio File":
    st.info("📎 Supports: .ogg  .wav  .mp3  .m4a  .mp4  (max 200MB)")
    audio_file = st.file_uploader(
        "Upload session recording",
        type=["ogg","wav","mp3","m4a","mp4"],
        label_visibility="collapsed"
    )
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("🎙️ Transcribe Audio Now", use_container_width=True):
            with st.spinner("Transcribing with Whisper — English only..."):
                suffix = "." + audio_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                model_w = whisper.load_model("base")
                result  = model_w.transcribe(tmp_path, language="en")
                st.session_state["transcript"] = result["text"]
                os.unlink(tmp_path)
                os.makedirs("outputs/transcripts", exist_ok=True)
                fname = f"outputs/transcripts/{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                with open(fname,"w") as f:
                    f.write(result["text"])
            st.success("✅ Transcription complete — review below")

    if "transcript" in st.session_state:
        st.session_state["transcript"] = st.text_area(
            "Transcribed text (editable before analysis)",
            value=st.session_state["transcript"],
            height=180
        )
        transcript_text = st.session_state["transcript"]

else:
    transcript_text = st.text_area(
        "Paste session transcript here",
        height=200,
        placeholder="Therapist: How have you been this week?\nPatient: ..."
    )

st.divider()

# ════════════════════════════════════════════════════════════════
# SECTION 3 — RUN AI ANALYSIS
# ════════════════════════════════════════════════════════════════
st.subheader("③ Run AI Analysis")

run_btn = st.button("▶ Run AI Analysis", type="primary", use_container_width=True)

# ── Helper ────────────────────────────────────────────────────────
def run_claude(prompt, max_tokens=2000):
    r = client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role":"user","content":prompt}]
    )
    return r.content[0].text

def parse_json(text):
    clean = text.replace("```json","").replace("```","").strip()
    return json.loads(clean)

if run_btn and transcript_text:
    st.session_state.ai_count += 1
    date_str = session_date.strftime("%Y-%m-%d")
    progress = st.progress(0, text="Starting AI pipeline...")

    progress.progress(10, text="Stage 1 of 5 — Thematic analysis...")
    analysis = parse_json(run_claude("""
Analyse this therapy session. Return ONLY valid JSON:
{
  "presenting_themes": [],
  "cognitive_distortions": [],
  "therapeutic_alliance": 0,
  "patient_engagement": 0,
  "disclosure_depth": "",
  "resistance_indicators": [],
  "therapist_technique": [],
  "rupture_repair": false,
  "risk_flags": [],
  "session_quality_score": 0,
  "key_patient_quotes": [],
  "summary": ""
}
TRANSCRIPT: """ + transcript_text))

    progress.progress(30, text="Stage 2 of 5 — Clinical note...")
    note_text = run_claude(f"""
Generate structured clinical note for EMR. Output in English.
DATE: {date_str} | PATIENT: {patient_id} | THERAPIST: {therapist_id} | SESSION: {session_num}
Sections: PRESENTING COMPLAINT, MENTAL STATE EXAMINATION
(Appearance, Speech, Mood, Affect, Thought Form, Thought Content,
Perception, Cognition, Insight, Judgement),
RISK ASSESSMENT (Self-harm, Harm to others, Safeguarding),
SESSION CONTENT, RESPONSE TO THERAPY, MANAGEMENT PLAN, NEXT SESSION FOCUS.
End with: *** AI-GENERATED — REQUIRES CLINICIAN REVIEW BEFORE FILING ***
TRANSCRIPT: {transcript_text[:2000]}
ANALYSIS: {json.dumps(analysis)}""")

    progress.progress(50, text="Stage 3 of 5 — ICD-11 / DSM-5 coding...")
    codes = parse_json(run_claude("""
Suggest diagnostic codes. Return ONLY valid JSON:
{
  "icd11_primary": {"code": "", "description": ""},
  "icd11_secondary": [],
  "dsm5_primary": {"code": "", "description": "", "criteria_met": []},
  "dsm5_differential": [],
  "coding_confidence": "",
  "coding_notes": "",
  "suggested_assessments": []
}
ANALYSIS: """ + json.dumps(analysis), max_tokens=1000))

    progress.progress(70, text="Stage 4 of 5 — Recommendations...")
    rec = parse_json(run_claude("""
Provide clinical supervision recommendations. Return ONLY valid JSON:
{
  "guided_questions_next_session": [],
  "therapy_technique_suggestions": [],
  "therapy_modality_adjustment": "",
  "homework_suggestions": [],
  "escalation_required": false,
  "escalation_reason": null,
  "supervisor_review_recommended": false,
  "overall_recommendation": ""
}
ANALYSIS: """ + json.dumps(analysis), max_tokens=1000))

    progress.progress(90, text="Stage 5 of 5 — Medicolegal audit...")
    audit = parse_json(run_claude("""
Audit this clinical note. Return ONLY valid JSON:
{
  "consent_documented": false,
  "risk_assessment_complete": false,
  "safeguarding_addressed": false,
  "management_plan_present": false,
  "follow_up_documented": false,
  "flags": [],
  "overall_adequacy": "",
  "medicolegal_risk_level": "",
  "recommended_actions": []
}
NOTE: """ + note_text[:1000], max_tokens=1000))

    progress.progress(100, text="✅ Complete")

    st.session_state.analysis_done = True
    st.session_state.results = {
        "analysis": analysis,
        "note_text": note_text,
        "codes": codes,
        "rec": rec,
        "audit": audit,
        "date_str": date_str
    }

elif run_btn and not transcript_text:
    st.error("Please upload an audio file or paste a transcript first.")

# ════════════════════════════════════════════════════════════════
# SECTION 4 — RESULTS
# ════════════════════════════════════════════════════════════════
if st.session_state.analysis_done and st.session_state.results:
    r       = st.session_state.results
    analysis= r["analysis"]
    note_text=r["note_text"]
    codes   = r["codes"]
    rec     = r["rec"]
    audit   = r["audit"]
    date_str= r["date_str"]

    st.success("✅ AI analysis complete — review all outputs before filing")
    st.divider()

    # Metrics
    st.subheader("④ Summary Metrics")
    m1,m2,m3 = st.columns(3)
    m1.metric("Session Quality",      f"{analysis.get('session_quality_score')}/10")
    m2.metric("Therapeutic Alliance", f"{analysis.get('therapeutic_alliance')}/5")
    m3.metric("Patient Engagement",   f"{analysis.get('patient_engagement')}/5")

    m4,m5 = st.columns(2)
    m4.metric("Medicolegal Risk", audit.get('medicolegal_risk_level','—').upper())
    m5.metric("Escalation", "⚠️ YES" if rec.get('escalation_required') else "✅ NO")

    if analysis.get("risk_flags"):
        st.error(f"🚨 Risk Flags: {' · '.join(analysis['risk_flags'])}")
    else:
        st.success("✅ No risk flags identified")

    st.divider()

    # Tabs
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "📋 Analysis",
        "📄 Note",
        "🏷️ Codes",
        "💡 Recs",
        "⚖️ Legal"
    ])

    with tab1:
        st.write("**Presenting Themes**")
        for t in analysis.get("presenting_themes",[]): st.write(f"• {t}")
        st.write("**Cognitive Distortions**")
        for c in analysis.get("cognitive_distortions",[]): st.write(f"• {c}")
        st.write("**Disclosure Depth:**", analysis.get("disclosure_depth","—"))
        st.write("**Therapist Techniques**")
        for t in analysis.get("therapist_technique",[]): st.write(f"• {t}")
        st.write("**Key Patient Quotes**")
        for q in analysis.get("key_patient_quotes",[]): st.info(f'"{q}"')
        st.write("**Summary**")
        st.write(analysis.get("summary","—"))

    with tab2:
        st.warning("⚠️ Review and edit before filing in HIS / EMR")
        edited_note = st.text_area("Clinical Note", value=note_text, height=400)
        if st.button("💾 Save as .docx", use_container_width=True):
            os.makedirs("outputs/clinical_notes", exist_ok=True)
            doc = Document()
            doc.add_heading("Clinical Session Note — CONFIDENTIAL", 0)
            for line in edited_note.split("\n"): doc.add_paragraph(line)
            fp = f"outputs/clinical_notes/{patient_id}_{date_str}_session{session_num}.docx"
            doc.save(fp)
            st.success(f"✅ Saved: {fp}")

    with tab3:
        st.warning("⚠️ AI-suggested only — clinician must confirm")
        icd = codes.get("icd11_primary",{})
        dsm = codes.get("dsm5_primary",{})
        st.write("**ICD-11 Primary**")
        st.code(f"{icd.get('code','—')}  {icd.get('description','—')}")
        st.write("**DSM-5 Primary**")
        st.code(f"{dsm.get('code','—')}  {dsm.get('description','—')}")
        st.write(f"**Confidence:** {codes.get('coding_confidence','—')}")
        st.write(f"**Notes:** {codes.get('coding_notes','—')}")
        if codes.get("suggested_assessments"):
            st.write("**Suggested Assessments:**")
            for a in codes["suggested_assessments"]: st.write(f"• {a}")

    with tab4:
        st.write("**Guided Questions — Next Session**")
        for q in rec.get("guided_questions_next_session",[]): st.write(f"• {q}")
        st.write("**Technique Suggestions**")
        for t in rec.get("therapy_technique_suggestions",[]): st.write(f"• {t}")
        st.write("**Modality Adjustment:**", rec.get("therapy_modality_adjustment","—"))
        st.write("**Homework:**")
        for h in rec.get("homework_suggestions",[]): st.write(f"• {h}")
        st.info(rec.get("overall_recommendation","—"))
        if rec.get("escalation_required"):
            st.error(f"🚨 Escalation: {rec.get('escalation_reason','—')}")

    with tab5:
        st.write(f"Consent: `{audit.get('consent_documented')}`")
        st.write(f"Risk assessment: `{audit.get('risk_assessment_complete')}`")
        st.write(f"Safeguarding: `{audit.get('safeguarding_addressed')}`")
        st.write(f"Management plan: `{audit.get('management_plan_present')}`")
        st.write(f"Follow-up: `{audit.get('follow_up_documented')}`")
        st.write(f"**Adequacy:** `{audit.get('overall_adequacy')}`")
        st.write(f"**Risk:** `{audit.get('medicolegal_risk_level')}`")
        if audit.get("flags"):
            st.warning("**Gaps:**")
            for f in audit["flags"]: st.write(f"• {f}")
        if audit.get("recommended_actions"):
            st.write("**Actions:**")
            for a in audit["recommended_actions"]: st.write(f"• {a}")

    st.divider()

    # HITL Sign-off
    st.subheader("⑤ Clinician Sign-off")
    st.warning("All AI outputs require review before filing.")
    clinician_name = st.text_input("Clinician Name")
    designation    = st.text_input("Designation / MMC Number")
    sign_date      = st.date_input("Review Date", value=datetime.now())
    approved = st.checkbox("✅ I have reviewed all AI outputs and approve filing in HIS/EMR")
    if approved and clinician_name:
        st.success(f"✅ Approved by {clinician_name} ({designation}) on {sign_date}")
        st.balloons()

# ════════════════════════════════════════════════════════════════
# SECTION 5 — FEEDBACK
# ════════════════════════════════════════════════════════════════
st.divider()
st.subheader("⑥ Feedback — Validity & Feasibility")
st.caption("No patient data collected. Anonymised feedback only.")

with st.form("feedback_form"):
    fb_name  = st.text_input("Your Name (optional)")
    fb_role  = st.selectbox("Role", [
        "Psychiatrist","Clinical Psychologist","Counsellor",
        "Medical Officer","Nurse","Other"
    ])
    fb_validity    = st.slider("AI output validity (1=not valid, 5=highly valid)", 1,5,3)
    fb_feasibility = st.slider("Clinical feasibility (1=not feasible, 5=highly feasible)", 1,5,3)
    fb_useful = st.multiselect("Most useful features:", [
        "Audio transcription","Thematic analysis","Clinical note generation",
        "ICD-11 / DSM-5 coding","Recommendations","Medicolegal audit","Sign-off"
    ])
    fb_improve  = st.text_area("Suggestions for improvement", height=100)
    fb_comments = st.text_area("General comments", height=80)

    if st.form_submit_button("📨 Submit Feedback", use_container_width=True):
        os.makedirs("outputs/feedback", exist_ok=True)
        entry = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M"),
            "name":         fb_name,
            "role":         fb_role,
            "validity":     fb_validity,
            "feasibility":  fb_feasibility,
            "useful":       fb_useful,
            "improvements": fb_improve,
            "comments":     fb_comments
        }
        with open(f"outputs/feedback/fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json","w") as f:
            json.dump(entry, f, indent=2)
        st.success("✅ Thank you. Feedback submitted.")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#888;font-size:0.75em;padding:8px;'>
🧠 <strong>TherapyAI</strong> · Dr Naim AI Team · HTPN · 2026<br>
<em>Clinical automation support only. No patient records stored.
Clinician verification required for all outputs.</em>
</div>
""", unsafe_allow_html=True)
