import streamlit as st
import json
import os
import tempfile
import anthropic
import base64
from dotenv import load_dotenv
from datetime import datetime

# ── API Setup ─────────────────────────────────────────────────────
try:
    ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    load_dotenv()
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    st.error("API key not found. Please configure ANTHROPIC_API_KEY in Streamlit secrets.")
    st.stop()

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-sonnet-4-20250514"

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="TherapyAI DEMO — HTPN",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 1rem 1rem 2rem 1rem; max-width: 700px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────
for key, val in [("visitor_count",0),("ai_count",0),
                  ("visited",False),("results",{})]:
    if key not in st.session_state:
        st.session_state[key] = val

if not st.session_state.visited:
    st.session_state.visitor_count += 1
    st.session_state.visited = True

# ════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background: linear-gradient(135deg, #1a1a2e, #0f3460);
     padding: 22px 20px; border-radius: 14px; margin-bottom: 8px;'>
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
        <span style='background:#e74c3c; color:white; font-size:0.7em;
              font-weight:700; padding:3px 10px; border-radius:20px;
              letter-spacing:1px;'>DEMO VERSION</span>
    </div>
    <h1 style='color: #ffffff; margin: 0; font-size: 1.35em; font-weight: 700;'>
        🧠 TherapyAI — Automated Clerking,<br>
        Transcription & Risk Stratification
    </h1>
    <p style='color: #a0c4ff; margin: 8px 0 0 0; font-size: 0.85em;'>
        by <strong>Dr Naim AI Team</strong> ·
        Hospital Tengku Permaisuri Norashikin (HTPN)
    </p>
</div>
""", unsafe_allow_html=True)

st.caption("⚙️ This is a demo version for feasibility and validity testing purposes only.")

# ── Visitor Stats ─────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("👥 Visitors",    st.session_state.visitor_count)
c2.metric("🤖 AI Analyses", st.session_state.ai_count)
c3.metric("📅 Date",        datetime.now().strftime("%d %b %Y"))

# ── Disclaimer ────────────────────────────────────────────────────
st.warning("""
⚠️ **Disclaimer — Demo Version:** This application is designed for the automation
of the therapy clerking process and one-time AI analysis only.
**No patient records are stored.** Audio files are deleted immediately after
transcription. Only anonymised feedback notes are retained for quality improvement.
All AI-generated outputs require clinician review and sign-off before use in any
clinical record. This tool does not replace clinical judgement.
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

input_tab1, input_tab2, input_tab3 = st.tabs([
    "🎙️ Audio Recording",
    "🖼️ Upload Image / Referral Letter",
    "⌨️ Paste Transcript"
])

transcript_text = ""
referral_context = ""

# ── Tab 1: Audio ──────────────────────────────────────────────────
with input_tab1:
    st.info("Supports: .ogg  .wav  .mp3  .m4a  .mp4 · Max 200MB")

    lang_choice = st.radio(
        "Session language:",
        ["English", "Bahasa Malaysia", "Auto-detect"],
        horizontal=True
    )
    lang_map = {
        "English": "en",
        "Bahasa Malaysia": "ms",
        "Auto-detect": None
    }
    selected_lang = lang_map[lang_choice]

    audio_file = st.file_uploader(
        "Upload session recording",
        type=["ogg","wav","mp3","m4a","mp4"],
        label_visibility="collapsed"
    )
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("🎙️ Transcribe Audio Now", use_container_width=True):
            with st.spinner(f"Transcribing in {lang_choice}..."):
                import whisper
                suffix = "." + audio_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                model_w = whisper.load_model("base")
                if selected_lang:
                    result = model_w.transcribe(tmp_path, language=selected_lang)
                else:
                    result = model_w.transcribe(tmp_path)
                st.session_state["transcript"] = result["text"]
                detected = result.get("language","unknown")
                os.unlink(tmp_path)
            st.success(f"✅ Transcription complete · Detected language: {detected}")

    if "transcript" in st.session_state:
        st.session_state["transcript"] = st.text_area(
            "Transcribed text (editable before analysis)",
            value=st.session_state["transcript"], height=180
        )
        transcript_text = st.session_state["transcript"]

# ── Tab 2: Image Upload ───────────────────────────────────────────
with input_tab2:
    st.info("""
    Upload a photo or scan of:
    - Referral letter
    - Handwritten clinical notes
    - Previous assessment forms
    - Discharge summary

    Claude will extract the clinical information and incorporate it into the analysis.
    """)

    image_file = st.file_uploader(
        "Upload image",
        type=["jpg","jpeg","png","webp"],
        label_visibility="collapsed"
    )

    if image_file is not None:
        st.image(image_file, caption="Uploaded document", use_column_width=True)
        if st.button("📄 Extract Clinical Information from Image",
                     use_container_width=True):
            with st.spinner("Extracting clinical information..."):
                img_bytes  = image_file.read()
                img_b64    = base64.standard_b64encode(img_bytes).decode("utf-8")
                ext        = image_file.name.split(".")[-1].lower()
                media_map  = {"jpg":"image/jpeg","jpeg":"image/jpeg",
                              "png":"image/png","webp":"image/webp"}
                media_type = media_map.get(ext, "image/jpeg")

                response = client.messages.create(
                    model=MODEL,
                    max_tokens=1500,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": img_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": """Extract all clinical information from this document.
Structure the output as:
DOCUMENT TYPE:
PATIENT DETAILS (if present):
REFERRING CLINICIAN:
REASON FOR REFERRAL / PRESENTING COMPLAINT:
RELEVANT HISTORY:
CURRENT MEDICATIONS:
INVESTIGATIONS:
CLINICAL FINDINGS:
RECOMMENDATIONS:
OTHER RELEVANT INFORMATION:

If any section is not present in the document, write 'Not stated'.
Be precise and do not add information not present in the document."""
                            }
                        ]
                    }]
                )
                referral_context = response.content[0].text
                st.session_state["referral_context"] = referral_context

            st.success("✅ Clinical information extracted")

    if "referral_context" in st.session_state:
        st.session_state["referral_context"] = st.text_area(
            "Extracted clinical information (editable)",
            value=st.session_state["referral_context"], height=200
        )
        referral_context = st.session_state["referral_context"]

# ── Tab 3: Paste Transcript ───────────────────────────────────────
with input_tab3:
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

def run_claude(prompt, max_tokens=2000):
    r = client.messages.create(model=MODEL, max_tokens=max_tokens,
        messages=[{"role":"user","content":prompt}])
    return r.content[0].text

def parse_json(text):
    import re
    # Remove markdown fences
    clean = text.replace("```json","").replace("```","").strip()
    # Find first { and last } to extract JSON only
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON found in response: {clean[:200]}")
    clean = clean[start:end]
    return json.loads(clean)

# Gather all available input sources
final_transcript = (
    transcript_text or
    st.session_state.get("transcript", "") or
    st.session_state.get("referral_context", "") or
    referral_context
)

if run_btn and final_transcript:
    st.session_state.ai_count += 1
    date_str  = session_date.strftime("%Y-%m-%d")

    combined_context = ""
    if referral_context:
        combined_context += f"\n\nREFERRAL / DOCUMENT CONTEXT:\n{referral_context}"
    if transcript_text:
        combined_context += f"\n\nSESSION TRANSCRIPT:\n{transcript_text}"

    progress = st.progress(0, text="Starting AI pipeline...")

    progress.progress(10, text="Stage 1 of 5 — Thematic analysis...")
    analysis = parse_json(run_claude("""
Analyse this therapy session. Transcript may be in English or Bahasa Malaysia.
Generate analysis in English. Return ONLY valid JSON:
{"presenting_themes":[],"cognitive_distortions":[],
"therapeutic_alliance":0,"patient_engagement":0,
"disclosure_depth":"","resistance_indicators":[],
"therapist_technique":[],"rupture_repair":false,
"risk_flags":[],"session_quality_score":0,
"key_patient_quotes":[],"summary":""}
""" + combined_context))

    progress.progress(30, text="Stage 2 of 5 — Clinical note...")
    note_text = run_claude(f"""
Generate structured clinical note for EMR in English.
DATE: {date_str} | PATIENT: {patient_id} | THERAPIST: {therapist_id} | SESSION: {session_num}
Sections: PRESENTING COMPLAINT, MENTAL STATE EXAMINATION
(Appearance, Speech, Mood, Affect, Thought Form, Thought Content,
Perception, Cognition, Insight, Judgement),
RISK ASSESSMENT (Self-harm, Harm to others, Safeguarding),
SESSION CONTENT, RESPONSE TO THERAPY, MANAGEMENT PLAN, NEXT SESSION FOCUS.
End with: *** AI-GENERATED — REQUIRES CLINICIAN REVIEW BEFORE FILING ***
{combined_context[:2500]}
ANALYSIS: {json.dumps(analysis)}""")

    progress.progress(50, text="Stage 3 of 5 — ICD-11 / DSM-5...")
    codes = parse_json(run_claude("""
Suggest diagnostic codes. Return ONLY valid JSON:
{"icd11_primary":{"code":"","description":""},
"icd11_secondary":[],"dsm5_primary":{"code":"","description":"","criteria_met":[]},
"dsm5_differential":[],"coding_confidence":"","coding_notes":"",
"suggested_assessments":[]}
ANALYSIS: """ + json.dumps(analysis), max_tokens=1000))

    progress.progress(70, text="Stage 4 of 5 — Recommendations...")
    rec = parse_json(run_claude("""
Clinical supervision recommendations. Return ONLY valid JSON:
{"guided_questions_next_session":[],"therapy_technique_suggestions":[],
"therapy_modality_adjustment":"","homework_suggestions":[],
"escalation_required":false,"escalation_reason":null,
"supervisor_review_recommended":false,"overall_recommendation":""}
ANALYSIS: """ + json.dumps(analysis), max_tokens=1000))

    progress.progress(90, text="Stage 5 of 5 — Documentation quality check...")
    audit = parse_json(run_claude("""
Review this clinical note for documentation completeness and quality.
Return ONLY valid JSON:
{"consent_documented":false,"risk_assessment_complete":false,
"safeguarding_addressed":false,"management_plan_present":false,
"follow_up_documented":false,
"missing_elements":[],
"documentation_quality_score":0,
"overall_completeness":"",
"standard_recommendations":[],
"priority_actions":[]}
NOTE: """ + note_text[:1000], max_tokens=1000))

    progress.progress(100, text="✅ Complete")
    st.session_state.results = {
        "analysis":analysis,"note_text":note_text,"codes":codes,
        "rec":rec,"audit":audit,"date_str":date_str}
    st.success("✅ Analysis complete — review all outputs before filing")

elif run_btn:
    st.error("Please transcribe an audio file, extract an image, or paste a transcript first.")

# ════════════════════════════════════════════════════════════════
# SECTION 4 — RESULTS
# ════════════════════════════════════════════════════════════════
if st.session_state.results:
    r         = st.session_state.results
    analysis  = r["analysis"]
    note_text = r["note_text"]
    codes     = r["codes"]
    rec       = r["rec"]
    audit     = r["audit"]
    date_str  = r["date_str"]

    st.divider()
    st.subheader("④ Summary Metrics")
    m1,m2,m3 = st.columns(3)
    m1.metric("Session Quality",      f"{analysis.get('session_quality_score')}/10")
    m2.metric("Therapeutic Alliance", f"{analysis.get('therapeutic_alliance')}/5")
    m3.metric("Patient Engagement",   f"{analysis.get('patient_engagement')}/5")
    m4,m5 = st.columns(2)
    m4.metric("Doc Completeness", audit.get('overall_completeness','—').upper())
    m5.metric("Escalation", "⚠️ YES" if rec.get('escalation_required') else "✅ NO")

    if analysis.get("risk_flags"):
        st.error(f"🚨 Risk Flags: {' · '.join(analysis['risk_flags'])}")
    else:
        st.success("✅ No risk flags identified")

    st.divider()
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "📋 Analysis",
        "📄 Clinical Note",
        "🏷️ ICD-11 / DSM-5",
        "💡 Recommendations",
        "📝 Documentation Quality"
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
            from docx import Document
            os.makedirs("outputs/clinical_notes", exist_ok=True)
            doc = Document()
            doc.add_heading("Clinical Session Note — CONFIDENTIAL", 0)
            for line in edited_note.split("\n"): doc.add_paragraph(line)
            fp = f"outputs/clinical_notes/{patient_id}_{date_str}_session{session_num}.docx"
            doc.save(fp)
            st.success(f"✅ Saved: {fp}")

    with tab3:
        st.warning("⚠️ AI-suggested only — clinician must confirm before filing")
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
        st.write("**Modality Adjustment:**",
                 rec.get("therapy_modality_adjustment","—"))
        st.write("**Homework Suggestions**")
        for h in rec.get("homework_suggestions",[]): st.write(f"• {h}")
        st.info(rec.get("overall_recommendation","—"))
        if rec.get("escalation_required"):
            st.error(f"🚨 Escalation Required: {rec.get('escalation_reason','—')}")

    with tab5:
        st.info("""
        **Documentation Quality & Completeness Checker**
        This section reviews your clinical note against standard documentation
        requirements to prevent incomplete records and ensure clinical safety.
        """)
        score = audit.get("documentation_quality_score", 0)
        st.metric("Documentation Quality Score", f"{score}/10")

        col1,col2 = st.columns(2)
        with col1:
            st.write(f"Consent: `{audit.get('consent_documented')}`")
            st.write(f"Risk assessment: `{audit.get('risk_assessment_complete')}`")
            st.write(f"Safeguarding: `{audit.get('safeguarding_addressed')}`")
        with col2:
            st.write(f"Management plan: `{audit.get('management_plan_present')}`")
            st.write(f"Follow-up: `{audit.get('follow_up_documented')}`")
            st.write(f"**Completeness:** `{audit.get('overall_completeness')}`")

        if audit.get("missing_elements"):
            st.warning("**Missing Documentation Elements:**")
            for m in audit["missing_elements"]: st.write(f"• {m}")

        if audit.get("priority_actions"):
            st.error("**Priority Actions Required:**")
            for a in audit["priority_actions"]: st.write(f"• {a}")

        if audit.get("standard_recommendations"):
            st.write("**Standard Documentation Recommendations:**")
            for r in audit["standard_recommendations"]: st.write(f"• {r}")

    # ── HITL Sign-off ──────────────────────────────────────────────
    st.divider()
    st.subheader("⑤ Clinician Sign-off")
    st.warning("All AI outputs require clinician review and approval before filing.")
    clinician_name = st.text_input("Clinician Name")
    designation    = st.text_input("Designation / MMC Number")
    sign_date      = st.date_input("Review Date", value=datetime.now())
    approved = st.checkbox(
        "✅ I have reviewed all AI outputs and approve filing in HIS / EMR")
    if approved and clinician_name:
        st.success(
            f"✅ Approved by {clinician_name} ({designation}) on {sign_date}")
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
        "Medical Officer","Nurse","Other"])
    fb_validity    = st.slider(
        "AI output validity (1=not valid, 5=highly valid)", 1,5,3)
    fb_feasibility = st.slider(
        "Clinical feasibility (1=not feasible, 5=highly feasible)", 1,5,3)
    fb_useful = st.multiselect("Most useful features:", [
        "Audio transcription",
        "Image / referral letter extraction",
        "Thematic analysis",
        "Clinical note generation",
        "ICD-11 / DSM-5 coding",
        "Clinical recommendations",
        "Documentation quality checker",
        "Clinician sign-off"])
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
        with open(
            f"outputs/feedback/fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "w") as f:
            json.dump(entry, f, indent=2)
        st.success("✅ Thank you. Feedback submitted.")

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#888;font-size:0.75em;padding:8px;'>
🧠 <strong>TherapyAI — Demo Version</strong> ·
Dr Naim AI Team · Hospital Tengku Permaisuri Norashikin (HTPN) · 2026<br>
<em>Clinical automation support only. No patient records stored.
All outputs require clinician verification before use.</em>
</div>
""", unsafe_allow_html=True)
