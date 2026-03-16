import streamlit as st
import json
import os
import tempfile
import whisper
import anthropic
from dotenv import load_dotenv
from datetime import datetime
from docx import Document

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-20250514"

st.set_page_config(
    page_title="Therapy AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Talking Therapy — AI Clinician Dashboard")
st.caption("AI-assisted session analysis | Human-in-the-Loop verification required")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Patient Details")
    patient_id   = st.text_input("Patient ID",   value="PT001")
    therapist_id = st.text_input("Therapist ID", value="TH001")
    session_num  = st.number_input("Session Number", min_value=1, value=1)
    st.divider()

    st.header("Session Input")
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload Audio File", "⌨️ Paste Transcript"],
        index=0
    )

    transcript_text = ""

    if input_method == "📁 Upload Audio File":
        st.info("Supports: .ogg, .wav, .mp3, .m4a, .mp4")
        audio_file = st.file_uploader(
            "Upload session recording",
            type=["ogg", "wav", "mp3", "m4a", "mp4"]
        )
        if audio_file is not None:
            with st.spinner("🎙️ Transcribing audio with Whisper..."):
                # Save to temp file
                suffix = "." + audio_file.name.split(".")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                # Transcribe
                model_whisper = whisper.load_model("base")
                result = model_whisper.transcribe(tmp_path, language='en')
                transcript_text = result["text"]
                os.unlink(tmp_path)

                # Save transcript
                os.makedirs("outputs/transcripts", exist_ok=True)
                fname = f"outputs/transcripts/{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
                with open(fname, "w") as f:
                    f.write(transcript_text)

            st.success("✅ Transcription complete")
            st.text_area("Transcribed text (editable)", value=transcript_text,
                        height=200, key="transcribed")
            transcript_text = st.session_state.get("transcribed", transcript_text)

    else:
        transcript_text = st.text_area(
            "Paste transcript here",
            height=300,
            placeholder="Therapist: How have you been this week?\nPatient: ..."
        )

    st.divider()
    run_btn = st.button("▶ Run AI Analysis", type="primary", use_container_width=True)

# ── Run Pipeline ─────────────────────────────────────────────────
def run_claude(prompt, max_tokens=2000):
    r = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text

def parse_json(text):
    clean = text.replace("```json","").replace("```","").strip()
    return json.loads(clean)

if run_btn and transcript_text:
    date_str = datetime.now().strftime("%Y-%m-%d")

    col_prog, _ = st.columns([3,1])
    progress = st.progress(0, text="Starting AI pipeline...")

    # Stage 1
    progress.progress(10, text="Stage 1 — Thematic analysis...")
    analysis = parse_json(run_claude("""
Analyse this therapy session. Transcript may be in Bahasa Malaysia or English.
Return ONLY valid JSON:
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

    # Stage 2
    progress.progress(30, text="Stage 2 — Generating clinical note...")
    note_text = run_claude(f"""
Generate structured clinical note for EMR copy-paste. Output in English.
DATE: {date_str} | PATIENT: {patient_id} | THERAPIST: {therapist_id} | SESSION: {session_num}
Sections required:
PRESENTING COMPLAINT:
MENTAL STATE EXAMINATION:
  Appearance and Behaviour:
  Speech:
  Mood (subjective):
  Affect (observed):
  Thought Form:
  Thought Content:
  Perception:
  Cognition:
  Insight:
  Judgement:
RISK ASSESSMENT:
  Self-harm:
  Harm to others:
  Safeguarding:
SESSION CONTENT:
RESPONSE TO THERAPY:
MANAGEMENT PLAN:
NEXT SESSION FOCUS:
CLINICIAN SIGNATURE: _________________ DATE: _________________
*** AI-GENERATED — REQUIRES CLINICIAN REVIEW BEFORE FILING ***
TRANSCRIPT: {transcript_text[:2000]}
ANALYSIS: {json.dumps(analysis)}""")

    # Stage 3
    progress.progress(50, text="Stage 3 — ICD-11 / DSM-5 coding...")
    codes = parse_json(run_claude("""
Suggest diagnostic codes based on this analysis.
Return ONLY valid JSON:
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

    # Stage 4
    progress.progress(70, text="Stage 4 — Generating recommendations...")
    rec = parse_json(run_claude("""
Provide clinical supervision recommendations.
Return ONLY valid JSON:
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

    # Stage 5
    progress.progress(90, text="Stage 5 — Medicolegal audit...")
    audit = parse_json(run_claude("""
Audit this clinical note for medicolegal adequacy.
Return ONLY valid JSON:
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

    progress.progress(100, text="✅ All stages complete")
    st.success("✅ AI analysis complete — review all outputs below before filing")

    # ── Metrics ───────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Session Quality",      f"{analysis.get('session_quality_score')}/10")
    m2.metric("Therapeutic Alliance", f"{analysis.get('therapeutic_alliance')}/5")
    m3.metric("Patient Engagement",   f"{analysis.get('patient_engagement')}/5")
    m4.metric("Medicolegal Risk",      audit.get('medicolegal_risk_level','—').upper())
    m5.metric("Escalation",           "⚠️ YES" if rec.get('escalation_required') else "✅ NO")

    # ── Risk Alert ────────────────────────────────────────────────
    st.divider()
    if analysis.get("risk_flags"):
        st.error(f"🚨 Risk Flags Identified: {' · '.join(analysis['risk_flags'])}")
    else:
        st.success("✅ No risk flags identified")

    # ── Tabs ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Thematic Analysis",
        "📄 Clinical Note",
        "🏷️ ICD-11 / DSM-5",
        "💡 Recommendations",
        "⚖️ Medicolegal Audit"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Presenting Themes")
            for t in analysis.get("presenting_themes", []):
                st.write(f"• {t}")
            st.subheader("Cognitive Distortions")
            for c in analysis.get("cognitive_distortions", []):
                st.write(f"• {c}")
            st.subheader("Disclosure Depth")
            st.write(analysis.get("disclosure_depth","—"))
        with col2:
            st.subheader("Therapist Techniques Observed")
            for t in analysis.get("therapist_technique", []):
                st.write(f"• {t}")
            st.subheader("Key Patient Quotes")
            for q in analysis.get("key_patient_quotes", []):
                st.info(f'"{q}"')
        st.subheader("Session Summary")
        st.write(analysis.get("summary","—"))

    with tab2:
        st.subheader("AI-Generated Clinical Note")
        st.warning("⚠️ Review and edit before filing in HIS / EMR")
        edited_note = st.text_area("Clinical Note (editable)", value=note_text, height=500)
        if st.button("💾 Save as .docx"):
            os.makedirs("outputs/clinical_notes", exist_ok=True)
            doc = Document()
            doc.add_heading("Clinical Session Note — CONFIDENTIAL", 0)
            for line in edited_note.split("\n"):
                doc.add_paragraph(line)
            fp = f"outputs/clinical_notes/{patient_id}_{date_str}_session{session_num}.docx"
            doc.save(fp)
            st.success(f"✅ Saved: {fp}")

    with tab3:
        st.subheader("Suggested Diagnostic Codes")
        st.warning("⚠️ AI-suggested only — clinician must confirm before filing")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**ICD-11 Primary**")
            icd = codes.get("icd11_primary", {})
            st.code(f"{icd.get('code','—')}  {icd.get('description','—')}")
            st.write("**ICD-11 Secondary**")
            for s in codes.get("icd11_secondary", []):
                st.write(f"• {s}")
        with col2:
            st.write("**DSM-5 Primary**")
            dsm = codes.get("dsm5_primary", {})
            st.code(f"{dsm.get('code','—')}  {dsm.get('description','—')}")
            st.write("**DSM-5 Differential**")
            for d in codes.get("dsm5_differential", []):
                st.write(f"• {d}")
        st.write(f"**Confidence:** {codes.get('coding_confidence','—')}")
        st.write(f"**Notes:** {codes.get('coding_notes','—')}")
        if codes.get("suggested_assessments"):
            st.write("**Suggested Assessments:**")
            for a in codes["suggested_assessments"]:
                st.write(f"• {a}")

    with tab4:
        st.subheader("Clinical Supervision Recommendations")
        st.write("**Guided Questions for Next Session:**")
        for q in rec.get("guided_questions_next_session", []):
            st.write(f"• {q}")
        st.write("**Therapy Technique Suggestions:**")
        for t in rec.get("therapy_technique_suggestions", []):
            st.write(f"• {t}")
        st.write("**Therapy Modality Adjustment:**")
        st.write(rec.get("therapy_modality_adjustment","—"))
        st.write("**Homework Suggestions:**")
        for h in rec.get("homework_suggestions", []):
            st.write(f"• {h}")
        st.write("**Overall Recommendation:**")
        st.info(rec.get("overall_recommendation","—"))
        if rec.get("escalation_required"):
            st.error(f"🚨 Escalation Required: {rec.get('escalation_reason','—')}")

    with tab5:
        st.subheader("Medicolegal Documentation Audit")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Consent documented: `{audit.get('consent_documented')}`")
            st.write(f"Risk assessment complete: `{audit.get('risk_assessment_complete')}`")
            st.write(f"Safeguarding addressed: `{audit.get('safeguarding_addressed')}`")
            st.write(f"Management plan present: `{audit.get('management_plan_present')}`")
            st.write(f"Follow-up documented: `{audit.get('follow_up_documented')}`")
        with col2:
            st.write(f"**Overall adequacy:** `{audit.get('overall_adequacy')}`")
            st.write(f"**Medicolegal risk:** `{audit.get('medicolegal_risk_level')}`")
        if audit.get("flags"):
            st.warning("**Documentation Gaps:**")
            for f in audit["flags"]:
                st.write(f"• {f}")
        if audit.get("recommended_actions"):
            st.write("**Recommended Actions:**")
            for a in audit["recommended_actions"]:
                st.write(f"• {a}")

    # ── HITL Sign-off ─────────────────────────────────────────────
    st.divider()
    st.subheader("⚕️ Human-in-the-Loop Clinician Sign-off")
    st.warning("All AI outputs require clinician review and approval before filing.")
    col1, col2 = st.columns(2)
    with col1:
        clinician_name = st.text_input("Clinician Name")
        designation    = st.text_input("Designation / MMC Number")
    with col2:
        sign_date = st.date_input("Date of Review")
        approved  = st.checkbox("I have reviewed all AI outputs and approve filing in HIS/EMR")
    if approved and clinician_name:
        st.success(f"✅ Approved by {clinician_name} ({designation}) on {sign_date}")
        st.balloons()

elif run_btn and not transcript_text:
    st.warning("Please upload an audio file or paste a transcript before running.")
else:
    st.info("👈 Upload audio or paste transcript in the sidebar, then click Run AI Analysis.")
