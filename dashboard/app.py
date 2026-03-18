import streamlit as st
import json, os, tempfile, anthropic, base64
import requests as _req
from dotenv import load_dotenv
from datetime import datetime

try:
    ANTHROPIC_API_KEY = st.secrets['ANTHROPIC_API_KEY']
except Exception:
    load_dotenv()
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if not ANTHROPIC_API_KEY:
    st.error('API key not found.')
    st.stop()

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = 'claude-sonnet-4-20250514'

st.set_page_config(page_title='TherapyAI DEMO', page_icon='🧠', layout='centered')
st.markdown('<style>[data-testid="collapsedControl"]{display:none}section[data-testid="stSidebar"]{display:none}.block-container{padding:1rem;max-width:700px}</style>', unsafe_allow_html=True)

_JBIN_KEY = '$2a$10$brr7j5hjNvmonkjb.pzdG.soVYQMdRQwe3fiATQEPziiE81WGjqfO'
_JBIN_BIN = '69b9ff17c3097a1dd5351110'
_JBIN_URL = f'https://api.jsonbin.io/v3/b/{_JBIN_BIN}'
_JBIN_HDR = {'X-Access-Key': _JBIN_KEY, 'Content-Type': 'application/json'}

def get_counts():
    try:
        r = _req.get(_JBIN_URL + '/latest', headers=_JBIN_HDR, timeout=5)
        return r.json().get('record', {'visitors': 0, 'analyses': 0})
    except Exception:
        return {'visitors': 0, 'analyses': 0}

def save_counts(data):
    try:
        _req.put(_JBIN_URL, json=data, headers=_JBIN_HDR, timeout=5)
    except Exception:
        pass

for key, val in [('visited', False), ('results', {}), ('analysis_just_done', False)]:
    if key not in st.session_state:
        st.session_state[key] = val

_counts = get_counts()
if not st.session_state.visited:
    _counts['visitors'] = _counts.get('visitors', 0) + 1
    save_counts(_counts)
    st.session_state.visited = True
    st.rerun()

st.markdown("""
<div style='background:linear-gradient(135deg,#1a1a2e,#0f3460);padding:22px 20px;border-radius:14px;margin-bottom:8px;'>
<span style='background:#e74c3c;color:white;font-size:0.7em;font-weight:700;padding:3px 10px;border-radius:20px;'>DEMO VERSION</span>
<h1 style='color:#fff;margin:8px 0 0;font-size:1.35em;font-weight:700;'>🧠 TherapyAI — Automated Clerking, Transcription & Risk Stratification</h1>
<p style='color:#a0c4ff;margin:6px 0 0;font-size:0.85em;'>by <strong>Dr Naim AI Team</strong> · Hospital Tengku Permaisuri Norashikin (HTPN)</p>
</div>""", unsafe_allow_html=True)

st.caption('⚙️ Demo version for feasibility and validity testing purposes only.')

live = get_counts()
c1,c2,c3 = st.columns(3)
c1.metric('👥 Total Visitors',  live.get('visitors', 0))
c2.metric('🤖 AI Analyses Run', live.get('analyses', 0))
c3.metric('📅 Today',           datetime.now().strftime('%d %b %Y'))

st.warning('⚠️ **Disclaimer — Demo Version:** For automation of therapy clerking and one-time AI analysis only. **No patient records are stored.** Audio deleted immediately after transcription. Only anonymised feedback retained. All AI outputs require clinician review before filing. This tool does not replace clinical judgement.')

st.divider()
st.subheader('① Patient Details')
col1,col2 = st.columns(2)
with col1:
    patient_id   = st.text_input('Patient ID',   value='PT001')
    therapist_id = st.text_input('Therapist ID', value='TH001')
with col2:
    session_num  = st.number_input('Session Number', min_value=1, value=1)
    session_date = st.date_input('Session Date', value=datetime.now())

st.divider()
st.subheader('② Session Input')
st.caption('Upload audio, image, or paste transcript — only one is needed.')

t1,t2,t3 = st.tabs(['🎙️ Audio Recording','🖼️ Upload Image / Referral Letter','⌨️ Paste Transcript'])
transcript_text = ''
referral_context = ''

with t1:
    st.info('Supports: .ogg .wav .mp3 .m4a .mp4 · Max 200MB')
    lang = st.radio('Session language:', ['English','Bahasa Malaysia','Auto-detect'], horizontal=True)
    lmap = {'English':'en','Bahasa Malaysia':'ms','Auto-detect':None}
    af = st.file_uploader('Upload session recording', type=['ogg','wav','mp3','m4a','mp4'], label_visibility='collapsed')
    if af:
        st.audio(af)
        if st.button('🎙️ Transcribe Audio Now', use_container_width=True):
            with st.spinner(f'Transcribing in {lang}...'):
                import whisper
                sx = '.' + af.name.split('.')[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=sx) as tmp:
                    tmp.write(af.read()); tp = tmp.name
                mw = whisper.load_model('base')
                res = mw.transcribe(tp, language=lmap[lang]) if lmap[lang] else mw.transcribe(tp)
                st.session_state['transcript'] = res['text']
                os.unlink(tp)
            st.success(f"✅ Transcription complete · Detected language: {res.get('language','?')}")
    if 'transcript' in st.session_state:
        st.session_state['transcript'] = st.text_area('Transcribed text (editable before analysis)', value=st.session_state['transcript'], height=180)
        transcript_text = st.session_state['transcript']

with t2:
    st.info('Upload a photo of a referral letter, handwritten notes, assessment form, or discharge summary.')
    img = st.file_uploader('Upload image', type=['jpg','jpeg','png','webp'], label_visibility='collapsed')
    if img:
        st.image(img, caption='Uploaded document', use_column_width=True)
        if st.button('📄 Extract Clinical Information from Image', use_container_width=True):
            with st.spinner('Extracting clinical information...'):
                ib = img.read()
                ib64 = base64.standard_b64encode(ib).decode()
                mt = {'jpg':'image/jpeg','jpeg':'image/jpeg','png':'image/png','webp':'image/webp'}.get(img.name.split('.')[-1].lower(),'image/jpeg')
                rr = client.messages.create(model=MODEL, max_tokens=1500,
                    messages=[{'role':'user','content':[
                        {'type':'image','source':{'type':'base64','media_type':mt,'data':ib64}},
                        {'type':'text','text':'Extract clinical info. Structure as: DOCUMENT TYPE: PATIENT DETAILS: REFERRING CLINICIAN: REASON FOR REFERRAL: RELEVANT HISTORY: CURRENT MEDICATIONS: INVESTIGATIONS: CLINICAL FINDINGS: RECOMMENDATIONS: Write Not stated if absent.'}
                    ]}])
                st.session_state['referral_context'] = rr.content[0].text
            st.success('✅ Clinical information extracted')
    if 'referral_context' in st.session_state:
        st.session_state['referral_context'] = st.text_area('Extracted clinical information (editable)', value=st.session_state['referral_context'], height=200)
        referral_context = st.session_state['referral_context']

with t3:
    transcript_text = st.text_area('Paste session transcript here', height=200, placeholder='Therapist: How have you been this week?\nPatient: ...')

st.divider()
st.subheader('③ Run AI Analysis')
run_btn = st.button('▶ Run AI Analysis', type='primary', use_container_width=True)

def run_claude(prompt, max_tokens=2000):
    return client.messages.create(model=MODEL, max_tokens=max_tokens, messages=[{'role':'user','content':prompt}]).content[0].text

def parse_json(text):
    clean = text.replace('```json','').replace('```','').strip()
    s = clean.find('{'); e = clean.rfind('}')+1
    if s == -1 or e == 0: raise ValueError('No JSON')
    return json.loads(clean[s:e])

ft = transcript_text or st.session_state.get('transcript','') or st.session_state.get('referral_context','') or referral_context

if run_btn and ft:
    _c = get_counts()
    _c['analyses'] = _c.get('analyses', 0) + 1
    save_counts(_c)
    st.session_state['analysis_just_done'] = True

    ds = session_date.strftime('%Y-%m-%d')
    ref = st.session_state.get('referral_context','') or referral_context
    tra = st.session_state.get('transcript','') or transcript_text
    ctx = ''
    if ref: ctx += f'\n\nREFERRAL:\n{ref}'
    if tra: ctx += f'\n\nTRANSCRIPT:\n{tra}'

    pg = st.progress(0, text='Starting...')
    pg.progress(10, text='Stage 1 of 5 — Thematic analysis...')
    analysis = parse_json(run_claude('Analyse therapy session. Transcript may be English or BM. Output English JSON only: {"presenting_themes":[],"cognitive_distortions":[],"therapeutic_alliance":0,"patient_engagement":0,"disclosure_depth":"","resistance_indicators":[],"therapist_technique":[],"rupture_repair":false,"risk_flags":[],"session_quality_score":0,"key_patient_quotes":[],"summary":""}\n' + ctx))

    pg.progress(30, text='Stage 2 of 5 — Clinical note...')
    note_text = run_claude(f'Generate structured clinical note for EMR in English. DATE:{ds} PATIENT:{patient_id} THERAPIST:{therapist_id} SESSION:{session_num}\nSections: PRESENTING COMPLAINT, MENTAL STATE EXAMINATION (Appearance Speech Mood Affect Thought-Form Thought-Content Perception Cognition Insight Judgement), RISK ASSESSMENT (Self-harm Harm-to-others Safeguarding), SESSION CONTENT, RESPONSE TO THERAPY, MANAGEMENT PLAN, NEXT SESSION FOCUS.\nEnd with: ***AI-GENERATED — REQUIRES CLINICIAN REVIEW BEFORE FILING***\n{ctx[:2500]}\nANALYSIS:{json.dumps(analysis)}')

    pg.progress(50, text='Stage 3 of 5 — ICD-11 / DSM-5...')
    codes = parse_json(run_claude('Suggest diagnostic codes. JSON only: {"icd11_primary":{"code":"","description":""},"icd11_secondary":[],"dsm5_primary":{"code":"","description":"","criteria_met":[]},"dsm5_differential":[],"coding_confidence":"","coding_notes":"","suggested_assessments":[]}\nANALYSIS:' + json.dumps(analysis), max_tokens=1000))

    pg.progress(70, text='Stage 4 of 5 — Recommendations...')
    rec = parse_json(run_claude('Clinical supervision recommendations. JSON only: {"guided_questions_next_session":[],"therapy_technique_suggestions":[],"therapy_modality_adjustment":"","homework_suggestions":[],"escalation_required":false,"escalation_reason":null,"supervisor_review_recommended":false,"overall_recommendation":""}\nANALYSIS:' + json.dumps(analysis), max_tokens=1000))

    pg.progress(90, text='Stage 5 of 5 — Documentation quality...')
    audit = parse_json(run_claude('Documentation completeness review. JSON only: {"consent_documented":false,"risk_assessment_complete":false,"safeguarding_addressed":false,"management_plan_present":false,"follow_up_documented":false,"missing_elements":[],"documentation_quality_score":0,"overall_completeness":"","standard_recommendations":[],"priority_actions":[]}\nNOTE:' + note_text[:800], max_tokens=1000))

    pg.progress(100, text='✅ Complete')
    st.session_state.results = {'analysis':analysis,'note_text':note_text,'codes':codes,'rec':rec,'audit':audit,'date_str':ds}
    st.success('✅ Analysis complete — review all outputs before filing')

elif run_btn:
    st.error('Please transcribe audio, extract an image, or paste a transcript first.')

if st.session_state.get('analysis_just_done'):
    st.session_state['analysis_just_done'] = False
    st.rerun()

if st.session_state.results:
    rv = st.session_state.results
    analysis  = rv['analysis']
    note_text = rv['note_text']
    codes     = rv['codes']
    rec       = rv['rec']
    audit     = rv['audit']
    ds        = rv['date_str']

    st.divider()
    st.subheader('④ Summary Metrics')
    m1,m2,m3 = st.columns(3)
    m1.metric('Session Quality',      f"{analysis.get('session_quality_score')}/10")
    m2.metric('Therapeutic Alliance', f"{analysis.get('therapeutic_alliance')}/5")
    m3.metric('Patient Engagement',   f"{analysis.get('patient_engagement')}/5")
    m4,m5 = st.columns(2)
    m4.metric('Doc Completeness', audit.get('overall_completeness','—').upper())
    m5.metric('Escalation', '⚠️ YES' if rec.get('escalation_required') else '✅ NO')

    if analysis.get('risk_flags'):
        st.error(f"🚨 Risk Flags: {' · '.join(analysis['risk_flags'])}")
    else:
        st.success('✅ No risk flags identified')

    st.divider()
    tb1,tb2,tb3,tb4,tb5 = st.tabs(['📋 Analysis','📄 Clinical Note','🏷️ ICD-11 / DSM-5','💡 Recommendations','📝 Documentation Quality'])

    with tb1:
        st.write('**Presenting Themes**')
        for t in analysis.get('presenting_themes',[]): st.write(f'• {t}')
        st.write('**Cognitive Distortions**')
        for c in analysis.get('cognitive_distortions',[]): st.write(f'• {c}')
        st.write(f"**Disclosure Depth:** {analysis.get('disclosure_depth','—')}")
        st.write('**Therapist Techniques**')
        for t in analysis.get('therapist_technique',[]): st.write(f'• {t}')
        st.write('**Key Patient Quotes**')
        for q in analysis.get('key_patient_quotes',[]): st.info(f'"{q}"')
        st.write('**Summary**')
        st.write(analysis.get('summary','—'))

    with tb2:
        st.warning('⚠️ Review and edit before filing in HIS / EMR')
        en = st.text_area('Clinical Note (editable)', value=note_text, height=400)
        st.write('**Download Options:**')
        d1,d2 = st.columns(2)
        with d1:
            st.download_button('⬇️ Download .txt (Notepad)', data=en.encode(), file_name=f'{patient_id}_{ds}_session{session_num}_note.txt', mime='text/plain', use_container_width=True)
        with d2:
            from docx import Document as DD
            from io import BytesIO
            dc = DD()
            dc.add_heading('Clinical Session Note — CONFIDENTIAL', 0)
            dc.add_paragraph(f'Patient: {patient_id}  |  Therapist: {therapist_id}  |  Date: {ds}  |  Session: {session_num}')
            dc.add_paragraph('—' * 40)
            for line in en.split('\n'): dc.add_paragraph(line)
            bf = BytesIO(); dc.save(bf); bf.seek(0)
            st.download_button('⬇️ Download .docx (Word)', data=bf.read(), file_name=f'{patient_id}_{ds}_session{session_num}_note.docx', mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document', use_container_width=True)
        st.info('💡 .txt → phone Notes app → copy-paste into HIS/EMR  |  .docx → Word or Google Docs')

    with tb3:
        st.warning('⚠️ AI-suggested only — clinician must confirm before filing')
        icd = codes.get('icd11_primary',{})
        dsm = codes.get('dsm5_primary',{})
        st.write('**ICD-11 Primary**'); st.code(f"{icd.get('code','—')}  {icd.get('description','—')}")
        st.write('**DSM-5 Primary**');  st.code(f"{dsm.get('code','—')}  {dsm.get('description','—')}")
        st.write(f"**Confidence:** {codes.get('coding_confidence','—')}  |  **Notes:** {codes.get('coding_notes','—')}")
        if codes.get('suggested_assessments'):
            st.write('**Suggested Assessments:**')
            for a in codes['suggested_assessments']: st.write(f'• {a}')

    with tb4:
        st.write('**Guided Questions — Next Session**')
        for q in rec.get('guided_questions_next_session',[]): st.write(f'• {q}')
        st.write('**Technique Suggestions**')
        for t in rec.get('therapy_technique_suggestions',[]): st.write(f'• {t}')
        st.write(f"**Modality Adjustment:** {rec.get('therapy_modality_adjustment','—')}")
        st.write('**Homework Suggestions**')
        for h in rec.get('homework_suggestions',[]): st.write(f'• {h}')
        st.info(rec.get('overall_recommendation','—'))
        if rec.get('escalation_required'):
            st.error(f"🚨 Escalation Required: {rec.get('escalation_reason','—')}")

    with tb5:
        st.info('**Documentation Quality & Completeness Checker** — Reviews your clinical note against standard documentation requirements to prevent incomplete records and support clinical safety.')
        st.metric('Documentation Quality Score', f"{audit.get('documentation_quality_score',0)}/10")
        col1,col2 = st.columns(2)
        with col1:
            st.write(f"Consent: `{audit.get('consent_documented')}`")
            st.write(f"Risk assessment: `{audit.get('risk_assessment_complete')}`")
            st.write(f"Safeguarding: `{audit.get('safeguarding_addressed')}`")
        with col2:
            st.write(f"Management plan: `{audit.get('management_plan_present')}`")
            st.write(f"Follow-up: `{audit.get('follow_up_documented')}`")
            st.write(f"**Completeness:** `{audit.get('overall_completeness')}`")
        if audit.get('missing_elements'):
            st.warning('**Missing Documentation Elements:**')
            for m in audit['missing_elements']: st.write(f'• {m}')
        if audit.get('priority_actions'):
            st.error('**Priority Actions Required:**')
            for a in audit['priority_actions']: st.write(f'• {a}')
        if audit.get('standard_recommendations'):
            st.write('**Standard Documentation Recommendations:**')
            for r in audit['standard_recommendations']: st.write(f'• {r}')

    st.divider()
    st.subheader('⑤ Clinician Sign-off')
    st.warning('All AI outputs require clinician review and approval before filing.')
    clinician_name = st.text_input('Clinician Name')
    designation    = st.text_input('Designation / MMC Number')
    sign_date      = st.date_input('Review Date', value=datetime.now())
    approved = st.checkbox('✅ I have reviewed all AI outputs and approve filing in HIS / EMR')
    if approved and clinician_name:
        st.success(f'✅ Approved by {clinician_name} ({designation}) on {sign_date}')
        st.balloons()

st.divider()
st.subheader('⑥ Feedback — Validity & Feasibility')
st.caption('No patient data collected. Feedback sent directly to Dr Naim.')
with st.form('feedback_form'):
    fb_name  = st.text_input('Your Name (optional)')
    fb_role  = st.selectbox('Role', ['Psychiatrist','Clinical Psychologist','Counsellor','Medical Officer','Nurse','Other'])
    fb_validity    = st.slider('AI output validity (1=not valid, 5=highly valid)', 1,5,3)
    fb_feasibility = st.slider('Clinical feasibility (1=not feasible, 5=highly feasible)', 1,5,3)
    fb_useful = st.multiselect('Most useful features:', ['Audio transcription','Image / referral letter extraction','Thematic analysis','Clinical note generation','ICD-11 / DSM-5 coding','Clinical recommendations','Documentation quality checker','Clinician sign-off'])
    fb_improve  = st.text_area('Suggestions for improvement', height=100)
    fb_comments = st.text_area('General comments', height=80)
    if st.form_submit_button('📨 Submit Feedback', use_container_width=True):
        try:
            resp = _req.post('https://api.emailjs.com/api/v1.0/email/send',
                json={'service_id':'service_z0rh03y','template_id':'template_6bzqjul','user_id':'jRs1TWPZym8iUjNck',
                    'template_params':{'to_email':'ppnaim@moh.gov.my','name':fb_name or 'Anonymous','role':fb_role,
                    'validity':str(fb_validity),'feasibility':str(fb_feasibility),
                    'useful':', '.join(fb_useful) if fb_useful else 'None',
                    'improvements':fb_improve or 'None','comments':fb_comments or 'None',
                    'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M')}},
                headers={'Content-Type':'application/json'}, timeout=10)
            st.success('✅ Thank you. Feedback sent to Dr Naim (ppnaim@moh.gov.my)') if resp.status_code == 200 else st.warning(f'Email failed: {resp.text}')
        except Exception as e:
            st.warning(f'Email failed: {str(e)}')

st.divider()
st.markdown("<div style='text-align:center;color:#888;font-size:0.75em;padding:8px;'>🧠 <strong>TherapyAI — Demo Version</strong> · Dr Naim AI Team · Hospital Tengku Permaisuri Norashikin (HTPN) · 2026<br><em>Clinical automation support only. No patient records stored. All outputs require clinician verification before use.</em></div>", unsafe_allow_html=True)
