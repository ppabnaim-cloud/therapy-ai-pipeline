import anthropic, json, os
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def analyse_transcript(transcript):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": """
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
TRANSCRIPT: """ + transcript["full_text"]}]
    )
    raw = response.content[0].text
    clean = raw.replace("```json","").replace("```","").strip()
    analysis = json.loads(clean)
    analysis["patient_id"]   = transcript["patient_id"]
    analysis["therapist_id"] = transcript["therapist_id"]
    analysis["date"]         = transcript["date"]
    return analysis

if __name__ == "__main__":
    print("Analysis module ready.")
