import anthropic, json, os
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def audit_documentation(note_text, analysis):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": """
Audit this clinical note for medicolegal adequacy. Return ONLY valid JSON:
{
  "consent_documented": false,
  "risk_assessment_complete": false,
  "capacity_assessment_documented": false,
  "safeguarding_addressed": false,
  "confidentiality_limits_documented": false,
  "management_plan_present": false,
  "follow_up_documented": false,
  "flags": [],
  "overall_adequacy": "",
  "medicolegal_risk_level": "",
  "recommended_actions": []
}
NOTE: """ + note_text + """
ANALYSIS: """ + json.dumps(analysis, indent=2)}]
    )
    raw = response.content[0].text
    clean = raw.replace("```json","").replace("```","").strip()
    return json.loads(clean)

if __name__ == "__main__":
    print("Medicolegal module ready.")
