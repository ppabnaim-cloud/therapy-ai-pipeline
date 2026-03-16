import anthropic, json, os
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_codes(analysis):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": """
Suggest ICD-11 and DSM-5 codes for this session. Return ONLY valid JSON:
{
  "icd11_primary": {"code": "", "description": ""},
  "icd11_secondary": [],
  "dsm5_primary": {"code": "", "description": "", "criteria_met": []},
  "dsm5_differential": [],
  "coding_confidence": "",
  "coding_notes": "",
  "requires_further_assessment": true,
  "suggested_assessments": []
}
*** AI-suggested only. Clinician must confirm. ***
ANALYSIS: """ + json.dumps(analysis, indent=2)}]
    )
    raw = response.content[0].text
    clean = raw.replace("```json","").replace("```","").strip()
    codes = json.loads(clean)
    codes["patient_id"] = analysis["patient_id"]
    codes["date"]       = analysis["date"]
    return codes

if __name__ == "__main__":
    print("Coding module ready.")
