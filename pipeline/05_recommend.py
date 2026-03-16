import anthropic, json, os
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_recommendations(analysis):
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": """
Provide clinical supervision recommendations. Return ONLY valid JSON:
{
  "guided_questions_next_session": [],
  "under_explored_areas": [],
  "therapy_technique_suggestions": [],
  "therapy_modality_adjustment": "",
  "homework_suggestions": [],
  "escalation_required": false,
  "escalation_reason": null,
  "supervisor_review_recommended": false,
  "overall_recommendation": ""
}
ANALYSIS: """ + json.dumps(analysis, indent=2)}]
    )
    raw = response.content[0].text
    clean = raw.replace("```json","").replace("```","").strip()
    return json.loads(clean)

if __name__ == "__main__":
    print("Recommendations module ready.")
