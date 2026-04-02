import os
import json
import logging
from google import genai 
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL   = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash') 

_client = genai.Client(api_key=GEMINI_API_KEY)


def _format_conversation(messages: list) -> str:
    lines = []
    for msg in messages:
        role = msg.get('role', 'Unknown')
        text = msg.get('text', '')
        lines.append(f"[{role}]: {text}")
    return '\n'.join(lines)


def _format_reference_data(ref_data: dict) -> str:
    lines = []
    for section, items in ref_data.items():
        lines.append(f"\n## {section}")
        if isinstance(items, list):
            if items and isinstance(items[0], dict):
                for item in items:
                    lines.append("  - " + ", ".join(f"{k}: {v}" for k, v in item.items()))
            else:
                lines.append("  " + ", ".join(str(x) for x in items))
    return '\n'.join(lines)


def analyze_session(session_data: dict) -> dict:
    """
    Send session conversation, result JSON, and reference data to Gemini.
    Returns an analysis dict with overall_status, issues list, and summary.
    """
    conversation = session_data.get('conversation', [])
    result_json = session_data.get('result_json')
    reference_data = session_data.get('reference_data', {})

    # Skip analysis for sessions with no conversation or no result
    if not conversation or result_json is None:
        return {
            'overall_status': 'pending',
            'issues': [],
            'summary': 'No conversation or result data available for analysis.',
        }

    conv_text = _format_conversation(conversation)
    result_text = json.dumps(result_json, indent=2)
    ref_text = _format_reference_data(reference_data)

    prompt = f"""You are a quality assurance AI for a travel quotation system.
Your job is to verify that the AI quotation assistant correctly captured ALL information
from the user conversation into the result JSON, using the available reference data.

=== CONVERSATION ===
{conv_text}

=== RESULT JSON (produced by quotation AI) ===
{result_text}

=== REFERENCE DATA (valid options in the system) ===
{ref_text}

=== YOUR TASK ===
Carefully identify any of these issues:
1. Information mentioned in the conversation that is MISSING from the result JSON
2. Wrong IDs used (supplier_id, hotel_id, room_type_id, etc.) — cross-check names vs IDs in reference data
3. Fields that should be filled based on the conversation but are null or missing
4. Values in the JSON that contradict or don't match what the user requested
5. Reference data items mentioned in conversation that don't exist in the reference data

Return ONLY a valid JSON object — no markdown, no extra text:
{{
  "overall_status": "ok",
  "issues": [
    {{
      "type": "missing_field|wrong_id|wrong_value|not_in_reference|null_field",
      "field": "exact field path in result JSON (e.g. hotels[0].meal_type)",
      "expected": "what it should be",
      "actual": "what it currently is",
      "description": "clear human-readable explanation",
      "severity": "high|medium|low"
    }}
  ],
  "summary": "1-2 sentence summary of the analysis result"
}}

Rules:
- Set overall_status to "error" if any high-severity issues exist
- Set overall_status to "warning" if only medium/low severity issues exist
- Set overall_status to "ok" if no issues found
- Empty issues array means no problems
"""

    try:
        response = _client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text.strip()

        # Strip markdown fences if present
        if text.startswith('```'):
            parts = text.split('```')
            text = parts[1] if len(parts) > 1 else text
            if text.startswith('json'):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Gemini JSON parse error: {e}")
        return {
            'overall_status': 'error',
            'issues': [],
            'summary': f'Failed to parse AI response: {e}',
        }
    except Exception as e:
        logger.error(f"Gemini analysis error: {e}")
        return {
            'overall_status': 'error',
            'issues': [],
            'summary': f'Analysis error: {str(e)}',
        }
