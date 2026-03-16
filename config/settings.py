import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_CREDENTIALS  = os.getenv("GOOGLE_CREDENTIALS_PATH")
GOOGLE_SHEET_ID     = os.getenv("GOOGLE_SHEET_ID")
GOOGLE_DRIVE_FOLDER = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

WHISPER_MODEL  = "base"
CLAUDE_MODEL   = "claude-sonnet-4-20250514"
AUDIO_DIR      = "audio/recordings"
TRANSCRIPT_DIR = "outputs/transcripts"
NOTES_DIR      = "outputs/clinical_notes"
REPORTS_DIR    = "outputs/reports"
