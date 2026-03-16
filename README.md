# Therapy AI Pipeline

A comprehensive pipeline for processing therapy session audio recordings, generating transcripts, clinical notes, and reports.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**

   Copy the `.env` file and update the values with your actual credentials:

   - `ANTHROPIC_API_KEY`: Your Anthropic API key for AI processing (requires credits)
   - `GEMINI_API_KEY`: Your Google Gemini API key for AI processing (free tier available, upgrade for higher limits)
   - `GOOGLE_CREDENTIALS_PATH`: Path to your Google service account credentials JSON file
   - `GOOGLE_SHEET_ID`: ID of your Google Sheet for data storage (replace 'placeholder')
   - `GOOGLE_DRIVE_FOLDER_ID`: ID of your Google Drive folder for file storage (replace 'placeholder')

   **Important**:
   - Never commit the `.env` file to version control. It contains sensitive API keys.
   - Ensure you have sufficient credits in your Anthropic account for API usage.
   - Google Gemini has a free tier with rate limits - upgrade your plan for higher usage limits.

   **Important**: 
   - Never commit the `.env` file to version control. It contains sensitive API keys.
   - Ensure you have sufficient credits in your Anthropic account for API usage.

3. **Google Credentials Setup**

   - Create a Google Cloud Project
   - Enable the Google Sheets and Google Drive APIs
   - Create a service account and download the credentials JSON file
   - Place the JSON file at `config/google_credentials.json`
   - Share your Google Sheet and Drive folder with the service account email

## Usage

Run the complete pipeline:
```bash
python pipeline/run_pipeline.py
```

Or run individual steps:
- `python pipeline/01_transcribe.py` - Audio transcription
- `python pipeline/02_analyse.py` - Session analysis
- `python pipeline/03_document.py` - Clinical documentation
- And so on...

## Project Structure

- `audio/recordings/` - Input audio files
- `config/` - Configuration files
- `dashboard/` - Web dashboard (app.py)
- `outputs/` - Generated outputs (transcripts, notes, reports)
- `pipeline/` - Processing pipeline steps