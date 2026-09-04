import gradio as gr
import os
import shutil
import gc
import requests
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
import gspread
from google.oauth2.service_account import Credentials

# ── Environment ───────────────────────────────────
load_dotenv(override=True)

# FFmpeg auto-detection for Windows
if not shutil.which("ffmpeg"):
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        winget_packages_dir = os.path.join(
            local_appdata, "Microsoft", "WinGet", "Packages"
        )
        if os.path.exists(winget_packages_dir):
            for root, dirs, files in os.walk(winget_packages_dir):
                if "ffmpeg.exe" in files:
                    os.environ["PATH"] = root + os.pathsep + os.environ["PATH"]
                    print(f"FFmpeg found at: {root}")
                    break

INTRON_API_KEY         = os.environ.get("INTRON_API_KEY", "")
OPENAI_API_KEY         = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_CREDENTIALS_PATH = os.environ.get("GOOGLE_CREDENTIALS_PATH", "")
GOOGLE_SHEET_NAME      = os.environ.get("GOOGLE_SHEET_NAME", "AgentPesa ASR Evaluation")
INTRON_ENDPOINT        = "https://infer.voice.intron.io/file/v1/upload/sync"

# ── Google Sheets client ──────────────────────────
def get_sheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds  = Credentials.from_service_account_file(
        GOOGLE_CREDENTIALS_PATH, scopes=scopes
    )
    client = gspread.authorize(creds)
    return client.open(GOOGLE_SHEET_NAME).sheet1

# ── Eager load LyngualLabs at startup ─────────────
print("Loading LyngualLabs model at startup...")
lynguallabs_model = WhisperModel(
    "./codeswitched",
    device="cpu",
    compute_type="int8",
    cpu_threads=2,
    num_workers=2
)
print("LyngualLabs model ready.")

# ── LyngualLabs transcription ─────────────────────
def transcribe_lynguallabs(audio_path):
    try:
        segments, _ = lynguallabs_model.transcribe(
            audio_path,
            language="yo",
            beam_size=5,
            vad_filter=True
        )
        return " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        return f"❌ LyngualLabs error: {str(e)}"

# ── Intron Sahara API transcription ───────────────
def transcribe_intron(audio_path, language_code):
    if not INTRON_API_KEY:
        return "⚠️ INTRON_API_KEY not set in .env file."
    try:
        filename = os.path.basename(audio_path)
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                INTRON_ENDPOINT,
                headers={"Authorization": f"Bearer {INTRON_API_KEY}"},
                data={
                    "audio_file_name":             filename,
                    "use_language_asr_input":      language_code,
                    "use_category":                "file_category_general",
                    "use_disable_llm_corrections": "TRUE"
                },
                files={"audio_file_blob": (filename, audio_file)},
                timeout=120
            )
        if response.status_code == 200:
            return response.json()["data"]["audio_transcript"]
        elif response.status_code == 400:
            return "❌ Audio too long — Intron supports max 120 seconds."
        elif response.status_code == 503:
            return "⏳ Intron timed out — try again."
        else:
            return f"❌ Intron API error {response.status_code}: {response.text}"
    except requests.exceptions.Timeout:
        return "❌ Request timed out after 120 seconds."
    except Exception as e:
        return f"❌ Intron error: {str(e)}"

# ── Translation via OpenAI ─────────────────────────
def translate_to_english(text, language_label):
    if not text or not text.strip():
        return "Nothing to translate."
    if text.startswith(("❌", "⚠️", "⏳")):
        return "Cannot translate — transcription failed."
    if not OPENAI_API_KEY:
        return "⚠️ OPENAI_API_KEY not set in .env file."
    try:
        client   = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a translation assistant for a Nigerian banking app. "
                        f"Translate the following {language_label} text into natural English. "
                        f"Keep names, bank names, account numbers, and amounts exactly as they are. "
                        f"Output only the English translation, nothing else."
                    )
                },
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Translation error: {str(e)}"

# ── Tab 1: Yoruba A/B Comparison ──────────────────
def compare_yoruba(audio_path):
    if audio_path is None:
        return (
            "Please record or upload audio.",
            "Please record or upload audio."
        )
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_lyngu  = executor.submit(transcribe_lynguallabs, audio_path)
        f_intron = executor.submit(transcribe_intron, audio_path, "yo")
        return f_lyngu.result(), f_intron.result()

def translate_comparison(lyngu_text, intron_text):
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_lyngu  = executor.submit(
            translate_to_english, lyngu_text,
            "Yoruba-English code-switched"
        )
        f_intron = executor.submit(
            translate_to_english, intron_text,
            "Yoruba-English code-switched"
        )
        return f_lyngu.result(), f_intron.result()

# ── Google Sheets submit ───────────────────────────
RATINGS = ["🟢 Excellent", "🔵 Good", "🟡 Fair", "🔴 Bad"]

def submit_evaluation(
    audio_path,
    lyngu_transcript,
    lyngu_translation,
    lyngu_rating,
    intron_transcript,
    intron_translation,
    intron_rating,
    notes
):
    # Guard — all required fields must be filled
    missing = []
    if not lyngu_transcript or lyngu_transcript.startswith(("❌","⚠️","⏳","Please")):
        missing.append("LyngualLabs transcript")
    if not intron_transcript or intron_transcript.startswith(("❌","⚠️","⏳","Please")):
        missing.append("Intron transcript")
    if not lyngu_rating:
        missing.append("LyngualLabs rating")
    if not intron_rating:
        missing.append("Intron rating")

    if missing:
        return f"⚠️ Please complete: {', '.join(missing)} before submitting."

    if not GOOGLE_CREDENTIALS_PATH:
        return "⚠️ GOOGLE_CREDENTIALS_PATH not set in .env file."

    try:
        sheet = get_sheet()
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),       # A: Timestamp
            os.path.basename(audio_path) if audio_path else "",  # B: Audio Filename
            lyngu_transcript,                                     # C: LyngualLabs Transcript
            lyngu_translation,                                    # D: LyngualLabs English
            lyngu_rating,                                         # E: LyngualLabs Rating
            intron_transcript,                                    # F: Intron Transcript
            intron_translation,                                   # G: Intron English
            intron_rating,                                        # H: Intron Rating
            notes or ""                                           # I: Tester Notes
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
        return "✅ Evaluation submitted successfully! Row added to Google Sheets."
    except FileNotFoundError:
        return "❌ Credentials file not found. Check GOOGLE_CREDENTIALS_PATH in .env."
    except gspread.exceptions.SpreadsheetNotFound:
        return f"❌ Sheet '{GOOGLE_SHEET_NAME}' not found. Check the sheet name and sharing."
    except Exception as e:
        return f"❌ Failed to submit: {str(e)}"

# ── Tab 2: Multilingual Intron API ────────────────
LANGUAGE_MAP = {
    "Yoruba-English": "yo",
    "Igbo-English":   "ig",
    "Hausa-English":  "ha"
}

def transcribe_multilingual(audio_path, language_choice):
    if audio_path is None:
        return "Please record or upload audio."
    return transcribe_intron(audio_path, LANGUAGE_MAP[language_choice])

def translate_multilingual(text, language_choice):
    return translate_to_english(text, language_choice)

# ── Gradio Interface ───────────────────────────────
with gr.Blocks(title="AgentPesa ASR Tester") as demo:

    gr.Markdown("""
    # 🎙️ AgentPesa ASR Tester
    **Internal use only — testers only**
    """)

    # ── Tab 1: Yoruba Comparison + Evaluation ─────
    with gr.Tab("🆚 Yoruba Model Comparison"):
        gr.Markdown("""
        ### LyngualLabs vs Intron Sahara v2.5
        Transcribe → Translate → Rate → Submit to Google Sheets.
        """)

        audio_compare = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Audio Input (max 120 seconds)"
        )

        # ── Stage 1: Transcribe ───────────────────
        compare_btn = gr.Button("▶ Transcribe Both Models", variant="primary")

        with gr.Row():
            lyngu_out = gr.Textbox(
                label="LyngualLabs Transcript",
                placeholder="LyngualLabs transcript appears here...",
                lines=4
            )
            intron_out = gr.Textbox(
                label="Intron Sahara v2.5 Transcript",
                placeholder="Intron transcript appears here...",
                lines=4
            )

        compare_btn.click(
            fn=compare_yoruba,
            inputs=[audio_compare],
            outputs=[lyngu_out, intron_out]
        )

        gr.Markdown("---")

        # ── Stage 2: Translate ────────────────────
        translate_btn = gr.Button("🌐 Translate Both to English", variant="secondary")

        with gr.Row():
            lyngu_translation = gr.Textbox(
                label="LyngualLabs → English",
                placeholder="English translation appears here...",
                lines=3
            )
            intron_translation = gr.Textbox(
                label="Intron Sahara → English",
                placeholder="English translation appears here...",
                lines=3
            )

        translate_btn.click(
            fn=translate_comparison,
            inputs=[lyngu_out, intron_out],
            outputs=[lyngu_translation, intron_translation]
        )

        gr.Markdown("---")

        # ── Stage 3: Rate and Submit ──────────────
        gr.Markdown("### ⭐ Rate Each Model")
        gr.Markdown(
            "🟢 Excellent — perfect transcription &nbsp;|&nbsp; "
            "🔵 Good — mostly correct, minor errors &nbsp;|&nbsp; "
            "🟡 Fair — understandable but significant errors &nbsp;|&nbsp; "
            "🔴 Bad — largely wrong, unusable"
        )

        with gr.Row():
            lyngu_rating = gr.Radio(
                choices=RATINGS,
                label="Rate LyngualLabs",
                value=None
            )
            intron_rating = gr.Radio(
                choices=RATINGS,
                label="Rate Intron Sahara",
                value=None
            )

        notes_box = gr.Textbox(
            label="Tester Notes (optional)",
            placeholder="Any observations about the transcriptions...",
            lines=2
        )

        submit_btn = gr.Button("📤 Submit Evaluation to Google Sheets", variant="primary")

        submit_status = gr.Textbox(
            label="Submission Status",
            interactive=False,
            lines=1
        )

        submit_btn.click(
            fn=submit_evaluation,
            inputs=[
                audio_compare,
                lyngu_out,
                lyngu_translation,
                lyngu_rating,
                intron_out,
                intron_translation,
                intron_rating,
                notes_box
            ],
            outputs=[submit_status]
        )

    # ── Tab 2: Multilingual ────────────────────────
    with gr.Tab("🌍 Multilingual (Intron API)"):
        gr.Markdown("""
        ### Hausa, Igbo, Yoruba — Code-Switched
        Intron Sahara v2.5 handles all three.
        """)

        language_choice = gr.Radio(
            choices=["Yoruba-English", "Igbo-English", "Hausa-English"],
            label="Select Language",
            value="Yoruba-English"
        )
        audio_multi = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            format="wav",
            label="Audio Input (max 120 seconds)"
        )
        transcribe_multi_btn = gr.Button("▶ Transcribe", variant="primary")
        transcript_multi = gr.Textbox(
            label="Transcript",
            placeholder="Transcript appears here...",
            lines=4
        )

        transcribe_multi_btn.click(
            fn=transcribe_multilingual,
            inputs=[audio_multi, language_choice],
            outputs=[transcript_multi]
        )

        gr.Markdown("---")
        translate_multi_btn = gr.Button(
            "🌐 Translate to English", variant="secondary"
        )
        translation_multi = gr.Textbox(
            label="English Translation",
            placeholder="English translation appears here...",
            lines=3
        )

        translate_multi_btn.click(
            fn=translate_multilingual,
            inputs=[transcript_multi, language_choice],
            outputs=[translation_multi]
        )

    # ── Tab 3: Type Text ───────────────────────────
    with gr.Tab("⌨️ Type Text"):
        gr.Markdown(
            "Type text directly and translate to English — no audio needed."
        )
        typed_language = gr.Radio(
            choices=["Yoruba-English", "Igbo-English", "Hausa-English"],
            label="Language of typed text",
            value="Yoruba-English"
        )
        typed_input = gr.Textbox(
            label="Type text here",
            placeholder="Type Yoruba, Igbo, or Hausa code-switched text...",
            lines=4
        )
        translate_typed_btn = gr.Button(
            "🌐 Translate to English", variant="primary"
        )
        typed_translation = gr.Textbox(
            label="English Translation",
            placeholder="English translation appears here...",
            lines=3
        )

        translate_typed_btn.click(
            fn=translate_multilingual,
            inputs=[typed_input, typed_language],
            outputs=[typed_translation]
        )

# ── Launch ─────────────────────────────────────────
# Local:
#demo.launch()

# DigitalOcean:
demo.launch(server_name="0.0.0.0", server_port=7860)