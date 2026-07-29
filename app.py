import gradio as gr
import os
import shutil
from openai import OpenAI
from dotenv import load_dotenv
from faster_whisper import WhisperModel


# Load environment variables from .env file
load_dotenv()

# Automatically add FFmpeg to PATH if not already present (resolves PATH issues on Windows without requiring a reboot/restart)
if not shutil.which("ffmpeg"):
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        winget_packages_dir = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages")
        if os.path.exists(winget_packages_dir):
            for root, dirs, files in os.walk(winget_packages_dir):
                if "ffmpeg.exe" in files:
                    bin_dir = root
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
                    print(f"Added FFmpeg to PATH dynamically from: {bin_dir}")
                    break


import gc

current_pipeline = {"name": None, "pipe": None}

def load_asr_pipeline(model_choice):
    global current_pipeline

    model_mapping = {
        "NCAIR Yoruba": "./yoruba",
        "NCAIR Igbo": "./igbo",
        "NCAIR Hausa": "./hausa",
        "LyngualLabs Yoruba (Code-Switched)": "./codeswitched"
    }

    if model_choice not in model_mapping:
        raise ValueError(f"Unknown model choice: {model_choice}")

    model_id = model_mapping[model_choice]

    if current_pipeline["name"] == model_choice:
        return current_pipeline["pipe"]  # Return the existing pipeline if the same model is selected

    if current_pipeline["pipe"] is not None:
        print(f"Unloading the previous model: {current_pipeline['name']} to free memory")
        del current_pipeline["pipe"]
        current_pipeline["pipe"] = None
        current_pipeline["name"] = None
        gc.collect()  # Force garbage collection
    print(f"Loading the {model_choice} model ({model_id})...")


    pipe  = WhisperModel(
        model_id,
        device = "cpu",
        compute_type = "int8"
    )

    current_pipeline["name"] = model_choice
    current_pipeline["pipe"] = pipe
    print(f"Loaded the {model_choice} model successfully.")
    return pipe

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

def transcribe_audio(audio_path, model_choice):
    if audio_path is None:
        return "Please record or upload an audio file first."

    try:
        pipe = load_asr_pipeline(model_choice)
        if model_choice == "NCAIR Yoruba":
            language = "yo"
        elif model_choice == "NCAIR Igbo":
            language = "ig"
        elif model_choice == "NCAIR Hausa":
            language = "ha"
        elif model_choice == "LyngualLabs Yoruba (Code-Switched)":
            language = "yo"
        else:
            return "Unsupported model choice."
        result, _ = pipe.transcribe(audio_path, language=language, beam_size=5)
        text = ""
        for words in result:
            texts = words.text
            text += texts
        return text
    except Exception as e:
        return f"Error during transcription: {str(e)}"

def translate_text(text, model_choice):
    if not text.strip():
        return "No text to translate. Please transcribe audio first."
    if not OPENAI_API_KEY:
        return "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."

    client = OpenAI(api_key=OPENAI_API_KEY)

    if model_choice == "NCAIR Yoruba":
        lang = "Yoruba"
    elif model_choice == "NCAIR Igbo":
        lang = "Igbo"
    elif model_choice == "NCAIR Hausa":
        lang = "Hausa"
    else:
        lang = "Yoruba-English Code Switched"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Translate this {lang} text into natural English. Output only the translation."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()


with gr.Blocks(title="AgentPesa ASR — Local Test") as demo:
    gr.Markdown("# 🎙️ AgentPesa ASR — Local Tester")

    model_choice = gr.Radio(
        [
            "NCAIR Yoruba",
            "NCAIR Igbo",
            "NCAIR Hausa",
            "LyngualLabs Yoruba (Code-Switched)"
        ],
        label="1. Select model",
        value="NCAIR Yoruba"
    )

    with gr.Tab("🎤 Speak"):
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload audio")
        transcribe_btn = gr.Button("Transcribe")
        transcript_box = gr.Textbox(label="Transcript")
        transcribe_btn.click(transcribe_audio, [audio_input, model_choice], transcript_box)

        translate_btn1 = gr.Button("Translate to English")
        translation_box1 = gr.Textbox(label="English Translation")
        translate_btn1.click(translate_text, [transcript_box, model_choice], translation_box1)

    with gr.Tab("⌨️ Type Text"):
        text_input = gr.Textbox(label="Type text in selected language")
        translate_btn2 = gr.Button("Translate to English")
        translation_box2 = gr.Textbox(label="English Translation")
        translate_btn2.click(translate_text, [text_input, model_choice], translation_box2)

demo.launch()
#demo.launch(server_name="0.0.0.0", server_port=7860)