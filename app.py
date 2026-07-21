import gradio as gr
import torch
from transformers import pipeline
import os
import shutil
from openai import OpenAI
from dotenv import load_dotenv

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

HF_TOKEN = os.environ.get("HF_TOKEN", "")  # needed for NCAIR (gated model)

import gc

# Global storage for the currently loaded model to optimize RAM (lazy loading)
current_pipeline = {"name": None, "pipe": None}

def load_asr_pipeline(model_choice):
    global current_pipeline
    
    model_mapping = {
        "NCAIR Yoruba": ("NCAIR1/Yoruba-ASR", HF_TOKEN),
        "NCAIR Igbo": ("NCAIR1/Igbo-ASR", HF_TOKEN),
        "NCAIR Hausa": ("NCAIR1/Hausa-ASR", HF_TOKEN),
        "LyngualLabs Yoruba (Code-Switched)": ("LyngualLabs/whisper-small-yoruba", None)
    }
    
    if model_choice not in model_mapping:
        raise ValueError(f"Unknown model choice: {model_choice}")
        
    model_id, token = model_mapping[model_choice]
    
    # If the requested model is already loaded, reuse it
    if current_pipeline["name"] == model_choice:
        return current_pipeline["pipe"]
        
    # Free memory of the previously loaded model to stay within droplet RAM limits
    if current_pipeline["pipe"] is not None:
        print(f"Unloading previous model ({current_pipeline['name']}) to free memory...")
        del current_pipeline["pipe"]
        current_pipeline["pipe"] = None
        current_pipeline["name"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"Loading {model_choice} model ({model_id})...")
    
    # Dynamically select device and precision (CPU does not support float16 for many operations)
    if torch.cuda.is_available():
        device = 0
        torch_dtype = torch.float16
    else:
        device = -1
        torch_dtype = torch.float32
        
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        token=token,
        torch_dtype=torch_dtype,
        device=device
    )
    
    current_pipeline["name"] = model_choice
    current_pipeline["pipe"] = pipe
    print(f"Model {model_choice} loaded successfully!")
    return pipe

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

def transcribe_audio(audio_path, model_choice):
    if audio_path is None:
        return "Please record or upload audio first."
    
    try:
        pipe = load_asr_pipeline(model_choice)
        result = pipe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error during transcription: {str(e)}"

def translate_text(text, model_choice):
    if not text.strip():
        return "Nothing to translate yet."
    if not OPENAI_KEY:
        return "⚠️ No OpenAI key set yet. Add one to enable translation (see Step 6 below)."

    client = OpenAI(api_key=OPENAI_KEY)
    
    if model_choice == "NCAIR Yoruba":
        lang = "Yoruba"
    elif model_choice == "NCAIR Igbo":
        lang = "Igbo"
    elif model_choice == "NCAIR Hausa":
        lang = "Hausa"
    else:
        lang = "Yoruba-English code-switched"

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

#demo.launch(server_name="0.0.0.0", server_port=7860)
demo.launch()