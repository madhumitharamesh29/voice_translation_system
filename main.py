import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request

# Load the Whisper model and processor
model_name = "openai/whisper-large"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize FastAPI
app = FastAPI()

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Function for audio translation
def translate_audio(audio):
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    audio_input = inputs['input_features'].to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="translate")
    
    with torch.no_grad():
        generated_ids = model.generate(audio_input, forced_decoder_ids=forced_decoder_ids)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Serve the HTML form and display translated text
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "translated_text": ""})

# Handle file upload and return translated text
@app.post("/uploadfile/", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    audio, _ = librosa.load(file.file, sr=16000)  # Convert the audio file to 16kHz for Whisper
    translated_text = translate_audio(audio)
    return templates.TemplateResponse("index.html", {"request": request, "translated_text": translated_text})

# Mount the static directory (optional if using external files like images)
app.mount("/static", StaticFiles(directory="static"), name="static")