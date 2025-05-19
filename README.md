# 🎙️ Voice Translation System

A Python-based voice translation tool that uses a Hugging Face model to transcribe and translate audio files into English. 
It’s built to handle voice recordings in multiple languages and output English translations.

## ✨ Features

- 🗣️ Accepts audio files (e.g., WAV, MP3)
- 🌐 Translates non-English speech into English text
- 🤖 Powered by Hugging Face speech-to-text model.
- 🧾 Outputs clean, readable transcripts
- 🖥️ Simple web-based usage

## 🛠️ Tech Stack

- Python 
- Transformers 
- Flask (optional web interface)

## 🚀 How It Works

1. User uploads an audio file (any language).
2. The system uses the whisper multilingual speech recognition model 
   - Transcribe
   - Translate
3. English text is returned as the output.
