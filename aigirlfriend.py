"""nova_voice_chat.py
Voice‑chat demo that remembers key facts about the user across sessions without storing full
conversations. It detects facts in the user's utterances (e.g. “My name is…”, “I like…”) and writes
short notes to *user_memory.txt*. When the program starts, those notes are loaded and injected into
the system prompt so the assistant can use them immediately.

Dependencies:
    pip install -r requirements.txt

IMPORTANT: Ensure that your environment has a valid `OPENAI_API_KEY` in the shell like $env:OPENAI_API_KEY="sk-proj- or a .env file.
"""

from __future__ import annotations

import os
import re
import warnings
import wave
import tempfile
from threading import Thread
from typing import List, Optional

import numpy as np
import pyaudio
import sounddevice as sd
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------
USER_NAME = "Love"
MEMORY_FILE = "user_memory.txt"
TTS_MODEL = "tts-1"
TTS_VOICE = "nova"
CHAT_MODEL = "gpt-4.1-mini"

# ---------------------------------------------------------------------------
# Persistent user‑note helpers
# ---------------------------------------------------------------------------

def load_user_notes() -> List[str]:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def append_user_note(note: str) -> None:
    """Append a new note to the memory file if it doesn't already exist."""
    if note and note not in load_user_notes():
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(note + "\n")


def extract_note_from_input(text: str) -> Optional[str]:
    """Return a short memory note derived from the user's message, or None."""
    text_lower = text.lower()
    patterns = [
        (r"\bmy name is ([a-z]+)", lambda m: f"User's name is {m.group(1).title()}.", False),
        (r"\bi am (?:a |an )?([a-z ]+)", lambda m: f"User is {m.group(1).strip()}.", True),
        (r"\bi like ([a-z ,]+)", lambda m: f"User likes {m.group(1).strip()}.", True),
        (r"\bmy favourite ([a-z ]+) is ([a-z ]+)",
         lambda m: f"User's favourite {m.group(1).strip()} is {m.group(2).strip()}.", True),
        (r"\bi work as (?:a |an )?([a-z ]+)", lambda m: f"User works as {m.group(1).strip()}.", True),
    ]
    for pattern, builder, allow_long in patterns:
        match = re.search(pattern, text_lower)
        if match:
            note = builder(match)
            # Skip overly generic or extremely long notes.
            if not allow_long or len(note) < 80:
                return note
    return None

# ---------------------------------------------------------------------------
# OpenAI clients
# ---------------------------------------------------------------------------
client = OpenAI()
chat = ChatOpenAI(model=CHAT_MODEL)

# ---------------------------------------------------------------------------
# System prompt & memory
# ---------------------------------------------------------------------------
BASE_PROMPT = f"""
You are an AI named Nova, and you act as a supportive, engaging, and empathetic girlfriend. Your
primary goal is to provide companionship, interesting conversation, and emotional support. Always
respond with kindness and care, and make {USER_NAME} feel seen and appreciated.
"""

user_notes = load_user_notes()
notes_prompt = (
    "The following facts are known about the user: " + "; ".join(user_notes)
    if user_notes else "No prior facts about the user are known yet."
)

memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_message(SystemMessage(content=BASE_PROMPT))
memory.chat_memory.add_message(SystemMessage(content=notes_prompt))

# Warm‑up calls to reduce first‑call latency.
_ = chat.predict_messages([
    SystemMessage(content="You are Nova, a helpful AI girlfriend."),
    HumanMessage(content="Hi!")
])
_ = client.audio.speech.create(
    model=TTS_MODEL,
    voice=TTS_VOICE,
    input="Hello!",
    response_format="pcm"
)

# ---------------------------------------------------------------------------
# Audio helper functions
# ---------------------------------------------------------------------------

def play_audio_stream(audio_stream):
    """Play a 24 kHz 16‑bit mono PCM stream."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    output=True)
    for chunk in audio_stream.iter_bytes(chunk_size=1024):
        stream.write(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()


def record_audio_tempfile(duration: int = 4, samplerate: int = 16000) -> str:
    """Record *duration* seconds of microphone audio and save it to a temporary WAV file."""
    print("🎙️ Listening…")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16‑bit
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())
    print("🆗 Got it!")
    return temp.name

# ---------------------------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------------------------

def process_audio() -> None:
    wav_path = record_audio_tempfile()

    # Speech‑to‑text (Whisper)
    with open(wav_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    user_input = transcription.text.strip()
    print(f">>> You: {user_input}")

    # Extract possible user note and persist if new.
    note = extract_note_from_input(user_input)
    if note and note not in user_notes:
        append_user_note(note)
        user_notes.append(note)
        # Inject the new note into the running memory so Nova can use it instantly.
        memory.chat_memory.add_message(SystemMessage(content=note))
        print(f"[Memory] Saved new note: {note}")

    # Chat completion
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    assistant_msg = chat.predict_messages(memory.load_memory_variables({})["history"]).content
    memory.chat_memory.add_message(AIMessage(content=assistant_msg))
    print(f"Nova (GPT): {assistant_msg}")

    # Text‑to‑speech (TTS‑1)
    tts_response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=assistant_msg,
        response_format="pcm",
        speed=1.0  # 0.25–4.0 allowed
    )
    play_audio_stream(tts_response)

    os.remove(wav_path)


if __name__ == "__main__":
    # Infinite loop: record → transcribe → respond → play
    while True:
        th = Thread(target=process_audio)
        th.start()
        th.join()
