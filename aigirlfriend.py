from __future__ import annotations

import os
import re
import wave
import tempfile
import warnings
import queue
from threading import Thread
from typing import List, Optional

import numpy as np
import pyaudio
import sounddevice as sd
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Konstanter och konfiguration
# ---------------------------------------------------------------------------
USER_NAME = "Love"
MEMORY_FILE = "user_memory.txt"
TTS_MODEL = "tts-1"
TTS_VOICE = "nova"
CHAT_MODEL = "gpt-4.1-mini"

# ---------------------------------------------------------------------------
# Användarminne
# ---------------------------------------------------------------------------
def load_user_notes() -> List[str]:
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def append_user_note(note: str) -> None:
    if note and note not in load_user_notes():
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(note + "\n")

def extract_note_from_input(text: str) -> Optional[str]:
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
            if not allow_long or len(note) < 80:
                return note
    return None

# ---------------------------------------------------------------------------
# OpenAI-klienter
# ---------------------------------------------------------------------------
client = OpenAI()

# ---------------------------------------------------------------------------
# Systemprompt och minne
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

# Förvärm GPT och TTS
_ = client.chat.completions.create(
    model=CHAT_MODEL,
    messages=[
        {"role": "system", "content": "You are Nova, a helpful AI girlfriend."},
        {"role": "user", "content": "Hi!"}
    ]
)
_ = client.audio.speech.create(
    model=TTS_MODEL,
    voice=TTS_VOICE,
    input="Hello!",
    response_format="pcm"
)

# ---------------------------------------------------------------------------
# Ljudfunktioner
# ---------------------------------------------------------------------------
def play_audio_stream(audio_stream):
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
    print("🎙️ Lyssnar…")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())
    print("🆗 Fångat!")
    return temp.name

# ---------------------------------------------------------------------------
# GPT-streaming och TTS med threading
# ---------------------------------------------------------------------------
def stream_gpt_response_and_play() -> str:
    # Konvertera langchain-meddelanden till OpenAI-format
    messages = []
    for msg in memory.chat_memory.messages:
        if isinstance(msg, SystemMessage):
            messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True
    )

    buffer = ""
    full_text = ""
    audio_queue = queue.Queue()

    def play_audio_worker():
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            play_audio_stream(audio)

    player_thread = Thread(target=play_audio_worker)
    player_thread.start()

    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            buffer += delta
            full_text += delta
            if delta.endswith(('.', '!', '?')) or len(buffer) > 100:
                print(f"🗣️ Nova (stream): {buffer.strip()}")
                audio_response = client.audio.speech.create(
                    model=TTS_MODEL,
                    voice=TTS_VOICE,
                    input=buffer.strip(),
                    response_format="pcm",
                    speed=1.1  # faster speech
                )
                audio_queue.put(audio_response)
                buffer = ""

    if buffer.strip():
        print(f"🗣️ Nova (stream): {buffer.strip()}")
        audio_response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=buffer.strip(),
            response_format="pcm",
            speed=1.1
        )
        audio_queue.put(audio_response)

    audio_queue.put(None)
    player_thread.join()
    return full_text

# ---------------------------------------------------------------------------
# Huvudfunktion
# ---------------------------------------------------------------------------
def process_audio() -> None:
    wav_path = record_audio_tempfile()

    with open(wav_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    user_input = transcription.text.strip()
    print(f">>> Du: {user_input}")

    note = extract_note_from_input(user_input)
    if note and note not in user_notes:
        append_user_note(note)
        user_notes.append(note)
        memory.chat_memory.add_message(SystemMessage(content=note))
        print(f"[🧠 Minnesanteckning] Sparat: {note}")

    memory.chat_memory.add_message(HumanMessage(content=user_input))
    print("🧠 Nova tänker...")

    full_response = stream_gpt_response_and_play()
    memory.chat_memory.add_message(AIMessage(content=full_response))

    os.remove(wav_path)

# ---------------------------------------------------------------------------
# Kör programmet
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    while True:
        th = Thread(target=process_audio)
        th.start()
        th.join()
