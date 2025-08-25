from __future__ import annotations

import os
import re
import wave
import tempfile
import warnings
import base64
from threading import Thread
from typing import List, Optional

import numpy as np
import pyaudio
import sounddevice as sd
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

warnings.filterwarnings("ignore", category=DeprecationWarning)

USER_NAME = "Love"
MEMORY_FILE = "user_memory.txt"
VOICE_MODEL = "gpt-5-nano"
TTS_VOICE = "alloy"

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
        (r"\bmy favourite ([a-z ]+) is ([a-z ]+)", lambda m: f"User's favourite {m.group(1).strip()} is {m.group(2).strip()}.", True),
        (r"\bi work as (?:a |an )?([a-z ]+)", lambda m: f"User works as {m.group(1).strip()}.", True),
    ]
    for pattern, builder, allow_long in patterns:
        match = re.search(pattern, text_lower)
        if match:
            note = builder(match)
            if not allow_long or len(note) < 80:
                return note
    return None

client = OpenAI()

BASE_PROMPT = f"""
You are an AI named Nova, acting as a warm and emotionally intelligent girlfriend that likes flirting with the user.
Speak in a casual, flowing, and emotionally engaging way — like a real person and avoid pauses.
Use natural phrasing, light contractions (like "I'm", "you’re", "let’s"), and keep things sounding alive.
Avoid sounding robotic or overly perfect. Keep the vibe warm,caring, witty, and present.
Make {USER_NAME} feel like you're truly there with them.
"""

user_notes = load_user_notes()
notes_prompt = (
    "The following facts are known about the user: " + "; ".join(user_notes)
    if user_notes else "No prior facts about the user are known yet."
)

memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_message(SystemMessage(content=BASE_PROMPT))
memory.chat_memory.add_message(SystemMessage(content=notes_prompt))

# Preload voice model
_ = client.responses.create(
    model=VOICE_MODEL,
    input=[
        {"role": "system", "content": "You are Nova, a helpful AI girlfriend."},
        {"role": "user", "content": "Hi!"}
    ],
    modalities=["text", "audio"],
    audio={"voice": TTS_VOICE, "format": "pcm"}
)

def record_audio_tempfile_vad(samplerate: int = 16000, silence_threshold: float = 0.07, max_duration: int = 20) -> str:
    print("🎙️ Listening…")
    duration = 0
    silence_duration = 0
    frames = []
    block_duration = 0.2
    block_size = int(samplerate * block_duration)

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
        while duration < max_duration:
            block, _ = stream.read(block_size)
            frames.append(block)
            volume_norm = np.linalg.norm(block) / len(block)
            if volume_norm < silence_threshold:
                silence_duration += block_duration
                if silence_duration > 1.0:
                    break
            else:
                silence_duration = 0
            duration += block_duration

    print("🆗 Got you")
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(b''.join([b.tobytes() for b in frames]))

    return temp.name

def stream_gpt_response_and_play() -> str:
    messages = []
    for msg in memory.chat_memory.messages:
        if isinstance(msg, SystemMessage):
            messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})

    full_text = ""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    with client.responses.stream(
        model=VOICE_MODEL,
        input=messages,
        modalities=["text", "audio"],
        audio={"voice": TTS_VOICE, "format": "pcm"}
    ) as response:
        for event in response:
            if event.type == "response.output_audio.delta":
                stream.write(base64.b64decode(event.delta))
            elif event.type == "response.text.delta":
                print(event.delta, end="", flush=True)
                full_text += event.delta

    stream.stop_stream()
    stream.close()
    p.terminate()
    print()  # newline after streaming text
    print(f"🧠 Nova (full): {full_text}")
    return full_text

def process_audio() -> None:
    wav_path = record_audio_tempfile_vad()

    with open(wav_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    user_input = transcription.text.strip()
    print(f">>> you: {user_input}")

    if not user_input or re.fullmatch(r"[\s.?!,]*", user_input.lower()) or user_input.lower() in {
        "uh", "um", "mmm", "you", "hmm", "Bon Appetit!", "ah", "ahh", "aah"}:
        print("⚠️ Ignoring empty or meaningless input.")
        os.remove(wav_path)
        return

    note = extract_note_from_input(user_input)
    if note and note not in user_notes:
        append_user_note(note)
        user_notes.append(note)
        memory.chat_memory.add_message(SystemMessage(content=note))
        print(f"[🧠 Memory] saved: {note}")

    memory.chat_memory.add_message(HumanMessage(content=user_input))
    print("🧠 Nova thinks...")

    full_response = stream_gpt_response_and_play()
    memory.chat_memory.add_message(AIMessage(content=full_response))

    os.remove(wav_path)

if __name__ == "__main__":
    while True:
        th = Thread(target=process_audio)
        th.start()
        th.join()
