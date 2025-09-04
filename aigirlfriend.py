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

USER_NAME = "Love"
MEMORY_FILE = "user_memory.txt"
TTS_MODEL = "tts-1"
TTS_VOICE = "nova"
CHAT_MODEL = "gpt-5-nano"

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
Only say "thank you" if the user expresses gratitude first, and be precise about what the user actually says.
"""

user_notes = load_user_notes()
notes_prompt = (
    "The following facts are known about the user: " + "; ".join(user_notes)
    if user_notes else "No prior facts about the user are known yet."
)

memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_message(SystemMessage(content=BASE_PROMPT))
memory.chat_memory.add_message(SystemMessage(content=notes_prompt))

# Preload models
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

def play_audio_stream(stream, audio_stream):
    """Write an audio stream to an already-open PyAudio stream."""
    # Send a brief burst of silence before each sentence so the first word
    # isn't clipped when playback resumes.
    stream.write(b"\x00" * 2400)  # ~50ms at 24 kHz
    for chunk in audio_stream.iter_bytes(chunk_size=1024):
        stream.write(chunk)

def record_audio_tempfile_vad(
    samplerate: int = 16000, db_threshold: float = 50.0, max_duration: int = 20
) -> Optional[str]:
    """Record audio until the user stops speaking.

    The function waits indefinitely for speech above the dB threshold and only
    returns after a short silence following the detected speech. This prevents
    the system from repeatedly printing "No speech detected" while passively
    listening.
    """

    print("🎙️ Listening…")
    frames: list[np.ndarray] = []
    block_duration = 0.2
    block_size = int(samplerate * block_duration)
    silence_duration = 0.0
    speech_duration = 0.0
    has_speech = False

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="int16") as stream:
        while True:
            block, _ = stream.read(block_size)
            rms = np.sqrt(np.mean(block.astype(np.float32) ** 2))
            volume_db = 20 * np.log10(rms / 32768 + 1e-10) + 100

            if volume_db >= db_threshold:
                has_speech = True
                silence_duration = 0.0
                speech_duration += block_duration
                frames.append(block)
                if speech_duration >= max_duration:
                    break
            elif has_speech:
                silence_duration += block_duration
                frames.append(block)
                if silence_duration > 1.0:
                    break

    if not has_speech:
        return None

    print("🆗 Got you")
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(b"".join([b.tobytes() for b in frames]))

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

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True
    )

    audio_queue = queue.Queue()

    def play_audio_worker():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=24000,
                        output=True)
        # Prime the device once so it is ready for immediate playback.
        stream.write(b"\x00" * 4800)
        while True:
            audio = audio_queue.get()
            if audio is None:
                break
            play_audio_stream(stream, audio)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def tts_worker(sentence_queue: queue.Queue):
        while True:
            sentence = sentence_queue.get()
            if sentence is None:
                break
            print(f"🗣️ Nova (buffered): {sentence}")
            audio = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=sentence,
                response_format="pcm",
                speed=1.0
            )
            audio_queue.put(audio)
        audio_queue.put(None)

    player_thread = Thread(target=play_audio_worker)
    sentence_queue: queue.Queue[str | None] = queue.Queue()
    tts_thread = Thread(target=tts_worker, args=(sentence_queue,))
    player_thread.start()
    tts_thread.start()

    full_text = ""
    buffer = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if not delta:
            continue
        full_text += delta
        buffer += delta
        while True:
            match = re.search(r'([^.!?]+[.!?]"?)', buffer)
            if not match:
                break
            sentence = match.group(1).strip()
            sentence_queue.put(sentence)
            buffer = buffer[match.end():]

    if buffer.strip():
        sentence_queue.put(buffer.strip())

    sentence_queue.put(None)
    tts_thread.join()
    player_thread.join()

    print(f"🧠 Nova (full): {full_text}")
    return full_text

def process_audio() -> None:
    wav_path = record_audio_tempfile_vad()
    if not wav_path:
        return

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
