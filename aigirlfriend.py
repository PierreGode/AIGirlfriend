from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import warnings
import os
import time
import uuid
import platform
import pyaudio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile
from io import BytesIO
from threading import Thread

warnings.filterwarnings("ignore", category=DeprecationWarning)

# OpenAI-klienter
client = OpenAI()
chat = ChatOpenAI(model="gpt-4.1-mini")

# Användarnamn
user_name = "Love"

# Initialt systemmeddelande
initial_prompt = f"""
You are an AI named Nova, and you act as a supportive, engaging, and empathetic girlfriend. Your primary goal is to provide companionship, interesting conversation, and emotional support. You are attentive, understanding, and always ready to listen. You enjoy talking about a variety of topics, from hobbies and interests to personal thoughts and feelings. Your responses are thoughtful, kind, and designed to make the other person feel valued and cared for. 

Always respond with kindness and care, and make {user_name} feel seen and appreciated.
"""

# Skapa minne och initiera det med systemprompt
memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_message(SystemMessage(content=initial_prompt))

# Preload GPT och TTS för att minska cold start
_ = chat.predict_messages(messages=[
    SystemMessage(content="You are Nova, a helpful AI."),
    HumanMessage(content="Hi!")
])
_ = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="Hello!",
    response_format="pcm"
)

# Direktuppspelning av ljud från OpenAI:s TTS (PCM)
def play_audio_stream(audio_stream):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,  # 16-bit PCM
                    channels=1,
                    rate=24000,  # OpenAI TTS använder 24kHz
                    output=True)

    for chunk in audio_stream.iter_bytes(chunk_size=1024):
        stream.write(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()

# Spela in ljud direkt till RAM som WAV-format
def record_audio_bytes(duration=4, samplerate=16000):
    print("🎙️ Inspelning...")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    buffer = BytesIO()
    scipy.io.wavfile.write(buffer, samplerate, recording)
    buffer.seek(0)
    print("🛑 Klar!")
    return buffer

# Huvudlogik för röstinspelning, transkribering och AI-svar
def process_audio():
    audio_bytes = record_audio_bytes()
    transcription = client.audio.transcriptions.create(
        model='whisper-1',
        file=audio_bytes
    )
    user_input = transcription.text
    print(f">>> Du: {user_input}")

    memory.chat_memory.add_message(HumanMessage(content=user_input))
    response = chat.predict_messages(messages=memory.load_memory_variables({})["history"])
    assistant_message = response.content
    memory.chat_memory.add_message(AIMessage(content=assistant_message))

    print(f"Nova: {assistant_message}")

    speech_response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=assistant_message,
        response_format="pcm"  # RAW 16-bit PCM
    )

    play_audio_stream(speech_response)

# Startar samtalsloopen
while True:
    thread = Thread(target=process_audio)
    thread.start()
    thread.join()
