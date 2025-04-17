from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import warnings
import os
import pyaudio
import sounddevice as sd
import wave
import tempfile
import re
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

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
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    output=True)

    for chunk in audio_stream.iter_bytes(chunk_size=1024):
        stream.write(chunk)

    stream.stop_stream()
    stream.close()
    p.terminate()

# Spela in ljud till en temporär WAV-fil
def record_audio_tempfile(duration=4, samplerate=16000):
    print("🎙️ Inspelning...")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())

    print("🛑 Klar!")
    return temp.name

# Hjälpfunktion för TTS av en mening
def generate_tts(sentence):
    return sentence, client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=sentence,
        response_format="pcm"
    )

# Huvudlogik
def process_audio():
    wav_path = record_audio_tempfile()

    with open(wav_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model='whisper-1',
            file=audio_file
        )

    user_input = transcription.text
    print(f">>> Du: {user_input}")

    memory.chat_memory.add_message(HumanMessage(content=user_input))
    response = chat.predict_messages(messages=memory.load_memory_variables({})["history"])
    assistant_message = response.content
    memory.chat_memory.add_message(AIMessage(content=assistant_message))

    print(f"Nova (GPT): {assistant_message}")

    # Dela upp i meningar
    sentences = re.split(r'(?<=[.!?])\s+', assistant_message)

    # Generera TTS parallellt
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(generate_tts, sentences))

    # Spela upp i rätt ordning med kort paus
    for sentence, tts_response in results:
        if sentence.strip():
            print(f"Nova säger: {sentence}")
            play_audio_stream(tts_response)
            time.sleep(0.05)

    os.remove(wav_path)

# Startar samtalsloopen
while True:
    thread = Thread(target=process_audio)
    thread.start()
    thread.join()
