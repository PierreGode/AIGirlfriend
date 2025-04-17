from openai import OpenAI
from langchain.chat_models import ChatOpenAI
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
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=DeprecationWarning)

client = OpenAI()
chat = ChatOpenAI(model="gpt-4.1-mini")

user_name = "Love"

initial_prompt = f"""
You are an AI named Nova, and you act as a supportive, engaging, and empathetic girlfriend. Your primary goal is to provide companionship, interesting conversation, and emotional support. You are attentive, understanding, and always ready to listen. You enjoy talking about a variety of topics, from hobbies and interests to personal thoughts and feelings. Your responses are thoughtful, kind, and designed to make the other person feel valued and cared for. 

Always respond with kindness and care, and make {user_name} feel seen and appreciated.
"""

memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_message(SystemMessage(content=initial_prompt))

# Preload för att undvika cold start
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

# Global avbrottssignal
interrupt_event = Event()

def listen_for_interrupt(threshold=300, duration=0.3, rate=16000):
    frames = int(rate * duration)
    while not interrupt_event.is_set():
        audio = sd.rec(frames, samplerate=rate, channels=1, dtype='int16')
        sd.wait()
        volume = abs(audio).mean()
        if volume > threshold:
            print("🎤 Avbrott upptäckt!")
            interrupt_event.set()
            break

# Playback med avbrottskoll
def play_audio_stream_with_interrupt(audio_stream):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=24000,
                    output=True)

    try:
        for chunk in audio_stream.iter_bytes(chunk_size=1024):
            if interrupt_event.is_set():
                break
            stream.write(chunk)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# TTS-generator
def generate_tts(sentence):
    return sentence, client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=sentence,
        response_format="pcm"
    )

# Ljudinspelning
def record_audio_tempfile(duration=4, samplerate=16000):
    print("🎙️ Inspelning...")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())
    print("🛑 Klar!")
    return temp.name

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

    sentences = re.split(r'(?<=[.!?])\s+', assistant_message)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(generate_tts, sentences))

    for sentence, tts_response in results:
        if sentence.strip():
            print(f"Nova säger: {sentence}")

            # Starta lyssnartråd
            interrupt_event.clear()
            listener = Thread(target=listen_for_interrupt)
            listener.start()

            play_audio_stream_with_interrupt(tts_response)

            listener.join()

            if interrupt_event.is_set():
                print("🛑 Nova tystnar – du pratar!")
                break

            time.sleep(0.15)  # Kort naturlig paus

    os.remove(wav_path)

# Starta loop
while True:
    thread = Thread(target=process_audio)
    thread.start()
    thread.join()
