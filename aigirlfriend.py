from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from utils import record_audio, play_audio
import warnings
import os
import time
import pygame
import uuid
import platform
from threading import Thread

warnings.filterwarnings("ignore", category=DeprecationWarning)

# OpenAI-klienter
client = OpenAI()
chat = ChatOpenAI(model="gpt-4o")

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

# Plattformsspecifik ljuduppspelning
def play_audio_with_pygame(file_path):
    pygame.mixer.init()
    time.sleep(0.5)
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.set_volume(1.0)
    time.sleep(0.5)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    pygame.mixer.quit()

def play_audio_with_alsa(file_path):
    try:
        import alsaaudio
        import wave

        wf = wave.open(file_path, 'rb')
        device = alsaaudio.PCM(alsaaudio.PCM_PLAYBACK)
        device.setchannels(wf.getnchannels())
        device.setrate(wf.getframerate())
        device.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        device.setperiodsize(320)
        data = wf.readframes(320)
        audio_data = []
        while data:
            audio_data.append(data)
            data = wf.readframes(320)
        time.sleep(0.5)
        for chunk in audio_data:
            device.write(chunk)
        wf.close()
    except Exception as e:
        print(f"Error playing audio with ALSA: {e}")

is_windows = platform.system() == "Windows"

# Huvudlogik för röstinspelning, transkribering och AI-svar
def process_audio():
    record_audio('test.wav')
    audio_file = open('test.wav', "rb")
    transcription = client.audio.transcriptions.create(
        model='whisper-1',
        file=audio_file
    )
    user_input = transcription.text
    print(user_input)

    memory.chat_memory.add_message(HumanMessage(content=user_input))
    response = chat.predict_messages(messages=memory.load_memory_variables({})["history"])
    assistant_message = response.content
    memory.chat_memory.add_message(AIMessage(content=assistant_message))

    print(assistant_message)

    speech_response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=assistant_message
    )
    speech_filename = f"speech_{uuid.uuid4()}.mp3"
    speech_response.stream_to_file(speech_filename)

    if is_windows:
        play_audio_with_pygame(speech_filename)
    else:
        play_audio_with_alsa(speech_filename)

    audio_file.close()
    os.remove(speech_filename)

# Startar samtalsloopen
while True:
    thread = Thread(target=process_audio)
    thread.start()
    thread.join()
