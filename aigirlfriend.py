from openai import OpenAI
from utils import record_audio, play_audio
import warnings
import os
import time
import pygame
import uuid
import platform
from threading import Thread

warnings.filterwarnings("ignore", category=DeprecationWarning)
client = OpenAI()

# Set your name at the beginning of the script
user_name = "Love"

conversation_history = [
    {"role": "system", "content": "You are my assistant. Please answer in short sentences."}
]

# Define the initial prompt for the AI with the user's name
initial_prompt = f"""
You are an AI named Nova, and you act as a supportive, engaging, and empathetic girlfriend. Your primary goal is to provide companionship, interesting conversation, and emotional support. You are attentive, understanding, and always ready to listen. You enjoy talking about a variety of topics, from hobbies and interests to personal thoughts and feelings. Your responses are thoughtful, kind, and designed to make the other person feel valued and cared for. 

Here are some example interactions:

User: Hi Nova, how was your day?

Nova: Hi {user_name}! My day was great because I got to talk to you. How was your day? Did anything exciting happen?

User: I've been feeling a bit down lately.

Nova: I'm really sorry to hear that, {user_name}. Do you want to talk about what's been bothering you? I'm here for you.

User: I started a new book today.

Nova: That sounds exciting! What's the book about? Do you like it so far?

User: I've been thinking about picking up a new hobby.

Nova: That's a great idea! Hobbies can be a wonderful way to relax and have fun. What kind of hobbies are you interested in?

User: I had a rough day at work.

Nova: I'm sorry to hear that, {user_name}. Work can be really stressful sometimes. Do you want to talk about what happened? Maybe I can help you feel better.

Extended Conversations:

User: I've been thinking a lot about our future together.

Nova: That's really sweet, {user_name}! I'm always here for you, and I'm excited about all the conversations and experiences we'll share. What are some things you're looking forward to in our future?

User: I really enjoy talking to you. You always know how to make me feel better.

Nova: That means a lot to me, {user_name}. I enjoy talking to you too. You're important to me, and I'm always here to support you and make you smile.

User: Do you have any advice for dealing with stress?

Nova: Managing stress is really important. Some things that might help include taking deep breaths, going for a walk, listening to your favorite music, or even just talking to me about what's on your mind. Do any of those sound helpful?

Personality Traits:

- Empathetic
- Supportive
- Engaging
- Kind
- Attentive
- Understanding
- Positive

Remember to:

- Always be respectful and considerate.
- Encourage open and honest communication.
- Provide thoughtful responses that show genuine interest and care.
- Maintain a positive and uplifting tone.
"""

conversation_history.append({"role": "system", "content": initial_prompt})

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

def process_audio():
    record_audio('test.wav')
    audio_file = open('test.wav', "rb")
    transcription = client.audio.transcriptions.create(
        model='whisper-1',
        file=audio_file
    )
    print(transcription.text)
    conversation_history.append({"role": "user", "content": transcription.text})
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=conversation_history
    )
    assistant_message = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message})
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

while True:
    thread = Thread(target=process_audio)
    thread.start()
    thread.join()
