import openai
import gradio as gr
import random
import time
import os


import numpy as np
from gradio_client import Client
from elevenlabs import voices, generate, set_api_key, UnauthenticatedRateLimitError
from dotenv import load_dotenv
from config import OPENAI_MODEL_ENGINE

load_dotenv()


openai.api_key = os.getenv("openai_api_key")
set_api_key(os.getenv("eleven_api_key"))


def user(user_message, history):
    return gr.update(value="", interactive=False), history + [[user_message, None]]


def bot(mood, history):
    message = history[-1][0]

    completion = openai.ChatCompletion.create(
        model=OPENAI_MODEL_ENGINE,
        messages=[
            {
                "role": "system",
                "content": f"You are a helpful assistant who is very {mood}.",
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
        stream=True,
    )
    history[-1][1] = ""

    for chunk in completion:
        # print(chunk)
        history[-1][1] += (
            chunk.choices[0].delta.content
            if "content" in chunk.choices[0].delta.keys()
            else ""
        )
        # time.sleep(0.05)
        yield history


# Reference: https://github.com/gradio-app/gradio/discussions/3921#discussioncomment-5738136
def transcribe(audio):
    client = Client("abidlabs/whisper")
    text = client.predict(audio, api_name="/predict")
    return gr.update(value=text, interactive=True)


# Reference: https://huggingface.co/spaces/elevenlabs/tts/blob/main/app.py
def pad_buffer(audio):
    # Pad buffer to multiple of 2 bytes
    buffer_size = len(audio)
    element_size = np.dtype(np.int16).itemsize
    if buffer_size % element_size != 0:
        audio = audio + b"\0" * (element_size - (buffer_size % element_size))
    return audio


def generate_voice(history):
    text = history[-1][1]
    try:
        audio = generate(
            text,  # Limit to 250 characters
            voice="Arnold",
            model="eleven_monolingual_v1",
        )
        return (44100, np.frombuffer(pad_buffer(audio), dtype=np.int16))
    except UnauthenticatedRateLimitError as e:
        raise gr.Error(
            "Thanks for trying out ElevenLabs TTS! You've reached the free tier limit. Please provide an API key to continue."
        )
    except Exception as e:
        raise gr.Error(e)


# With microphone streaming
with gr.Blocks() as demo:
    mood = gr.Dropdown(
        ["cheerful", "pessimistic", "optimistic"],
        label="Bot Mood",
        info="Select the mood for the bot",
    )

    with gr.Row():
        with gr.Column(scale=3):
            record = gr.Audio(source="microphone", type="filepath")
        with gr.Column(scale=1):
            transcribe_btn = gr.Button("Submit Audio")

    msg = gr.Textbox()
    chatbot = gr.Chatbot()
    bot_tts = gr.Audio()
    clear = gr.Button("Clear")

    transcribe_btn.click(fn=transcribe, inputs=record, outputs=msg).then(
        lambda: gr.update(interactive=True), None, [msg], queue=False
    )

    response = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [mood, chatbot], chatbot
    )
    response.then(generate_voice, chatbot, bot_tts, queue=False)
    response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)

    clear.click(
        lambda: [None, None, None], None, [record, chatbot, bot_tts], queue=False
    )


demo.queue()
demo.launch()
# demo.launch(server_port=8080)
# demo.launch()
