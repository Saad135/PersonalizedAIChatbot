import openai
import gradio as gr
import random
import time
import os


import numpy as np
import pandas as pd
from gradio_client import Client
from elevenlabs import voices, generate, set_api_key, UnauthenticatedRateLimitError
from dotenv import load_dotenv
from config import CONTEXT_DF_FILE, CONTEXT_EMBED_FILE, OPENAI_MODEL_ENGINE
from utils import (
    construct_prompt,
    get_channel_videos,
    get_transcripts,
    load_embeddings,
    merge_transcripts,
    pad_buffer,
)

load_dotenv()


openai.api_key = os.getenv("openai_api_key")
set_api_key(os.getenv("eleven_api_key"))


def user(user_message, history):
    return gr.update(value="", interactive=False), history + [[user_message, None]]


def bot(mood, history, context_embeds):
    header = f"""Answer the question in a {mood} way as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""

    message = history[-1][0]
    prompt = construct_prompt(message, context_embeds)

    completion = openai.ChatCompletion.create(
        model=OPENAI_MODEL_ENGINE,
        messages=[
            {
                "role": "system",
                "content": header,
            },
            {
                "role": "user",
                "content": prompt,
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


def convert_to_embeddings(playlist_id, num_vids, pr=gr.Progress()):
    video_ids = get_channel_videos(playlist_id, num_vids)
    transcripts = get_transcripts(video_ids, pr)
    merged_transcripts, merged_transcript_wo_embed = merge_transcripts(transcripts, pr)
    return (
        "Embeddings generated successfully",
        merged_transcripts,
        merged_transcript_wo_embed,
    )


# Without microphone streaming
with gr.Blocks() as demo:
    gr.Markdown(
        """
        ## Context from Youtube playlists
        """
    )
    playlist_id = gr.Textbox(
        label="Youtube playlist ID",
        info="Videos will be fetched from the playlist with the provided it and their transcripts will be used as context.",
    )
    num_videos = gr.Number(
        value=5,
        label="No. of videos",
        info="No. of videos whose transcripts will be fetched from the playlist",
    )
    convert_to_embeddings_btn = gr.Button("Convert")
    context_embeddings_state = gr.State()

    gr.Markdown(
        """
        ## Bot Configuration
        """
    )
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
        bot, [mood, chatbot, context_embeddings_state], chatbot
    )
    response.then(generate_voice, chatbot, bot_tts, queue=False)
    response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)

    gr.Markdown(
        """
        ## Retrieved Transcripts
        """
    )

    embedding_json = gr.JSON()
    convert_to_embeddings_btn.click(
        convert_to_embeddings,
        [playlist_id, num_videos],
        [playlist_id, context_embeddings_state, embedding_json],
    )

    clear.click(
        lambda: [None, None, None, None, None],
        None,
        [record, chatbot, bot_tts, playlist_id, embedding_json],
        queue=False,
    )


demo.queue()
demo.launch()
# demo.launch(server_port=8080)
# demo.launch()
