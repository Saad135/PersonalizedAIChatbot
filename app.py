import openai
import gradio as gr
import random
import time
import os
from dotenv import load_dotenv
from config import OPENAI_MODEL_ENGINE

load_dotenv()


openai.api_key = os.getenv("openai_api_key")


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


with gr.Blocks() as demo:
    mood = gr.Dropdown(
        ["cheerful", "pessimistic", "optimistic"],
        label="Bot Mood",
        info="Select the mood for the bot",
    )
    msg = gr.Textbox()
    chatbot = gr.Chatbot()
    clear = gr.Button("Clear")

    response = msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [mood, chatbot], chatbot
    )
    response.then(lambda: gr.update(interactive=True), None, [msg], queue=False)

    clear.click(lambda: None, None, chatbot, queue=False)


demo.queue()
demo.launch()
# demo.launch(server_port=8080)
# demo.launch()
