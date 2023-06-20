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
    return "", history + [[user_message, None]]


def bot(history):
    message = history[-1][0]

    completion = openai.ChatCompletion.create(
        model=OPENAI_MODEL_ENGINE,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
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
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chatbot = gr.Chatbot()

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)


demo.queue()
demo.launch()
# demo.launch(server_port=8080)
# demo.launch()
